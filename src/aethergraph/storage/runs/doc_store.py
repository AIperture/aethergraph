from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Any

from aethergraph.contracts.services.runs import RunStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus

# Generic DocStore-backed RunStore implementation


def _encode_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    # ISO-8601 string; JSON friendly
    return dt.isoformat()


def _decode_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    try:
        return datetime.fromisoformat(str(raw))
    except Exception:
        return None


def _encode_status(status: Any) -> str:
    """
    Normalize status to a plain string for storage.

    - RunStatus.succeeded -> "succeeded"
    - "RunStatus.succeeded" -> "succeeded"
    - "succeeded" -> "succeeded"
    """
    # If it's already the enum, use its value
    if isinstance(status, RunStatus):
        return status.value

    s = str(status)
    if s.startswith("RunStatus."):
        # tolerate older stored form
        return s.split(".", 1)[1]
    return s


def _decode_status(raw: Any) -> RunStatus:
    """
    Accept both 'succeeded' and 'RunStatus.succeeded' and normalize to RunStatus.
    """
    if isinstance(raw, RunStatus):
        return raw

    s = str(raw)
    if s.startswith("RunStatus."):
        s = s.split(".", 1)[1]
    return RunStatus(s)


def _runrecord_to_doc(record: RunRecord) -> dict[str, Any]:
    d = asdict(record)
    d["status"] = _encode_status(record.status)
    d["started_at"] = _encode_dt(record.started_at)
    d["finished_at"] = _encode_dt(record.finished_at)
    return d


def _doc_to_runrecord(doc: dict[str, Any]) -> RunRecord:
    return RunRecord(
        run_id=doc["run_id"],
        graph_id=doc["graph_id"],
        kind=doc.get("kind", "other"),
        status=_decode_status(doc.get("status")),
        started_at=_decode_dt(doc.get("started_at")) or datetime.utcnow(),
        finished_at=_decode_dt(doc.get("finished_at")),
        tags=list(doc.get("tags") or []),
        user_id=doc.get("user_id"),
        org_id=doc.get("org_id"),
        error=doc.get("error"),
        meta=dict(doc.get("meta") or {}),
    )


class DocRunStore(RunStore):
    """
    RunStore backed by an arbitrary DocStore.

    - Uses doc IDs like "<prefix><run_id>" (prefix defaults to "run:").
    - Persists RunRecord as JSON-friendly dicts (ISO datetimes, status as string).
    - Supports FS-backed or SQLite-backed DocStore transparently.

    The only requirement is that the underlying DocStore implements `list()`
    if you want `RunStore.list()` to work.
    """

    def __init__(self, doc_store: DocStore, *, prefix: str = "run:") -> None:
        self._ds = doc_store
        self._prefix = prefix
        self._lock = asyncio.Lock()

    def _doc_id(self, run_id: str) -> str:
        return f"{self._prefix}{run_id}"

    async def create(self, record: RunRecord) -> None:
        doc_id = self._doc_id(record.run_id)
        doc = _runrecord_to_doc(record)
        async with self._lock:
            await self._ds.put(doc_id, doc)

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        doc_id = self._doc_id(run_id)
        async with self._lock:
            doc = await self._ds.get(doc_id)
            if doc is None:
                # You could choose to create a minimal record here instead.
                return

            doc["status"] = str(status)
            if finished_at is not None:
                doc["finished_at"] = _encode_dt(finished_at)
            if error is not None:
                doc["error"] = error

            await self._ds.put(doc_id, doc)

    async def get(self, run_id: str) -> RunRecord | None:
        doc_id = self._doc_id(run_id)
        async with self._lock:
            doc = await self._ds.get(doc_id)
        if doc is None:
            return None
        return _doc_to_runrecord(doc)

    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
    ) -> list[RunRecord]:
        # Require DocStore.list; if not implemented, raise a clear error
        if not hasattr(self._ds, "list"):
            raise RuntimeError(
                "Underlying DocStore does not implement list(); " "cannot support RunStore.list()."
            )

        async with self._lock:
            doc_ids: list[str] = await self._ds.list()  # type: ignore[attr-defined]
            # Only consider docs under our prefix
            doc_ids = [d for d in doc_ids if d.startswith(self._prefix)]

            records: list[RunRecord] = []
            for doc_id in doc_ids:
                doc = await self._ds.get(doc_id)
                if not doc:
                    continue
                rec = _doc_to_runrecord(doc)

                if graph_id is not None and rec.graph_id != graph_id:
                    continue
                if status is not None and rec.status != status:
                    continue

                records.append(rec)

        # Sort newest first, then truncate
        records.sort(key=lambda r: r.started_at, reverse=True)
        return records[:limit]

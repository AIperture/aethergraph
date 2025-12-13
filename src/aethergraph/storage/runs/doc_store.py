from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Any

from aethergraph.contracts.services.runs import RunStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunStatus,
    RunVisibility,
)

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


def _encode_status(status: RunStatus) -> str:
    """
    Store RunStatus as its plain value ("running").
    """
    if isinstance(status, RunStatus):
        return status.value
    # tolerate weird callers, but normalize via str
    s = str(status)
    if s.startswith("RunStatus."):
        return s.split(".", 1)[1]
    return s


def _decode_status(raw: Any) -> RunStatus:
    """
    Decode status from:
      - RunStatus enum
      - "running"
      - "RunStatus.running"
    """
    if raw is None:
        # If we really want to be strict, we can raise instead. For now default:
        return RunStatus.pending

    if isinstance(raw, RunStatus):
        return raw

    s = str(raw)
    if s.startswith("RunStatus."):
        s = s.split(".", 1)[1]

    # This will raise ValueError if s is invalid, which is fine:
    return RunStatus(s)


def _encode_origin(origin: RunOrigin | None) -> str | None:
    if origin is None:
        return None
    if isinstance(origin, RunOrigin):
        return origin.value
    s = str(origin)
    if s.startswith("RunOrigin."):
        return s.split(".", 1)[1]
    return s


def _decode_origin(raw: Any) -> RunOrigin:
    if raw is None:
        # sensible default
        return RunOrigin.app
    if isinstance(raw, RunOrigin):
        return raw
    s = str(raw)
    if s.startswith("RunOrigin."):
        s = s.split(".", 1)[1]
    return RunOrigin(s)


def _encode_visibility(visibility: RunVisibility | None) -> str | None:
    if visibility is None:
        return None
    if isinstance(visibility, RunVisibility):
        return visibility.value
    s = str(visibility)
    if s.startswith("RunVisibility."):
        return s.split(".", 1)[1]
    return s


def _decode_visibility(raw: Any) -> RunVisibility:
    if raw is None:
        return RunVisibility.normal
    if isinstance(raw, RunVisibility):
        return raw
    s = str(raw)
    if s.startswith("RunVisibility."):
        s = s.split(".", 1)[1]
    return RunVisibility(s)


def _encode_importance(importance: RunImportance | None) -> str | None:
    if importance is None:
        return None
    if isinstance(importance, RunImportance):
        return importance.value
    s = str(importance)
    if s.startswith("RunImportance."):
        return s.split(".", 1)[1]
    return s


def _decode_importance(raw: Any) -> RunImportance:
    if raw is None:
        return RunImportance.normal
    if isinstance(raw, RunImportance):
        return raw
    s = str(raw)
    if s.startswith("RunImportance."):
        s = s.split(".", 1)[1]
    return RunImportance(s)


def _runrecord_to_doc(record: RunRecord) -> dict[str, Any]:
    d = asdict(record)
    d["status"] = _encode_status(record.status)
    d["started_at"] = _encode_dt(record.started_at)
    d["finished_at"] = _encode_dt(record.finished_at)
    d["origin"] = _encode_origin(record.origin)
    d["visibility"] = _encode_visibility(record.visibility)
    d["importance"] = _encode_importance(record.importance)
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
        session_id=doc.get("session_id"),
        origin=_decode_origin(doc.get("origin")),
        visibility=_decode_visibility(doc.get("visibility")),
        importance=_decode_importance(doc.get("importance")),
        app_id=doc.get("app_id"),
        agent_id=doc.get("agent_id"),
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

            doc["status"] = _encode_status(status)
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
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        # NOTE: This implementation is fine for small/medium numbers of runs, but it:
        #   - Calls DocStore.list() to load ALL doc_ids
        #   - Loads each run doc and filters in Python
        #   - Sorts in memory by started_at
        # For large volumes / multi-tenant cloud use, replace this with a real DB-backed
        # RunStore that pushes filtering + sorting + LIMIT/OFFSET (or keyset) into SQL.
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
                if session_id is not None and rec.session_id != session_id:
                    continue

                records.append(rec)

        # Sort newest first, then truncate
        records.sort(key=lambda r: r.started_at, reverse=True)

        # apply offset
        if offset > 0:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

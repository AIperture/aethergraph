from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import Any

from aethergraph.contracts.services.runs import RunResultStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.core.runtime.run_types import RunResult, RunStatus


def _encode_dt(dt: datetime | None) -> str | None:
    if dt is None:
        return None
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


def _encode_result(result: RunResult) -> dict[str, Any]:
    data = asdict(result)
    data["status"] = result.status.value
    data["created_at"] = _encode_dt(result.created_at)
    data["updated_at"] = _encode_dt(result.updated_at)
    return data


def _decode_result(doc: dict[str, Any]) -> RunResult:
    return RunResult(
        run_id=str(doc["run_id"]),
        graph_id=str(doc["graph_id"]),
        session_id=doc.get("session_id"),
        status=RunStatus(str(doc.get("status") or RunStatus.succeeded.value)),
        outputs=dict(doc.get("outputs") or {}),
        created_at=_decode_dt(doc.get("created_at")) or datetime.utcnow(),
        updated_at=_decode_dt(doc.get("updated_at")) or datetime.utcnow(),
        source=str(doc.get("source") or "direct"),
        snapshot_rev=doc.get("snapshot_rev"),
    )


class InMemoryRunResultStore(RunResultStore):
    def __init__(self) -> None:
        self._results: dict[str, RunResult] = {}
        self._lock = asyncio.Lock()

    async def save(self, run_id: str, result: RunResult) -> None:
        async with self._lock:
            self._results[run_id] = RunResult(**asdict(result))

    async def get(self, run_id: str) -> RunResult | None:
        async with self._lock:
            result = self._results.get(run_id)
            if result is None:
                return None
            return RunResult(**asdict(result))

    async def delete(self, run_id: str) -> None:
        async with self._lock:
            self._results.pop(run_id, None)


class DocRunResultStore(RunResultStore):
    def __init__(self, doc_store: DocStore, *, prefix: str = "run-result:") -> None:
        self._ds = doc_store
        self._prefix = prefix
        self._lock = asyncio.Lock()

    def _doc_id(self, run_id: str) -> str:
        return f"{self._prefix}{run_id}"

    async def save(self, run_id: str, result: RunResult) -> None:
        async with self._lock:
            await self._ds.put(self._doc_id(run_id), _encode_result(result))

    async def get(self, run_id: str) -> RunResult | None:
        async with self._lock:
            doc = await self._ds.get(self._doc_id(run_id))
        if doc is None:
            return None
        return _decode_result(doc)

    async def delete(self, run_id: str) -> None:
        async with self._lock:
            await self._ds.delete(self._doc_id(run_id))

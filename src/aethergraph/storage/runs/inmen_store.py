from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime

from aethergraph.contracts.services.runs import RunStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus


class InMemoryRunStore(RunStore):
    """
    Simple in-memory RunStore useful for sidecar/server default.

    Not persisted across process restarts.
    """

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: RunRecord) -> None:
        async with self._lock:
            self._records[record.run_id] = record

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: datetime | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            rec = self._records.get(run_id)
            if rec is None:
                # Optionally: create a minimal record; for now, just ignore.
                return
            rec.status = status
            if finished_at is not None:
                rec.finished_at = finished_at
            if error is not None:
                rec.error = error

    async def get(self, run_id: str) -> RunRecord | None:
        async with self._lock:
            rec = self._records.get(run_id)
            if rec is None:
                return None
            # return a deep copy to avoid external mutation of internal state
            return RunRecord(**asdict(rec))

    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        # NOTE: InMemoryRunStore is for dev/sidecar/demo only.
        # It scans all in-memory records and sorts in Python.
        # Do NOT use this in any environment where run counts can grow large;
        # prefer DocRunStore + SQLite/FS or a proper DB-backed RunStore.
        async with self._lock:
            records: list[RunRecord] = list(self._records.values())
            if graph_id is not None:
                records = [r for r in records if r.graph_id == graph_id]
            if status is not None:
                records = [r for r in records if r.status == status]

            records = sorted(records, key=lambda r: r.started_at, reverse=True)

            if offset > 0:
                records = records[offset:]
            if limit is not None:
                records = records[:limit]
            # return copies
            return [RunRecord(**asdict(r)) for r in records]

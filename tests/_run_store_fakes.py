"""Protocol fakes isolating RunManager unit behavior from provider persistence."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from aethergraph.core.runtime.run_types import RunRecord, RunResult, RunStatus


class RunStoreFake:
    def __init__(self, *_: object, **__: object) -> None:
        self._records: dict[str, RunRecord] = {}

    async def create(self, record: RunRecord) -> None:
        self._records[record.run_id] = RunRecord(**asdict(record))

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at=None,
        error: str | None = None,
        meta_update: dict[str, Any] | None = None,
        field_updates: dict[str, Any] | None = None,
    ) -> None:
        record = self._records.get(run_id)
        if record is None:
            return
        record.status = status
        if finished_at is not None:
            record.finished_at = finished_at
        if error is not None:
            record.error = error
        if meta_update:
            record.meta.update(meta_update)
        for key, value in (field_updates or {}).items():
            setattr(record, key, value)

    async def get(self, run_id: str) -> RunRecord | None:
        record = self._records.get(run_id)
        return RunRecord(**asdict(record)) if record is not None else None

    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        records = list(self._records.values())
        if graph_id is not None:
            records = [item for item in records if item.graph_id == graph_id]
        if status is not None:
            records = [item for item in records if item.status == status]
        if session_id is not None:
            records = [item for item in records if item.session_id == session_id]
        records.sort(key=lambda item: item.started_at, reverse=True)
        return [RunRecord(**asdict(item)) for item in records[offset : offset + limit]]


class RunResultStoreFake:
    def __init__(self) -> None:
        self._results: dict[str, RunResult] = {}

    async def save(self, run_id: str, result: RunResult) -> None:
        self._results[run_id] = RunResult(**asdict(result))

    async def get(self, run_id: str) -> RunResult | None:
        result = self._results.get(run_id)
        return RunResult(**asdict(result)) if result is not None else None

    async def delete(self, run_id: str) -> None:
        self._results.pop(run_id, None)

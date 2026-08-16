from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from aethergraph.contracts.services.runs import RunStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus


class InMemoryRunStore(RunStore):
    """
    Simple in-memory RunStore useful for sidecar/server default.

    Not persisted across process restarts.
    """

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._artifact_occurrences: dict[str, dict[str, tuple[str, datetime]]] = {}
        self._lock = asyncio.Lock()

    async def create(self, record: RunRecord) -> None:
        async with self._lock:
            self._records[record.run_id] = record
            self._artifact_occurrences[record.run_id] = {}

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: datetime | None = None,
        error: str | None = None,
        meta_update: dict[str, Any] | None = None,
        field_updates: dict[str, Any] | None = None,
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
            if meta_update:
                rec.meta.update(meta_update)
            if field_updates:
                for key, value in field_updates.items():
                    setattr(rec, key, value)

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
        session_id: str | None = None,
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
            if session_id is not None:
                records = [r for r in records if r.session_id == session_id]

            records = sorted(records, key=lambda r: r.started_at, reverse=True)

            if offset > 0:
                records = records[offset:]
            if limit is not None:
                records = records[:limit]
            # return copies
            return [RunRecord(**asdict(r)) for r in records]

    async def record_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one stable in-memory run artifact occurrence.

        The receipt preserves separate content and occurrence identities while
        updating the frozen run preview fields.

        Examples:
            Count an occurrence:
                ```python
                await runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

            Replay the occurrence:
                ```python
                await runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

        Args:
            run_id: Exact run identity to update.
            artifact_id: Stable content artifact identity.
            occurrence_id: Stable occurrence idempotency identity.
            created_at: Optional occurrence time; defaults to current UTC.

        Returns:
            None: The occurrence was counted, replayed, or its run was absent.

        Notes:
            Reusing an occurrence with different content or time raises `ValueError`.
        """
        _artifact_identity(artifact_id, occurrence_id)
        occurred_at = created_at or datetime.now(UTC)
        async with self._lock:
            record = self._records.get(run_id)
            if record is None:
                return
            receipts = self._artifact_occurrences.setdefault(run_id, {})
            previous = receipts.get(occurrence_id)
            if previous is not None:
                if previous != (artifact_id, occurred_at):
                    raise ValueError("Run artifact occurrence identity conflicts")
                return
            receipts[occurrence_id] = (artifact_id, occurred_at)
            record.artifact_count += 1
            record.first_artifact_at = min(
                occurred_at,
                record.first_artifact_at or occurred_at,
            )
            record.last_artifact_at = max(
                occurred_at,
                record.last_artifact_at or occurred_at,
            )
            record.recent_artifact_ids = [
                *record.recent_artifact_ids,
                artifact_id,
            ][-10:]


def _artifact_identity(artifact_id: str, occurrence_id: str) -> None:
    for name, value in (("artifact_id", artifact_id), ("occurrence_id", occurrence_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

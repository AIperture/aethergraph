from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from aethergraph.core.runtime.run_types import RunRecord, RunResult, RunStatus


class RunStore(Protocol):
    """
    Abstract interface for storing run metadata.

    Implementations can be in-memory, file-based, or backed by a DB.
    """

    async def create(self, record: RunRecord) -> None: ...
    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: datetime | None = None,
        error: str | None = None,
        meta_update: dict[str, Any] | None = None,
        field_updates: dict[str, Any] | None = None,
    ) -> None: ...
    async def get(self, run_id: str) -> RunRecord | None: ...
    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]: ...

    async def record_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """
        Update artifact-related metadata for a run:

          - increment artifact_count
          - update first_artifact_at / last_artifact_at
          - optionally maintain recent_artifact_ids (bounded list)

        No-op if run_id does not exist.
        """


class RunResultStore(Protocol):
    """Abstract interface for durable succeeded-run outputs keyed by run_id."""

    async def save(self, run_id: str, result: RunResult) -> None: ...
    async def get(self, run_id: str) -> RunResult | None: ...
    async def delete(self, run_id: str) -> None: ...

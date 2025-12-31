from __future__ import annotations

from datetime import datetime
from typing import Protocol

from aethergraph.core.runtime.run_types import RunRecord, RunStatus


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

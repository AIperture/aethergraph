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
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one exact artifact occurrence and retain its content identity.

        Occurrence identity makes retries idempotent while artifact identity remains
        available for the bounded recent-artifact preview.

        Examples:
            Count an artifact occurrence:
                ```python
                await runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

            Replay the same occurrence:
                ```python
                await runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

        Args:
            run_id: Exact run identity to update.
            artifact_id: Stable content artifact identity retained for presentation.
            occurrence_id: Stable occurrence idempotency identity.
            created_at: Optional occurrence time; defaults to current UTC.

        Returns:
            None: The occurrence was counted, replayed, or its run was absent.

        Notes:
            Reusing one occurrence identity with different content or time fails.
        """


class RunResultStore(Protocol):
    """Abstract interface for durable succeeded-run outputs keyed by run_id."""

    async def save(self, run_id: str, result: RunResult) -> None: ...
    async def get(self, run_id: str) -> RunResult | None: ...
    async def delete(self, run_id: str) -> None: ...

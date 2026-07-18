from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aethergraph.services.llm.correlation import complete_llm_call_correlation

from .models import LLMObservationRecord, ObservationFilter, ObservationRecord, PurgeResult
from .policy import ObservationPolicy
from .sqlite_store import SQLiteObservationStore


class ObservabilityFacade:
    """Coordinate the one supported AG observation read/write boundary.

    Intro:
        Exposes canonical observation, LLM, lifecycle, and retention operations.

    Examples:
        Append a structured observation:
        ```python
        await facade.append_observation(record)
        ```

        Purge one run after reviewing a dry run:
        ```python
        preview = await facade.delete_run_observations("run-1", dry_run=True)
        if preview.matching_observations:
            await facade.delete_run_observations("run-1")
        ```

    Args:
        store: Concrete SQLite observation store owned by this facade.

    Returns:
        ObservabilityFacade: A facade over canonical observations and prompts.

    Notes:
        This facade does not read legacy JSONL or engine trace stores.
    """

    def __init__(self, store: SQLiteObservationStore) -> None:
        self.store = store

    async def close(self) -> None:
        await self.store.close()

    async def emit(self, record: LLMObservationRecord, *, capture_mode: str) -> None:
        if capture_mode != self.store.policy.capture_mode:
            raise ValueError("LLM client capture mode does not match the observation store policy")
        await self.store.append_llm_call(record)
        complete_llm_call_correlation(
            record.llm_call_id,
            prompt_manifest_id=record.prompt_manifest_id,
        )

    async def append_observation(
        self,
        record: ObservationRecord,
        *,
        resource_links: Iterable[dict[str, Any]] = (),
    ) -> str:
        return await self.store.append_observation(record, resource_links=resource_links)

    async def get_observation(self, observation_id: str) -> dict[str, Any] | None:
        return await self.store.get_observation(observation_id)

    async def list_observations(
        self, filters: ObservationFilter | None = None, *, offset: int = 0
    ) -> list[dict[str, Any]]:
        return await self.store.list_observations(filters, offset=offset)

    async def get_llm_call(self, llm_call_id: str) -> dict[str, Any] | None:
        return await self.store.get_llm_call(llm_call_id)

    async def list_llm_calls(self, **filters: Any) -> list[dict[str, Any]]:
        return await self.store.query_llm_calls(**filters)

    async def get_trace(self, trace_id: str) -> list[dict[str, Any]]:
        return await self.store.list_observations(ObservationFilter(trace_id=trace_id))

    async def list_traces(self, filters: ObservationFilter | None = None) -> list[str]:
        rows = await self.store.list_observations(filters)
        return sorted({str(row["trace_id"]) for row in rows if row.get("trace_id")})

    async def update_trace_management(self, scope_key: str, **changes: Any) -> dict[str, Any]:
        return await self.store.update_trace_management(scope_key, **changes)

    async def delete_observation(self, observation_id: str) -> PurgeResult:
        return await self.store.delete_observation(observation_id)

    async def delete_trace(self, trace_id: str, *, dry_run: bool = False) -> PurgeResult:
        return await self.store.delete_trace(trace_id, dry_run=dry_run)

    async def delete_run_observations(self, run_id: str, *, dry_run: bool = False) -> PurgeResult:
        return await self.store.delete_run_observations(run_id, dry_run=dry_run)

    async def delete_session_observations(
        self, session_id: str, *, dry_run: bool = False
    ) -> PurgeResult:
        return await self.store.delete_session_observations(session_id, dry_run=dry_run)

    async def purge_observations(
        self, filters: ObservationFilter, *, dry_run: bool = True
    ) -> PurgeResult:
        return await self.store.purge_observations(filters, dry_run=dry_run)

    async def get_storage_stats(self):
        return await self.store.get_storage_stats()

    async def garbage_collect_fragments(self) -> int:
        return await self.store.garbage_collect_fragments()


def open_observability_facade(
    workspace_root: str | Path,
    *,
    read_only: bool = True,
    policy: ObservationPolicy | None = None,
) -> ObservabilityFacade:
    """Open the canonical observation store for one workspace.

    Intro:
        Resolves the single v2 database without creating a legacy read path.

    Examples:
        Open historical data read-only:
        ```python
        facade = open_observability_facade(".runtime/build-1")
        ```

        Open a writable local workspace:
        ```python
        facade = open_observability_facade(".runtime/local", read_only=False)
        ```

    Args:
        workspace_root: Existing AetherGraph workspace root.
        read_only: Whether writes must be rejected.
        policy: Optional capture policy for writable use.

    Returns:
        ObservabilityFacade: Facade bound to `events/observability.db`.

    Notes:
        Missing workspaces or databases fail directly; there is no legacy fallback.
    """
    root = Path(workspace_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return ObservabilityFacade(
        SQLiteObservationStore(
            root / "events" / "observability.db",
            read_only=read_only,
            policy=policy,
        )
    )

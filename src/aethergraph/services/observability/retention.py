from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .models import ObservationFilter, PurgeResult
from .sqlite_store import SQLiteObservationStore


@dataclass(frozen=True)
class RetentionPolicy:
    max_age_days: int = 30
    error_max_age_days: int = 90
    max_full_prompt_age_days: int = 3
    max_bytes_per_trace: int = 64 * 1024 * 1024
    max_total_bytes: int = 512 * 1024 * 1024
    max_retained_traces: int = 10_000
    max_retained_runs: int = 10_000
    max_observations_per_purge: int = 1_000


class RetentionJanitor:
    """Apply bounded age, size, count, and pin-aware retention rules.

    Intro:
        Runs one bounded cleanup pass at startup and at configured intervals.

    Examples:
        Run one startup pass:
        ```python
        await janitor.run_once()
        ```

        Run until service shutdown:
        ```python
        await janitor.run_forever(stop_event)
        ```

    Args:
        store: Writable observation store.
        policy: Retention thresholds.
        interval_seconds: Maximum delay between bounded passes.

    Returns:
        RetentionJanitor: Background retention coordinator.

    Notes:
        Pinned trace/run/session scopes are never evicted by automatic passes.
    """

    def __init__(
        self,
        store: SQLiteObservationStore,
        policy: RetentionPolicy,
        *,
        interval_seconds: int = 3_600,
    ) -> None:
        self.store = store
        self.policy = policy
        self.interval_seconds = interval_seconds

    async def run_once(self, *, now: datetime | None = None) -> list[PurgeResult]:
        current = now or datetime.now(UTC)
        results = [
            await self.store.purge_observations(
                ObservationFilter(
                    capture_mode="full",
                    created_before=(
                        current - timedelta(days=self.policy.max_full_prompt_age_days)
                    ).isoformat(),
                    pinned=False,
                    limit=self.policy.max_observations_per_purge,
                ),
                dry_run=False,
            ),
            await self.store.purge_observations(
                ObservationFilter(
                    expired_before=current.isoformat(),
                    pinned=False,
                    limit=self.policy.max_observations_per_purge,
                ),
                dry_run=False,
            ),
            await self.store.purge_observations(
                ObservationFilter(
                    created_before=(current - timedelta(days=self.policy.max_age_days)).isoformat(),
                    exclude_severity="error",
                    pinned=False,
                    limit=self.policy.max_observations_per_purge,
                ),
                dry_run=False,
            ),
            await self.store.purge_observations(
                ObservationFilter(
                    created_before=(
                        current - timedelta(days=self.policy.error_max_age_days)
                    ).isoformat(),
                    pinned=False,
                    limit=self.policy.max_observations_per_purge,
                ),
                dry_run=False,
            ),
        ]
        trace_scopes = await self.store.list_scope_storage("trace_id")
        for scope in trace_scopes:
            if scope["pinned"]:
                continue
            if scope["logical_bytes"] > self.policy.max_bytes_per_trace:
                results.append(await self.store.delete_trace(scope["scope_id"]))
        for scope in trace_scopes[self.policy.max_retained_traces :]:
            if not scope["pinned"]:
                results.append(await self.store.delete_trace(scope["scope_id"]))
        run_scopes = await self.store.list_scope_storage("run_id")
        for scope in run_scopes[self.policy.max_retained_runs :]:
            if not scope["pinned"]:
                results.append(await self.store.delete_run_observations(scope["scope_id"]))
        stats = await self.store.get_storage_stats()
        if stats.logical_bytes > self.policy.max_total_bytes:
            results.append(
                await self.store.purge_observations(
                    ObservationFilter(
                        pinned=False,
                        target_reclaimed_bytes=(stats.logical_bytes - self.policy.max_total_bytes),
                        limit=self.policy.max_observations_per_purge,
                    ),
                    dry_run=False,
                )
            )
        return results

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue

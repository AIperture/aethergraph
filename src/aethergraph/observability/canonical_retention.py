"""Inactive provider-neutral observation retention projection for the S9 cut."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from aethergraph.services.canonical_storage_scope import merge_storage_scope
from aethergraph.storage.contracts import (
    ObservationCaptureMode,
    ObservationPurgeRequest,
    ObservationPurgeResult,
    ObservationScopeUsageQuery,
    ObservationSeverity,
    ObservationUsageDimension,
    PageRequest,
)

from .canonical_service import CanonicalObservationService
from .retention import RetentionPolicy

_PAGE_SIZE = 500


class ProviderRetentionJanitor:
    """Apply bounded retention rules through one canonical observation service."""

    def __init__(
        self,
        service: CanonicalObservationService,
        policy: RetentionPolicy,
        *,
        interval_seconds: int = 3_600,
        scope_action_limit: int = 100,
    ) -> None:
        """Bind bounded retention work to one provider-neutral service.

        Construction captures thresholds and validates background/action bounds but
        performs no storage I/O or lifecycle operation.

        Examples:
            Bind the default retention policy:
                ```python
                janitor = ProviderRetentionJanitor(service, RetentionPolicy())
                ```

            Bound each pass more tightly:
                ```python
                janitor = ProviderRetentionJanitor(
                    service,
                    policy,
                    interval_seconds=60,
                    scope_action_limit=10,
                )
                ```

        Args:
            service: Canonical observation service and fixed owner scope.
            policy: Age, logical-byte, and retained-scope thresholds.
            interval_seconds: Positive maximum delay between background passes.
            scope_action_limit: Positive maximum scope purges attempted per pass.

        Returns:
            None: The inactive-until-S9 janitor is ready.

        Notes:
            The service bundle owns repository lifecycle; the janitor never compacts
            provider-private files or opens storage.
        """
        if isinstance(interval_seconds, bool) or interval_seconds < 1:
            raise ValueError("interval_seconds must be positive")
        if isinstance(scope_action_limit, bool) or scope_action_limit < 1:
            raise ValueError("scope_action_limit must be positive")
        _validate_policy(policy)
        self.service = service
        self.policy = policy
        self.interval_seconds = interval_seconds
        self.scope_action_limit = scope_action_limit

    async def run_once(self, *, now: datetime | None = None) -> list[ObservationPurgeResult]:
        """Run one bounded age, expiry, scope, and capacity retention pass.

        Every deletion is selected and pin-checked by the canonical provider. Scope
        usage is traversed in bounded cursor pages and work stops at the action limit.

        Examples:
            Run using the current UTC time:
                ```python
                results = await janitor.run_once()
                ```

            Run deterministically in a test:
                ```python
                results = await janitor.run_once(now=fixed_now)
                ```

        Args:
            now: Optional timezone-aware reference time; current UTC when omitted.

        Returns:
            list[ObservationPurgeResult]: Ordered results for attempted bounded purges.

        Notes:
            A single pass may leave eligible rows for later passes when provider or
            scope-action bounds are reached. Pinned applicable scopes remain retained.
        """
        current = _utc(now or datetime.now(UTC))
        repository = self.service.repository
        scope = self.service.owner_scope
        maximum = self.policy.max_observations_per_purge
        results = [
            await repository.purge(
                ObservationPurgeRequest(
                    scope=scope,
                    dry_run=False,
                    capture_modes=(ObservationCaptureMode.FULL,),
                    occurred_before=current - timedelta(days=self.policy.max_full_prompt_age_days),
                    max_observations=maximum,
                )
            ),
            await repository.purge(
                ObservationPurgeRequest(
                    scope=scope,
                    dry_run=False,
                    expired_before=current,
                    max_observations=maximum,
                )
            ),
            await repository.purge(
                ObservationPurgeRequest(
                    scope=scope,
                    dry_run=False,
                    excluded_severities=(
                        ObservationSeverity.ERROR,
                        ObservationSeverity.CRITICAL,
                    ),
                    occurred_before=current - timedelta(days=self.policy.max_age_days),
                    max_observations=maximum,
                )
            ),
            await repository.purge(
                ObservationPurgeRequest(
                    scope=scope,
                    dry_run=False,
                    severities=(
                        ObservationSeverity.ERROR,
                        ObservationSeverity.CRITICAL,
                    ),
                    occurred_before=current - timedelta(days=self.policy.error_max_age_days),
                    max_observations=maximum,
                )
            ),
        ]
        remaining_actions = self.scope_action_limit
        trace_results, remaining_actions = await self._purge_scope_usage(
            dimension=ObservationUsageDimension.TRACE,
            retained_limit=self.policy.max_retained_traces,
            remaining_actions=remaining_actions,
            maximum_logical_bytes=self.policy.max_bytes_per_trace,
        )
        results.extend(trace_results)
        if remaining_actions:
            run_results, remaining_actions = await self._purge_scope_usage(
                dimension=ObservationUsageDimension.RUN,
                retained_limit=self.policy.max_retained_runs,
                remaining_actions=remaining_actions,
            )
            results.extend(run_results)
        stats = await repository.storage_stats(scope)
        if stats.logical_bytes > self.policy.max_total_bytes:
            results.append(
                await repository.purge(
                    ObservationPurgeRequest(
                        scope=scope,
                        dry_run=False,
                        max_observations=maximum,
                        target_reclaimed_bytes=(stats.logical_bytes - self.policy.max_total_bytes),
                    )
                )
            )
        return results

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Run bounded retention passes until the supplied stop event is set.

        The loop waits interruptibly between passes and returns promptly when service
        shutdown signals the event.

        Examples:
            Start background retention:
                ```python
                task = asyncio.create_task(janitor.run_forever(stop_event))
                ```

            Stop and join the worker:
                ```python
                stop_event.set()
                await task
                ```

        Args:
            stop_event: Event signaling permanent loop termination.

        Returns:
            None: The loop ended after the stop event was set.

        Notes:
            Provider failures propagate to the owning task; they are never converted
            into a silent retry or alternate persistence path.
        """
        while not stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                continue

    async def _purge_scope_usage(
        self,
        *,
        dimension: ObservationUsageDimension,
        retained_limit: int,
        remaining_actions: int,
        maximum_logical_bytes: int | None = None,
    ) -> tuple[list[ObservationPurgeResult], int]:
        results: list[ObservationPurgeResult] = []
        position = 0
        cursor: str | None = None
        page_budget = (retained_limit + _PAGE_SIZE - 1) // _PAGE_SIZE + remaining_actions
        while remaining_actions and page_budget:
            page_budget -= 1
            query = ObservationScopeUsageQuery(
                scope=self.service.owner_scope,
                dimension=dimension,
                page=PageRequest(limit=_PAGE_SIZE, cursor=cursor),
            )
            page = await self.service.repository.query_scope_usage(query)
            if not page.items:
                break
            for usage in page.items:
                over_count = position >= retained_limit
                over_bytes = (
                    maximum_logical_bytes is not None
                    and usage.logical_bytes > maximum_logical_bytes
                )
                position += 1
                if usage.pinned or not (over_count or over_bytes):
                    continue
                purge_scope = self.service.owner_scope
                trace_id = None
                if dimension is ObservationUsageDimension.TRACE:
                    trace_id = usage.scope_id
                else:
                    purge_scope = merge_storage_scope(
                        self.service.owner_scope,
                        run_id=usage.scope_id,
                    )
                results.append(
                    await self.service.repository.purge(
                        ObservationPurgeRequest(
                            scope=purge_scope,
                            dry_run=False,
                            trace_id=trace_id,
                            max_observations=self.policy.max_observations_per_purge,
                        )
                    )
                )
                remaining_actions -= 1
                if not remaining_actions:
                    break
            if not remaining_actions or page.next_cursor is None:
                break
            cursor = page.next_cursor
        return results, remaining_actions


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("retention time must be timezone-aware")
    return value.astimezone(UTC)


def _validate_policy(policy: RetentionPolicy) -> None:
    positive = (
        "max_age_days",
        "error_max_age_days",
        "max_full_prompt_age_days",
        "max_bytes_per_trace",
        "max_total_bytes",
    )
    for name in positive:
        value = getattr(policy, name)
        if isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_retained_traces", "max_retained_runs"):
        value = getattr(policy, name)
        if isinstance(value, bool) or not 0 <= value <= 1_000_000:
            raise ValueError(f"{name} must be between 0 and 1000000")
    maximum = policy.max_observations_per_purge
    if isinstance(maximum, bool) or not 1 <= maximum <= 10_000:
        raise ValueError("max_observations_per_purge must be between 1 and 10000")
    if policy.error_max_age_days < policy.max_age_days:
        raise ValueError("error_max_age_days must not be shorter than max_age_days")


__all__ = ["ProviderRetentionJanitor"]

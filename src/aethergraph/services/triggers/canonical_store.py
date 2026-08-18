"""Canonical trigger-store projection over provider storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    ClaimedTrigger,
    PageRequest,
    StorageBundle,
    StorageCapacityError,
    StorageConflictError,
    StorageScope,
    TriggerClaimRecord as CanonicalTriggerClaimRecord,
    TriggerClaimRequest,
    TriggerClaimStatus,
    TriggerKind as CanonicalTriggerKind,
    TriggerQuery,
    TriggerRecord as CanonicalTriggerRecord,
    TriggerRepository,
)

from .scheduling import _normalize_utc
from .types import TriggerClaim, TriggerRecord

_PAGE_SIZE = 100
_MAX_TRIGGERS = 1_000
_PUBLIC_METADATA = "public_metadata"
_SERVICE_CONTEXT = "service_context"
_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_REVISION_ATTR = "_canonical_storage_revision"


class CanonicalTriggerStore:
    """Project the frozen trigger store onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: TriggerRepository,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind legacy trigger operations to one provider-authoritative owner.

        Construction performs no storage I/O and captures neither a provider selector
        nor a physical path.

        Examples:
            Bind runtime trigger storage:
                ```python
                store = CanonicalTriggerStore(
                    repository=bundle.triggers,
                    owner_scope=owner_scope,
                    clock=clock,
                )
                ```

            Bind a deterministic fake repository:
                ```python
                store = CanonicalTriggerStore(
                    repository=fake_triggers,
                    owner_scope=StorageScope(project_id="project-1"),
                    clock=lambda: fixed_now,
                )
                ```

        Args:
            repository: Exact canonical trigger repository.
            owner_scope: Trusted provider ownership scope.
            clock: Timezone-aware UTC transition timestamp source.

        Returns:
            None: The provider-backed projection is ready.

        Notes:
            The bundle owns repository lifecycle; this projection has no close method.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope
        self._clock = clock

    async def create(self, trig: TriggerRecord) -> None:
        """Create one canonical trigger and attach its authoritative revision.

        The provider receives canonical owner/graph scope and normalized schedule
        fields, while launch-only context remains nested non-indexed service metadata.

        Examples:
            Create a scheduled trigger:
                ```python
                await store.create(trigger)
                ```

            Retry the identical creation safely:
                ```python
                await store.create(trigger)
                ```

        Args:
            trig: Frozen service-facing trigger record to project.

        Returns:
            None: The record was created or an identical create was replayed.

        Notes:
            Conflicting identity reuse propagates directly from the canonical store.
        """
        record = _to_canonical_create(trig, owner_scope=self._owner_scope)
        stored = await self._repository.create(record)
        _attach_revision(trig, stored.revision)

    async def update(self, trig: TriggerRecord) -> None:
        """Replace one trigger through exact provider revision CAS.

        Only records produced by this projection carry the required private revision;
        detached stale objects fail instead of becoming unconditional overwrites.

        Examples:
            Persist a canceled trigger:
                ```python
                trigger.active = False
                trigger.next_fire_at = None
                await store.update(trigger)
                ```

            Persist event delivery time:
                ```python
                trigger.last_fired_at = fired_at
                await store.update(trigger)
                ```

        Args:
            trig: Service-facing trigger carrying its canonical revision.

        Returns:
            None: The exact next revision was committed.

        Notes:
            Stale revisions, changed ownership, and missing records fail directly.
        """
        expected = _attached_revision(trig)
        current = await self._repository.get(self._owner_scope, trig.trigger_id)
        if current is None:
            raise StorageConflictError("Canonical trigger update target is missing")
        if current.revision != expected:
            raise StorageConflictError("Canonical trigger revision is stale")
        updated_at = max(current.updated_at, _utc(self._clock()))
        record = _to_canonical(
            trig,
            owner_scope=self._owner_scope,
            revision=expected + 1,
            created_at=current.created_at,
            updated_at=updated_at,
        )
        stored = await self._repository.compare_and_set(record, expected)
        _attach_revision(trig, stored.revision)

    async def get(self, trigger_id: str) -> TriggerRecord | None:
        """Read one owner-authorized trigger and restore its public projection.

        Provider scope never broadens after a miss. Deprecated App identity is
        restored only from a marked compatibility envelope.

        Examples:
            Read an existing trigger:
                ```python
                trigger = await store.get("trigger-1")
                ```

            Detect an unknown trigger:
                ```python
                assert await store.get("missing") is None
                ```

        Args:
            trigger_id: Exact stable trigger identity.

        Returns:
            TriggerRecord | None: Detached service projection or `None`.

        Notes:
            The lookup uses canonical owner scope, never App or client aliases.
        """
        record = await self._repository.get(self._owner_scope, trigger_id)
        return _to_legacy(record) if record is not None else None

    async def delete(self, trigger_id: str) -> None:
        """Delete one trigger at its exact current canonical revision.

        A missing or out-of-owner trigger preserves the frozen no-op behavior; a
        concurrent replacement is protected by repository CAS.

        Examples:
            Delete an existing trigger:
                ```python
                await store.delete("trigger-1")
                ```

            Delete an already absent trigger:
                ```python
                await store.delete("missing")
                ```

        Args:
            trigger_id: Exact stable trigger identity.

        Returns:
            None: The exact revision was deleted or the trigger was absent.

        Notes:
            Terminal occurrence receipts retain the canonical repository policy.
        """
        current = await self._repository.get(self._owner_scope, trigger_id)
        if current is None:
            return
        await self._repository.delete(current.scope, trigger_id, current.revision)

    async def claim_due(
        self,
        now: datetime,
        *,
        worker_id: str,
        lease_until: datetime,
        limit: int,
        skip_missed_before: datetime | None = None,
    ) -> list[TriggerClaim]:
        """Claim due triggers atomically inside the exact provider owner.

        Schedule advancement, missed receipts, and worker leases remain one provider
        transaction. A non-positive legacy limit returns an empty batch without I/O.

        Examples:
            Claim due work:
                ```python
                claims = await store.claim_due(
                    now, worker_id="worker-1", lease_until=lease_until, limit=100
                )
                ```

            Apply startup missed-work policy:
                ```python
                claims = await store.claim_due(
                    now,
                    worker_id="worker-1",
                    lease_until=lease_until,
                    limit=100,
                    skip_missed_before=started_at,
                )
                ```

        Args:
            now: Current claim instant.
            worker_id: Exact lease owner identity.
            lease_until: Lease expiration instant.
            limit: Maximum claims requested.
            skip_missed_before: Optional startup missed-work boundary.

        Returns:
            list[TriggerClaim]: Service-facing worker-owned claims.

        Notes:
            The scan is owner-scoped and never falls back to a global repository scan.
        """
        if limit <= 0:
            return []
        claimed = await self._repository.claim_due(
            TriggerClaimRequest(
                now=_utc(now),
                worker_id=worker_id,
                lease_until=_utc(lease_until),
                limit=limit,
                scope=self._owner_scope,
                skip_missed_before=(
                    _utc(skip_missed_before) if skip_missed_before is not None else None
                ),
            )
        )
        return [_to_legacy_claim(item) for item in claimed]

    async def complete_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        run_id: str,
        completed_at: datetime,
    ) -> bool:
        """Commit one worker-owned delivery receipt with its submitted run.

        The canonical repository updates the receipt and trigger last-fired revision
        atomically. Ownership loss retains the frozen `False` result.

        Examples:
            Complete a submitted run:
                ```python
                completed = await store.complete_claim(
                    fire_id, worker_id="worker-1", run_id="run-1", completed_at=now
                )
                ```

            Handle lost ownership:
                ```python
                if not await store.complete_claim(
                    fire_id, worker_id="stale", run_id="run-1", completed_at=now
                ):
                    return
                ```

        Args:
            fire_id: Stable trigger occurrence identity.
            worker_id: Expected current lease owner.
            run_id: Submitted run identity.
            completed_at: UTC delivery completion instant.

        Returns:
            bool: Whether the exact active lease became delivered.

        Notes:
            Only revision/ownership conflicts become `False`; other failures propagate.
        """
        return await self._transition_claim(
            fire_id,
            worker_id=worker_id,
            status=TriggerClaimStatus.DELIVERED,
            updated_at=_utc(completed_at),
            run_id=run_id,
        )

    async def fail_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        error: str,
        retry_at: datetime,
    ) -> bool:
        """Release one worker-owned claim into canonical retry backoff.

        The diagnostic is bounded to the frozen legacy limit before provider access,
        while the retry instant remains exact UTC.

        Examples:
            Schedule a retry:
                ```python
                released = await store.fail_claim(
                    fire_id, worker_id="worker-1", error="offline", retry_at=retry_at
                )
                ```

            Detect stale ownership:
                ```python
                assert not await store.fail_claim(
                    fire_id, worker_id="stale", error="offline", retry_at=retry_at
                )
                ```

        Args:
            fire_id: Stable trigger occurrence identity.
            worker_id: Expected current lease owner.
            error: Caller-authored failure diagnostic.
            retry_at: UTC next eligibility instant.

        Returns:
            bool: Whether the exact active lease became retryable.

        Notes:
            The projection performs no alternate claim write after a conflict.
        """
        return await self._transition_claim(
            fire_id,
            worker_id=worker_id,
            status=TriggerClaimStatus.RETRY,
            updated_at=_utc(self._clock()),
            retry_at=_utc(retry_at),
            last_error=(error[:2000] or "trigger claim failed"),
        )

    async def skip_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        reason: str,
        completed_at: datetime,
    ) -> bool:
        """Commit one worker-owned skipped occurrence receipt.

        The legacy reason becomes canonical skip evidence rather than a dynamic
        provider status value.

        Examples:
            Record an overlap skip:
                ```python
                skipped = await store.skip_claim(
                    fire_id, worker_id="worker-1", reason="overlap", completed_at=now
                )
                ```

            Detect stale ownership:
                ```python
                assert not await store.skip_claim(
                    fire_id, worker_id="stale", reason="overlap", completed_at=now
                )
                ```

        Args:
            fire_id: Stable trigger occurrence identity.
            worker_id: Expected current lease owner.
            reason: Non-empty terminal skip reason.
            completed_at: UTC skip completion instant.

        Returns:
            bool: Whether the exact active lease became skipped.

        Notes:
            Terminal receipts remain immutable canonical deduplication evidence.
        """
        return await self._transition_claim(
            fire_id,
            worker_id=worker_id,
            status=TriggerClaimStatus.SKIPPED,
            updated_at=_utc(completed_at),
            skip_reason=reason,
        )

    async def get_claim(self, fire_id: str) -> dict[str, Any] | None:
        """Read one owner-authorized claim through the frozen mapping shape.

        Canonical status and timestamps are projected to the fields exposed by the
        existing trigger engine and tests without leaking provider rows.

        Examples:
            Inspect a delivery receipt:
                ```python
                receipt = await store.get_claim("fire-1")
                ```

            Detect an unknown occurrence:
                ```python
                assert await store.get_claim("missing") is None
                ```

        Args:
            fire_id: Stable trigger occurrence identity.

        Returns:
            dict[str, Any] | None: Frozen compatibility mapping or `None`.

        Notes:
            Reading a claim never renews, retries, or broadens its owner scope.
        """
        record = await self._repository.get_claim(self._owner_scope, fire_id)
        return _claim_mapping(record) if record is not None else None

    async def list_all(
        self,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        kind: str | None = None,
        active: bool | None = None,
    ) -> list[TriggerRecord]:
        """List a bounded owner-visible trigger projection.

        Canonical org/graph/kind/activity filters execute in the provider. Legacy
        user-or-client matching occurs only after bounded hydration because client is
        intentionally absent from canonical scope and indexes.

        Examples:
            List one owner:
                ```python
                rows = await store.list_all(org_id="org-1", user_id="user-1")
                ```

            List active interval triggers:
                ```python
                rows = await store.list_all(kind="interval", active=True)
                ```

        Args:
            org_id: Optional canonical organization filter.
            user_id: Optional frozen user-or-client compatibility filter.
            client_id: Optional launch-context compatibility filter.
            graph_id: Optional canonical graph filter.
            kind: Optional exact trigger-kind value.
            active: Optional activity filter.

        Returns:
            list[TriggerRecord]: Bounded service-facing trigger projections.

        Notes:
            More than `_MAX_TRIGGERS` fails explicitly; no unbounded list is restored.
        """
        scope = _query_scope(self._owner_scope, org_id=org_id, graph_id=graph_id)
        kinds = (CanonicalTriggerKind(kind),) if kind is not None else ()
        records = await self._query_all(TriggerQuery(scope=scope, kinds=kinds, active=active))
        rows = [_to_legacy(record) for record in records]
        return [
            row
            for row in rows
            if (user_id is None or row.user_id == user_id or row.client_id == user_id)
            and (client_id is None or row.client_id == client_id)
        ]

    async def list_by_event_key(
        self,
        event_key: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> list[TriggerRecord]:
        """List active event triggers inside one explicit tenant context.

        Event key, kind, activity, and canonical organization filter in the provider;
        client compatibility is evaluated only across the bounded returned page set.

        Examples:
            Resolve one user's event triggers:
                ```python
                rows = await store.list_by_event_key(
                    "invoice.paid", org_id="org-1", user_id="user-1"
                )
                ```

            Resolve a client-authored compatibility event:
                ```python
                rows = await store.list_by_event_key(
                    "invoice.paid", client_id="client-1"
                )
                ```

        Args:
            event_key: Exact event routing key.
            org_id: Optional canonical organization filter.
            user_id: Optional frozen user-or-client compatibility filter.
            client_id: Optional launch-context compatibility filter.

        Returns:
            list[TriggerRecord]: Matching active event-trigger projections.

        Notes:
            At least one tenant field is required; no global fallback is attempted.
        """
        if org_id is None and user_id is None and client_id is None:
            raise ValueError("Event trigger reads require an explicit tenant scope")
        scope = _query_scope(self._owner_scope, org_id=org_id)
        records = await self._query_all(
            TriggerQuery(
                scope=scope,
                kinds=(CanonicalTriggerKind.EVENT,),
                active=True,
                event_key=event_key,
            )
        )
        rows = [_to_legacy(record) for record in records]
        return [
            row
            for row in rows
            if (user_id is None or row.user_id == user_id or row.client_id == user_id)
            and (client_id is None or row.client_id == client_id)
        ]

    async def _query_all(self, query: TriggerQuery) -> list[CanonicalTriggerRecord]:
        records: list[CanonicalTriggerRecord] = []
        cursor: str | None = None
        while True:
            page = await self._repository.query(
                replace(query, page=PageRequest(limit=_PAGE_SIZE, cursor=cursor))
            )
            records.extend(page.items)
            if len(records) > _MAX_TRIGGERS or (
                len(records) == _MAX_TRIGGERS and page.next_cursor is not None
            ):
                raise StorageCapacityError(
                    f"Canonical trigger query exceeds {_MAX_TRIGGERS} records"
                )
            cursor = page.next_cursor
            if cursor is None:
                return records

    async def _transition_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        status: TriggerClaimStatus,
        updated_at: datetime,
        run_id: str | None = None,
        retry_at: datetime | None = None,
        last_error: str | None = None,
        skip_reason: str | None = None,
    ) -> bool:
        current = await self._repository.get_claim(self._owner_scope, fire_id)
        if (
            current is None
            or current.status is not TriggerClaimStatus.LEASED
            or current.worker_id != worker_id
        ):
            return False
        record = replace(
            current,
            status=status,
            revision=current.revision + 1,
            updated_at=updated_at,
            worker_id=None,
            lease_until=None,
            retry_at=retry_at,
            run_id=run_id,
            last_error=last_error,
            skip_reason=skip_reason,
            finished_at=(
                updated_at
                if status in {TriggerClaimStatus.DELIVERED, TriggerClaimStatus.SKIPPED}
                else None
            ),
        )
        try:
            await self._repository.compare_and_set_claim(record, current.revision)
        except StorageConflictError:
            return False
        return True


def bind_canonical_trigger_store(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
) -> CanonicalTriggerStore:
    """Bind the frozen trigger service to the bundle's exact trigger field.

    The function constructs the active service projection from the already-open
    provider bundle without selecting storage or taking over lifecycle ownership.

    Examples:
        Bind production composition inputs:
            ```python
            triggers = bind_canonical_trigger_store(
                bundle=bundle, owner_scope=owner_scope, clock=clock
            )
            ```

        Bind a deterministic fake bundle:
            ```python
            triggers = bind_canonical_trigger_store(
                bundle=fake_bundle,
                owner_scope=StorageScope(project_id="project-1"),
                clock=lambda: fixed_now,
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        clock: Timezone-aware UTC transition timestamp source.

    Returns:
        CanonicalTriggerStore: Service-facing trigger-store projection.

    Notes:
        The binding performs no provider selection, I/O, fallback, or close operation.
    """
    return CanonicalTriggerStore(
        repository=bundle.triggers,
        owner_scope=owner_scope,
        clock=clock,
    )


def _to_canonical_create(
    trig: TriggerRecord,
    *,
    owner_scope: StorageScope,
) -> CanonicalTriggerRecord:
    created_at = _utc(trig.created_at)
    return _to_canonical(
        trig,
        owner_scope=owner_scope,
        revision=1,
        created_at=created_at,
        updated_at=created_at,
    )


def _to_canonical(
    trig: TriggerRecord,
    *,
    owner_scope: StorageScope,
    revision: int,
    created_at: datetime,
    updated_at: datetime,
) -> CanonicalTriggerRecord:
    if trig.graph_id is None:
        raise ValueError("Canonical triggers require graph_id")
    scope = merge_storage_scope(
        owner_scope,
        **{
            key: value
            for key, value in {
                "org_id": trig.org_id,
                "user_id": trig.user_id,
                "session_id": trig.session_id,
                "graph_id": trig.graph_id,
                "agent_id": trig.agent_id,
            }.items()
            if value is not None
        },
    )
    service_context = {
        key: value
        for key, value in {
            "client_id": trig.client_id,
            "mode": trig.mode,
            "memory_level": trig.memory_level,
        }.items()
        if value is not None
    }
    metadata: dict[str, Any] = {
        _PUBLIC_METADATA: _plain(trig.meta or {}),
        _SERVICE_CONTEXT: service_context,
    }
    if trig.app_id is not None:
        metadata[_COMPATIBILITY_METADATA] = {
            _DEPRECATED_APP_ID: {
                "value": str(trig.app_id),
                "deprecated": True,
                "scheduled_removal": "future breaking release",
            }
        }
    return CanonicalTriggerRecord(
        trigger_id=trig.trigger_id,
        graph_id=trig.graph_id,
        scope=scope,
        kind=CanonicalTriggerKind(str(trig.kind)),
        revision=revision,
        created_at=created_at,
        updated_at=updated_at,
        default_inputs=_plain(trig.default_inputs or {}),
        name=trig.trigger_name,
        origin=trig.origin,
        cron_expression=trig.cron_expr,
        interval_seconds=trig.interval_seconds,
        run_at=_utc_optional(trig.run_at),
        event_key=trig.event_key,
        timezone=trig.tz,
        max_overlap_runs=trig.max_overlap_runs,
        catch_up_missed=trig.catch_up_missed,
        active=trig.active,
        last_fired_at=_utc_optional(trig.last_fired_at),
        next_fire_at=_utc_optional(trig.next_fire_at),
        metadata=metadata,
    )


def _to_legacy(record: CanonicalTriggerRecord) -> TriggerRecord:
    metadata = _plain(record.metadata)
    public = metadata.get(_PUBLIC_METADATA)
    service = metadata.get(_SERVICE_CONTEXT)
    compatibility = metadata.get(_COMPATIBILITY_METADATA)
    if not isinstance(public, dict) or not isinstance(service, dict):
        raise ValueError("Canonical trigger service metadata is malformed")
    app_id = None
    if isinstance(compatibility, dict):
        app = compatibility.get(_DEPRECATED_APP_ID)
        if isinstance(app, dict) and app.get("deprecated") is True and app.get("value"):
            app_id = str(app["value"])
    trig = TriggerRecord(
        trigger_id=record.trigger_id,
        trigger_name=record.name,
        org_id=record.scope.org_id,
        user_id=record.scope.user_id,
        client_id=_optional_text(service.get("client_id")),
        mode=_optional_text(service.get("mode")),
        app_id=app_id,
        agent_id=record.scope.agent_id,
        session_id=record.scope.session_id,
        memory_level=service.get("memory_level"),
        graph_id=record.graph_id,
        default_inputs=_plain(record.default_inputs),
        origin=record.origin,
        kind=record.kind.value,
        cron_expr=record.cron_expression,
        interval_seconds=record.interval_seconds,
        run_at=record.run_at,
        event_key=record.event_key,
        tz=record.timezone,
        max_overlap_runs=record.max_overlap_runs,
        catch_up_missed=record.catch_up_missed,
        active=record.active,
        created_at=record.created_at,
        last_fired_at=record.last_fired_at,
        next_fire_at=record.next_fire_at,
        meta=public,
    )
    _attach_revision(trig, record.revision)
    return trig


def _to_legacy_claim(claimed: ClaimedTrigger) -> TriggerClaim:
    return TriggerClaim(
        fire_id=claimed.claim.fire_id,
        trigger=_to_legacy(claimed.trigger),
        scheduled_for=claimed.claim.scheduled_for,
        worker_id=str(claimed.claim.worker_id),
        lease_until=claimed.claim.lease_until,  # type: ignore[arg-type]
        attempts=claimed.claim.attempts,
        reclaimed=claimed.reclaimed,
    )


def _claim_mapping(record: CanonicalTriggerClaimRecord) -> dict[str, Any]:
    status = record.status.value
    if record.status is TriggerClaimStatus.SKIPPED:
        status = f"skipped_{record.skip_reason}"
    return {
        "fire_id": record.fire_id,
        "trigger_id": record.trigger_id,
        "scheduled_for": record.scheduled_for.timestamp(),
        "worker_id": record.worker_id,
        "status": status,
        "lease_until": record.lease_until.timestamp() if record.lease_until else None,
        "attempts": record.attempts,
        "retry_at": record.retry_at.timestamp() if record.retry_at else None,
        "run_id": record.run_id,
        "last_error": record.last_error,
        "updated_at": record.updated_at.timestamp(),
    }


def _query_scope(
    owner_scope: StorageScope,
    *,
    org_id: str | None = None,
    graph_id: str | None = None,
) -> StorageScope:
    return merge_storage_scope(
        owner_scope,
        **{
            key: value
            for key, value in {"org_id": org_id, "graph_id": graph_id}.items()
            if value is not None
        },
    )


def _attach_revision(trig: TriggerRecord, revision: int) -> None:
    setattr(trig, _REVISION_ATTR, revision)


def _attached_revision(trig: TriggerRecord) -> int:
    value = getattr(trig, _REVISION_ATTR, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise StorageConflictError(
            "Canonical trigger updates require a repository-authored revision"
        )
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _optional_text(value: Any) -> str | None:
    return str(value) if value is not None else None


def _utc(value: datetime) -> datetime:
    return _normalize_utc(value)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None

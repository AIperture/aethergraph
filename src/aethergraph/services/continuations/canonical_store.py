"""Canonical continuation and timer-lease projections over provider storage."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    ContinuationCorrelator as CanonicalCorrelator,
    ContinuationDraft as CanonicalDraft,
    ContinuationLeaseRecord as CanonicalLease,
    ContinuationLeaseRepository,
    ContinuationLeaseRequest,
    ContinuationLeaseStatus as CanonicalLeaseStatus,
    ContinuationQuery as CanonicalQuery,
    ContinuationRecord as CanonicalContinuation,
    ContinuationRepository,
    ContinuationStatus as CanonicalStatus,
    PageRequest,
    StorageBundle,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageScope,
)

from .continuation import (
    Continuation,
    ContinuationDraft,
    ContinuationPage,
    ContinuationQuery,
    ContinuationStatus,
    Correlator,
    CreatedContinuation,
)
from .timer_lease import TimerLease, TimerLeaseStatus

_SERVICE_CONTEXT = "service_context"
_NULL_FIELDS = "null_fields"
_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_OPTIONAL_MAPPING_FIELDS = ("resume_schema", "payload", "poll")


class CanonicalContinuationStore:
    """Project the frozen runtime continuation API onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: ContinuationRepository,
        owner_scope: StorageScope,
    ) -> None:
        """Bind runtime continuation operations to one provider-authoritative owner.

        Intro:
            Captures an already-open canonical repository and trusted owner scope
            without selecting providers or performing storage I/O.

        Examples:
            Bind a production repository:
            ```python
            store = CanonicalContinuationStore(
                repository=bundle.continuations, owner_scope=owner_scope
            )
            ```

            Bind a deterministic fake:
            ```python
            store = CanonicalContinuationStore(
                repository=fake_repository,
                owner_scope=StorageScope(project_id="project-1"),
            )
            ```

        Args:
            repository: Exact canonical continuation repository.
            owner_scope: Trusted provider ownership scope.

        Returns:
            None: The provider-backed projection is ready.

        Notes:
            The bundle owns lifecycle; this projection has no fallback or close path.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically create one continuation through the canonical repository.

        Intro:
            Projects tokenless runtime content and returns the provider-minted token
            only in the one-time creation envelope.

        Examples:
            Create a wait:
            ```python
            created = await store.create(draft)
            ```

            Register its stable identity:
            ```python
            future = waits.register(created.record.continuation_id)
            ```

        Args:
            draft: Immutable runtime continuation draft.

        Returns:
            CreatedContinuation: Tokenless revision-one record and one-time token.

        Notes:
            Deprecated App identity is written only in marked compatibility metadata.
        """
        created = await self._repository.create(
            _to_canonical_draft(draft, owner_scope=self._owner_scope)
        )
        return CreatedContinuation(record=_to_runtime(created.record), token=created.token)

    async def get(self, run_id: str, node_id: str) -> Continuation | None:
        """Read the sole continuation for an exact runtime run/node identity.

        Intro:
            Uses a bounded canonical scope query and rejects ambiguous persisted
            identities instead of guessing which record is authoritative.

        Examples:
            Read a wait:
            ```python
            wait = await store.get("run-1", "approval")
            ```

            Detect absence:
            ```python
            assert await store.get("missing", "node") is None
            ```

        Args:
            run_id: Exact run identity.
            node_id: Exact node identity.

        Returns:
            Continuation | None: Current tokenless record or `None`.

        Notes:
            More than one match is canonical integrity failure, never newest-record fallback.
        """
        page = await self._repository.query(
            CanonicalQuery(
                scope=_scope(self._owner_scope, run_id=run_id, node_id=node_id),
                statuses=tuple(CanonicalStatus),
                page=PageRequest(limit=2),
            )
        )
        if len(page.items) > 1:
            raise StorageIntegrityError(
                "Multiple continuations share one runtime run/node identity"
            )
        return _to_runtime(page.items[0]) if page.items else None

    async def get_by_id(
        self,
        run_id: str,
        node_id: str,
        continuation_id: str,
    ) -> Continuation | None:
        """Read one continuation by exact scope and stable identity.

        Intro:
            Resolves trusted internal delivery without consulting bearer-token indexes.

        Examples:
            Read a timer candidate:
            ```python
            wait = await store.get_by_id("run-1", "node-1", "cont-1")
            ```

            Detect absence:
            ```python
            assert await store.get_by_id("run-1", "node-1", "missing") is None
            ```

        Args:
            run_id: Exact run identity.
            node_id: Exact node identity.
            continuation_id: Stable continuation identity.

        Returns:
            Continuation | None: Current tokenless record or `None`.

        Notes:
            A miss never broadens beyond the supplied owner/run/node scope.
        """
        record = await self._repository.get(
            _scope(self._owner_scope, run_id=run_id, node_id=node_id),
            continuation_id,
        )
        return _to_runtime(record) if record is not None else None

    async def resolve_token(self, token: str) -> Continuation | None:
        """Resolve one bearer token while enforcing this projection's owner scope.

        Intro:
            Delegates secret lookup to the provider and rejects a valid token owned by
            another projection before returning runtime state.

        Examples:
            Resolve an inbound token:
            ```python
            wait = await store.resolve_token(token)
            ```

            Reject an unknown token:
            ```python
            assert await store.resolve_token("invalid") is None
            ```

        Args:
            token: Raw bearer token received at the trusted resume boundary.

        Returns:
            Continuation | None: Owner-authorized tokenless record or `None`.

        Notes:
            There is no run scan, token revalidation, or cross-owner fallback.
        """
        record = await self._repository.resolve_token(token)
        if record is None or not _owned_by(record.scope, self._owner_scope):
            return None
        return _to_runtime(record)

    async def update(
        self,
        continuation: Continuation,
        *,
        expected_revision: int,
    ) -> Continuation:
        """Replace one continuation through exact canonical revision CAS.

        Intro:
            Reloads the canonical record to preserve its secret digest and unknown
            provider metadata before committing the complete next revision.

        Examples:
            Reschedule a poll:
            ```python
            stored = await store.update(changed, expected_revision=current.revision)
            ```

            Detect a stale writer:
            ```python
            await store.update(changed, expected_revision=1)
            ```

        Args:
            continuation: Complete runtime next revision.
            expected_revision: Exact authoritative revision required for the update.

        Returns:
            Continuation: Newly stored tokenless record.

        Notes:
            Token digest, owner identity, creation time, and terminal lifecycle stay immutable.
        """
        scope = _scope_for_continuation(self._owner_scope, continuation)
        current = await self._repository.get(scope, continuation.continuation_id)
        if current is None:
            raise StorageNotFoundError(continuation.continuation_id)
        if current.revision != expected_revision:
            raise StorageConflictError("Continuation revision is stale")
        record = _to_canonical_record(
            continuation,
            owner_scope=self._owner_scope,
            token_digest=current.token_digest,
            base_metadata=current.metadata,
        )
        stored = await self._repository.compare_and_set(record, expected_revision)
        return _to_runtime(stored)

    async def close(
        self,
        continuation: Continuation,
        *,
        status: ContinuationStatus,
        closed_at: datetime,
    ) -> Continuation:
        """Transition one waiting continuation to a retained terminal receipt.

        Intro:
            Closes the provider-authoritative current revision while retaining durable
            identity and token-digest evidence.

        Examples:
            Mark successful delivery:
            ```python
            closed = await store.close(wait, status=ContinuationStatus.RESUMED, closed_at=now)
            ```

            Cancel a wait:
            ```python
            closed = await store.close(wait, status=ContinuationStatus.CANCELED, closed_at=now)
            ```

        Args:
            continuation: Exact current runtime continuation.
            status: Requested terminal lifecycle status.
            closed_at: Timezone-aware terminal transition time.

        Returns:
            Continuation: Newly stored or already-matching terminal record.

        Notes:
            Repeating the same terminal transition is idempotent; deletion is forbidden.
        """
        if status is ContinuationStatus.WAITING:
            raise ValueError("close requires a terminal continuation status")
        scope = _scope_for_continuation(self._owner_scope, continuation)
        current = await self._repository.get(scope, continuation.continuation_id)
        if current is None:
            raise StorageNotFoundError(continuation.continuation_id)
        canonical_status = CanonicalStatus(status.value)
        if current.status is not CanonicalStatus.WAITING:
            if current.status is canonical_status:
                return _to_runtime(current)
            raise StorageConflictError("Continuation is already terminal")
        if current.revision != continuation.revision:
            raise StorageConflictError("Continuation revision is stale")
        timestamp = _utc(closed_at)
        stored = await self._repository.compare_and_set(
            replace(
                current,
                revision=current.revision + 1,
                status=canonical_status,
                closed_at=timestamp,
            ),
            current.revision,
        )
        return _to_runtime(stored)

    async def bind_correlator(
        self,
        *,
        continuation: Continuation,
        corr: Correlator,
    ) -> Continuation:
        """Atomically bind one correlator through the canonical reverse index.

        Intro:
            Uses stable continuation identity and exact revision without bearer-token
            lookup or a second persistence path.

        Examples:
            Bind a public interaction:
            ```python
            wait = await store.bind_correlator(continuation=wait, corr=interaction)
            ```

            Replay an existing binding:
            ```python
            same = await store.bind_correlator(continuation=wait, corr=interaction)
            ```

        Args:
            continuation: Exact current runtime continuation.
            corr: Exact provider-neutral correlation identity.

        Returns:
            Continuation: Current or newly revised tokenless record.

        Notes:
            The canonical repository commits record and reverse index atomically.
        """
        stored = await self._repository.bind_correlator(
            _scope_for_continuation(self._owner_scope, continuation),
            continuation.continuation_id,
            _to_canonical_correlator(corr),
            continuation.revision,
        )
        return _to_runtime(stored)

    async def query(self, query: ContinuationQuery) -> ContinuationPage:
        """Execute one bounded indexed canonical continuation query.

        Intro:
            Maps runtime session, status, kind, correlator, due, and open filters
            directly to the canonical repository cursor contract.

        Examples:
            Query due timers:
            ```python
            page = await store.query(ContinuationQuery(due_at_or_before=now, limit=50))
            ```

            Query one interaction:
            ```python
            page = await store.query(ContinuationQuery(correlator=interaction, limit=2))
            ```

        Args:
            query: Exact runtime filters, bound, and optional cursor.

        Returns:
            ContinuationPage: Bounded tokenless records and next cursor.

        Notes:
            The projection performs no unbounded scan or client-side filtering.
        """
        dimensions = {"session_id": query.session_id} if query.session_id is not None else {}
        page = await self._repository.query(
            CanonicalQuery(
                scope=_scope(self._owner_scope, **dimensions),
                page=PageRequest(limit=query.limit, cursor=query.cursor),
                statuses=tuple(CanonicalStatus(value.value) for value in query.statuses),
                kinds=query.kinds,
                correlator=(
                    _to_canonical_correlator(query.correlator)
                    if query.correlator is not None
                    else None
                ),
                due_at_or_before=_utc_optional(query.due_at_or_before),
                open_at=_utc_optional(query.open_at),
            )
        )
        return ContinuationPage(
            items=tuple(_to_runtime(record) for record in page.items),
            next_cursor=page.next_cursor,
        )


class CanonicalContinuationLeaseStore:
    """Project runtime timer claims onto one canonical lease repository."""

    def __init__(
        self,
        *,
        repository: ContinuationLeaseRepository,
        owner_scope: StorageScope,
    ) -> None:
        """Bind timer delivery to one provider-authoritative owner.

        Intro:
            Captures an already-open canonical lease repository without performing I/O.

        Examples:
            Bind production leases:
            ```python
            store = CanonicalContinuationLeaseStore(
                repository=bundle.continuation_leases, owner_scope=owner_scope
            )
            ```

            Bind a fake repository:
            ```python
            store = CanonicalContinuationLeaseStore(
                repository=fake_leases,
                owner_scope=StorageScope(project_id="project-1"),
            )
            ```

        Args:
            repository: Exact canonical continuation-lease repository.
            owner_scope: Trusted provider ownership scope.

        Returns:
            None: The provider-backed lease projection is ready.

        Notes:
            Lease state is never persisted beside the selected canonical repository.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope

    async def claim(
        self,
        *,
        fire_id: str,
        continuation_id: str,
        run_id: str,
        node_id: str,
        scheduled_for: datetime,
        worker_id: str,
        now: datetime,
        lease_until: datetime,
    ) -> TimerLease | None:
        """Atomically claim one exact scheduled continuation occurrence.

        Intro:
            Delegates creation, retry, and stale-lease reclamation to one provider
            transaction and preserves exact reclaim evidence for observability.

        Examples:
            Claim a due fire:
            ```python
            lease = await store.claim(
                fire_id=fire_id, continuation_id=wait_id, run_id=run_id,
                node_id=node_id, scheduled_for=scheduled_for,
                worker_id=worker_id, now=now, lease_until=lease_until,
            )
            ```

            Detect contention:
            ```python
            assert await store.claim(**request) is None
            ```

        Args:
            fire_id: Stable scheduled occurrence identity.
            continuation_id: Stable continuation identity.
            run_id: Exact run identity.
            node_id: Exact node identity.
            scheduled_for: Exact timezone-aware occurrence time.
            worker_id: Claiming worker identity.
            now: Injected timezone-aware transition time.
            lease_until: Time after which another worker may reclaim.

        Returns:
            TimerLease | None: Worker-owned lease or `None` when ineligible.

        Notes:
            The returned `reclaimed` flag is provider-authored, not inferred from attempts.
        """
        claimed = await self._repository.claim(
            ContinuationLeaseRequest(
                fire_id=fire_id,
                continuation_id=continuation_id,
                scope=_scope(self._owner_scope, run_id=run_id, node_id=node_id),
                scheduled_for=_utc(scheduled_for),
                worker_id=worker_id,
                now=_utc(now),
                lease_until=_utc(lease_until),
            )
        )
        if claimed is None:
            return None
        return _to_runtime_lease(claimed.record, reclaimed=claimed.reclaimed)

    async def complete(self, lease: TimerLease, *, now: datetime) -> bool:
        """Atomically mark one exact worker-owned lease delivered.

        Intro:
            Converts an owned active lease to a retained canonical terminal receipt.

        Examples:
            Complete delivery:
            ```python
            changed = await store.complete(lease, now=now)
            ```

            Detect lost ownership:
            ```python
            assert not await store.complete(stale, now=now)
            ```

        Args:
            lease: Exact current runtime worker-owned lease.
            now: Injected timezone-aware completion time.

        Returns:
            bool: Whether the exact terminal transition committed.

        Notes:
            CAS conflict is reported as `False`, matching the frozen timer boundary.
        """
        current = await self._current_lease(lease)
        if not _owns_lease(current, lease):
            return False
        timestamp = _utc(now)
        record = replace(
            current,
            status=CanonicalLeaseStatus.DELIVERED,
            revision=current.revision + 1,
            updated_at=timestamp,
            worker_id=None,
            lease_until=None,
            next_attempt_at=None,
            last_error=None,
            finished_at=timestamp,
        )
        return await self._compare_and_set(record, current.revision)

    async def record_failure(
        self,
        lease: TimerLease,
        *,
        now: datetime,
        next_attempt_at: datetime | None,
        error: str,
        dead_letter: bool,
    ) -> bool:
        """Atomically record retry backoff or a terminal dead letter.

        Intro:
            Releases one exact owned lease without consuming or mutating its continuation.

        Examples:
            Schedule retry:
            ```python
            changed = await store.record_failure(
                lease, now=now, next_attempt_at=retry_at,
                error="unavailable", dead_letter=False,
            )
            ```

            Dead-letter exhaustion:
            ```python
            changed = await store.record_failure(
                lease, now=now, next_attempt_at=None,
                error="exhausted", dead_letter=True,
            )
            ```

        Args:
            lease: Exact current runtime worker-owned lease.
            now: Injected timezone-aware transition time.
            next_attempt_at: Next eligible time or `None` for dead letter.
            error: Non-empty bounded failure description.
            dead_letter: Whether to terminalize instead of retry.

        Returns:
            bool: Whether the exact owned-lease transition committed.

        Notes:
            Retry and dead-letter state remain canonical provider receipts.
        """
        current = await self._current_lease(lease)
        if not _owns_lease(current, lease):
            return False
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be a non-empty string")
        timestamp = _utc(now)
        retry_at = _utc_optional(next_attempt_at)
        if dead_letter and retry_at is not None:
            raise ValueError("dead-letter failure must not define next_attempt_at")
        if not dead_letter and retry_at is None:
            raise ValueError("retry failure requires next_attempt_at")
        record = replace(
            current,
            status=(
                CanonicalLeaseStatus.DEAD_LETTER if dead_letter else CanonicalLeaseStatus.RETRY
            ),
            revision=current.revision + 1,
            updated_at=timestamp,
            worker_id=None,
            lease_until=None,
            next_attempt_at=retry_at,
            last_error=error[:1000],
            finished_at=timestamp if dead_letter else None,
        )
        return await self._compare_and_set(record, current.revision)

    async def get(self, run_id: str, node_id: str, fire_id: str) -> TimerLease | None:
        """Read one exact canonical lease or retained receipt.

        Intro:
            Resolves diagnostic state within exact owner/run/node scope without
            changing worker ownership or retry eligibility.

        Examples:
            Read a receipt:
            ```python
            receipt = await store.get("run-1", "node-1", "fire-1")
            ```

            Detect absence:
            ```python
            assert await store.get("run-1", "node-1", "missing") is None
            ```

        Args:
            run_id: Exact run identity.
            node_id: Exact node identity.
            fire_id: Stable occurrence identity.

        Returns:
            TimerLease | None: Current lease or receipt, or `None`.

        Notes:
            Ordinary reads report `reclaimed=False`; only atomic claim authors that event.
        """
        record = await self._repository.get(
            _scope(self._owner_scope, run_id=run_id, node_id=node_id),
            fire_id,
        )
        return _to_runtime_lease(record) if record is not None else None

    async def _current_lease(self, lease: TimerLease) -> CanonicalLease | None:
        return await self._repository.get(
            _scope(self._owner_scope, run_id=lease.run_id, node_id=lease.node_id),
            lease.fire_id,
        )

    async def _compare_and_set(self, record: CanonicalLease, expected_revision: int) -> bool:
        try:
            await self._repository.compare_and_set(record, expected_revision)
        except (StorageConflictError, StorageNotFoundError):
            return False
        return True


def bind_canonical_continuation_store(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
) -> CanonicalContinuationStore:
    """Bind the runtime continuation API to the bundle's exact repository field.

    Intro:
        Constructs the active service projection from an already-open bundle.

    Examples:
        Bind production composition:
        ```python
        store = bind_canonical_continuation_store(
            bundle=bundle, owner_scope=owner_scope
        )
        ```

        Bind a test bundle:
        ```python
        store = bind_canonical_continuation_store(
            bundle=fake_bundle,
            owner_scope=StorageScope(project_id="project-1"),
        )
        ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.

    Returns:
        CanonicalContinuationStore: Frozen runtime continuation projection.

    Notes:
        Binding performs no provider selection, fallback, I/O, or close operation.
    """
    return CanonicalContinuationStore(
        repository=bundle.continuations,
        owner_scope=owner_scope,
    )


def bind_canonical_continuation_lease_store(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
) -> CanonicalContinuationLeaseStore:
    """Bind timer delivery to the bundle's exact canonical lease field.

    Intro:
        Constructs the active timer-lease projection from the same open bundle.

    Examples:
        Bind production composition:
        ```python
        leases = bind_canonical_continuation_lease_store(
            bundle=bundle, owner_scope=owner_scope
        )
        ```

        Bind a test bundle:
        ```python
        leases = bind_canonical_continuation_lease_store(
            bundle=fake_bundle,
            owner_scope=StorageScope(project_id="project-1"),
        )
        ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.

    Returns:
        CanonicalContinuationLeaseStore: Frozen runtime timer-lease projection.

    Notes:
        Binding performs no provider selection, fallback, I/O, or close operation.
    """
    return CanonicalContinuationLeaseStore(
        repository=bundle.continuation_leases,
        owner_scope=owner_scope,
    )


def _to_canonical_draft(
    draft: ContinuationDraft,
    *,
    owner_scope: StorageScope,
) -> CanonicalDraft:
    return CanonicalDraft(
        continuation_id=draft.continuation_id,
        kind=draft.kind,
        scope=_scope_for_draft(owner_scope, draft),
        created_at=_utc(draft.created_at),
        prompt=draft.prompt,
        resume_schema=_plain_mapping(draft.resume_schema),
        payload=_plain_mapping(draft.payload),
        poll_payload=_plain_mapping(draft.poll),
        metadata=_service_metadata(draft),
        deadline=_utc_optional(draft.deadline),
        next_wakeup_at=_utc_optional(draft.next_wakeup_at),
        channel=draft.channel,
        correlators=tuple(_to_canonical_correlator(value) for value in draft.correlators),
        attempts=draft.attempts,
    )


def _to_canonical_record(
    continuation: Continuation,
    *,
    owner_scope: StorageScope,
    token_digest: str,
    base_metadata: Mapping[str, object],
) -> CanonicalContinuation:
    return CanonicalContinuation(
        continuation_id=continuation.continuation_id,
        kind=continuation.kind,
        scope=_scope_for_continuation(owner_scope, continuation),
        created_at=_utc(continuation.created_at),
        token_digest=token_digest,
        revision=continuation.revision,
        status=CanonicalStatus(continuation.status.value),
        prompt=continuation.prompt,
        resume_schema=_plain_mapping(continuation.resume_schema),
        payload=_plain_mapping(continuation.payload),
        poll_payload=_plain_mapping(continuation.poll),
        metadata=_service_metadata(continuation, base=base_metadata),
        deadline=_utc_optional(continuation.deadline),
        next_wakeup_at=_utc_optional(continuation.next_wakeup_at),
        channel=continuation.channel,
        correlators=tuple(_to_canonical_correlator(value) for value in continuation.correlators),
        attempts=continuation.attempts,
        closed_at=_utc_optional(continuation.closed_at),
    )


def _to_runtime(record: CanonicalContinuation) -> Continuation:
    metadata = _plain(record.metadata)
    service = metadata.get(_SERVICE_CONTEXT, {})
    if not isinstance(service, dict):
        raise StorageIntegrityError("Canonical continuation service metadata is malformed")
    null_fields = service.get(_NULL_FIELDS, [])
    if not isinstance(null_fields, list) or any(
        not isinstance(value, str) for value in null_fields
    ):
        raise StorageIntegrityError("Canonical continuation null-field metadata is malformed")
    nulls = set(null_fields)
    compatibility = metadata.get(_COMPATIBILITY_METADATA)
    app_id = None
    if isinstance(compatibility, dict):
        app = compatibility.get(_DEPRECATED_APP_ID)
        if isinstance(app, dict) and app.get("deprecated") is True and app.get("value"):
            app_id = str(app["value"])
    return Continuation(
        continuation_id=record.continuation_id,
        revision=record.revision,
        run_id=str(record.scope.run_id),
        node_id=str(record.scope.node_id),
        kind=record.kind,
        status=ContinuationStatus(record.status.value),
        prompt=record.prompt,
        resume_schema=(None if "resume_schema" in nulls else _plain_mapping(record.resume_schema)),
        deadline=record.deadline,
        poll=None if "poll" in nulls else _plain_mapping(record.poll_payload),
        next_wakeup_at=record.next_wakeup_at,
        attempts=record.attempts,
        channel=record.channel,
        created_at=record.created_at,
        closed_at=record.closed_at,
        payload=None if "payload" in nulls else _plain_mapping(record.payload),
        session_id=record.scope.session_id,
        agent_id=record.scope.agent_id,
        app_id=app_id,
        graph_id=record.scope.graph_id,
        correlators=tuple(
            Correlator(value.scheme, value.channel, value.thread, value.message)
            for value in record.correlators
        ),
    )


def _service_metadata(
    value: ContinuationDraft | Continuation,
    *,
    base: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    metadata = _plain(base or {})
    metadata[_SERVICE_CONTEXT] = {
        _NULL_FIELDS: [name for name in _OPTIONAL_MAPPING_FIELDS if getattr(value, name) is None]
    }
    compatibility = metadata.get(_COMPATIBILITY_METADATA)
    compatibility = compatibility if isinstance(compatibility, dict) else {}
    if value.app_id is None:
        compatibility.pop(_DEPRECATED_APP_ID, None)
    else:
        compatibility[_DEPRECATED_APP_ID] = {
            "value": value.app_id,
            "deprecated": True,
            "scheduled_removal": "future breaking release",
        }
    if compatibility:
        metadata[_COMPATIBILITY_METADATA] = compatibility
    else:
        metadata.pop(_COMPATIBILITY_METADATA, None)
    return metadata


def _scope_for_draft(owner_scope: StorageScope, draft: ContinuationDraft) -> StorageScope:
    return _scope(
        owner_scope,
        **_optional_dimensions(
            session_id=draft.session_id,
            run_id=draft.run_id,
            graph_id=draft.graph_id,
            node_id=draft.node_id,
            agent_id=draft.agent_id,
        ),
    )


def _scope_for_continuation(
    owner_scope: StorageScope,
    continuation: Continuation,
) -> StorageScope:
    return _scope(
        owner_scope,
        **_optional_dimensions(
            session_id=continuation.session_id,
            run_id=continuation.run_id,
            graph_id=continuation.graph_id,
            node_id=continuation.node_id,
            agent_id=continuation.agent_id,
        ),
    )


def _scope(owner_scope: StorageScope, **dimensions: str) -> StorageScope:
    return merge_storage_scope(owner_scope, **dimensions)


def _optional_dimensions(**dimensions: str | None) -> dict[str, str]:
    return {name: value for name, value in dimensions.items() if value is not None}


def _owned_by(scope: StorageScope, owner_scope: StorageScope) -> bool:
    return all(getattr(scope, name) == value for name, value in owner_scope.as_filter().items())


def _to_canonical_correlator(value: Correlator) -> CanonicalCorrelator:
    return CanonicalCorrelator(
        scheme=value.scheme,
        channel=value.channel,
        thread=value.thread,
        message=value.message,
    )


def _to_runtime_lease(
    record: CanonicalLease,
    *,
    reclaimed: bool = False,
) -> TimerLease:
    return TimerLease(
        fire_id=record.fire_id,
        continuation_id=record.continuation_id,
        run_id=str(record.scope.run_id),
        node_id=str(record.scope.node_id),
        scheduled_for=record.scheduled_for,
        status=TimerLeaseStatus(record.status.value),
        attempts=record.attempts,
        revision=record.revision,
        updated_at=record.updated_at,
        worker_id=record.worker_id,
        lease_until=record.lease_until,
        next_attempt_at=record.next_attempt_at,
        last_error=record.last_error,
        finished_at=record.finished_at,
        reclaimed=reclaimed,
    )


def _owns_lease(current: CanonicalLease | None, lease: TimerLease) -> bool:
    return bool(
        current is not None
        and current.status is CanonicalLeaseStatus.LEASED
        and current.fire_id == lease.fire_id
        and current.continuation_id == lease.continuation_id
        and current.scope.run_id == lease.run_id
        and current.scope.node_id == lease.node_id
        and current.scheduled_for == _utc(lease.scheduled_for)
        and current.worker_id == lease.worker_id
        and current.revision == lease.revision
    )


def _plain_mapping(value: Mapping[str, object] | None) -> dict[str, Any]:
    if value is None:
        return {}
    plain = _plain(value)
    if not isinstance(plain, dict):
        raise TypeError("continuation JSON field must be a mapping")
    return plain


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _utc_optional(value: datetime | None) -> datetime | None:
    return _utc(value) if value is not None else None

"""Canonical run, result, and session projections over provider storage."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from aethergraph.api.v1.schemas import Session
from aethergraph.contracts.services.runs import RunResultStore, RunStore
from aethergraph.contracts.services.sessions import SessionStore
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunResult,
    RunStatus,
    RunVisibility,
    SessionKind,
)
from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    PageRequest,
    RunQuery,
    RunRecord as CanonicalRunRecord,
    RunRepository,
    RunResultRecord as CanonicalRunResultRecord,
    RunResultRepository,
    RunStatus as CanonicalRunStatus,
    SessionKind as CanonicalSessionKind,
    SessionQuery,
    SessionRecord as CanonicalSessionRecord,
    SessionRepository,
    StorageBundle,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageScope,
)

_MAX_WINDOW = 1_000
_PUBLIC_METADATA = "public_metadata"
_SERVICE_CONTEXT = "service_context"
_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_RESERVED_PUBLIC_METADATA = frozenset(
    {
        _PUBLIC_METADATA,
        _SERVICE_CONTEXT,
        _COMPATIBILITY_METADATA,
        _DEPRECATED_APP_ID,
        "application_id",
        "client_id",
    }
)


class CanonicalRunStore(RunStore):
    """Project the frozen runtime RunStore onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: RunRepository,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind run operations to one provider-authoritative owner.

        Examples:
            Bind a bundle repository:
                ```python
                store = CanonicalRunStore(repository=bundle.runs, owner_scope=scope, clock=clock)
                ```
            Bind a fake repository:
                ```python
                store = CanonicalRunStore(repository=fake, owner_scope=scope, clock=clock)
                ```

        Args:
            repository: Exact canonical run repository.
            owner_scope: Trusted provider ownership scope.
            clock: Timezone-aware UTC default occurrence clock.

        Returns:
            None: The provider-backed projection is ready.

        Notes:
            Construction performs no I/O and the bundle retains lifecycle ownership.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope
        self._clock = clock

    async def create(self, record: RunRecord) -> None:
        """Create one initial canonical run from a frozen runtime record.

        Examples:
            Create a run:
                ```python
                await store.create(record)
                ```
            Replay an identical create:
                ```python
                await store.create(record)
                ```

        Args:
            record: Complete service-facing initial run.

        Returns:
            None: The provider accepted the create or exact replay.

        Notes:
            Provider-owned counters must be empty at creation and conflicts propagate.
        """
        await self._repository.create(_run_to_canonical(record, self._owner_scope, revision=1))

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
        """Advance mutable run state through exact provider revision CAS.

        Examples:
            Mark success:
                ```python
                await store.update_status(run_id, RunStatus.succeeded, finished_at=now)
                ```
            Merge metadata:
                ```python
                await store.update_status(run_id, status, meta_update={"phase": "done"})
                ```

        Args:
            run_id: Exact stable run identity.
            status: Frozen runtime lifecycle status.
            finished_at: Optional terminal timestamp.
            error: Optional error replacement when non-null.
            meta_update: Optional public metadata merge.
            field_updates: Optional provider-marker replay from RunManager.

        Returns:
            None: The run was absent, unchanged, or committed.

        Notes:
            Unknown/provider-owned field changes fail; stale revisions propagate.
        """
        current = await self._repository.get(
            _operation_scope(self._owner_scope, run_id=run_id), run_id
        )
        if current is None:
            return
        public, service, compatibility = _metadata_parts(current.metadata, "run")
        public.update(_run_public_metadata(meta_update or {}))
        if field_updates:
            allowed = {"result_available", "result_updated_at"}
            unknown = set(field_updates) - allowed
            if unknown:
                raise ValueError("Unsupported run field updates: " + ", ".join(sorted(unknown)))
            if (
                field_updates.get("result_available", current.result_available)
                != current.result_available
            ):
                raise ValueError("result_available is provider-owned")
            if (
                field_updates.get("result_updated_at", current.result_updated_at)
                != current.result_updated_at
            ):
                raise ValueError("result_updated_at is provider-owned")
        proposed = replace(
            current,
            revision=current.revision + 1,
            status=CanonicalRunStatus(status.value),
            finished_at=current.finished_at if finished_at is None else finished_at,
            error=current.error if error is None else error,
            metadata=_metadata(public, service, compatibility),
        )
        if replace(proposed, revision=current.revision) == current:
            return
        await self._repository.compare_and_set(proposed, current.revision)

    async def get(self, run_id: str) -> RunRecord | None:
        """Read one provider-authorized run and project it to runtime shape.

        Examples:
            Read a run:
                ```python
                record = await store.get("run-1")
                ```
            Detect absence:
                ```python
                assert await store.get("missing") is None
                ```

        Args:
            run_id: Exact stable run identity.

        Returns:
            RunRecord | None: Frozen runtime projection or `None`.

        Notes:
            App identity is decoded only from explicit deprecated compatibility metadata.
        """
        record = await self._repository.get(
            _operation_scope(self._owner_scope, run_id=run_id), run_id
        )
        return project_canonical_run_record(record) if record is not None else None

    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        """List one bounded recent-run window using provider-side filters.

        Examples:
            List recent runs:
                ```python
                records = await store.list(limit=20)
                ```
            List one session:
                ```python
                records = await store.list(session_id="session-1", limit=20)
                ```

        Args:
            graph_id: Optional exact graph filter.
            status: Optional exact status filter.
            user_id: Optional exact user scope.
            org_id: Optional exact organization scope.
            session_id: Optional exact session scope.
            limit: Positive requested result count.
            offset: Non-negative bounded compatibility offset.

        Returns:
            list[RunRecord]: Recent provider-authorized runtime projections.

        Notes:
            The compatibility window is capped at 1,000 and never scans unbounded data.
        """
        window = _window(limit, offset)
        dimensions = _present(user_id=user_id, org_id=org_id, session_id=session_id)
        if graph_id is not None:
            dimensions["graph_id"] = graph_id
        query = RunQuery(
            scope=merge_storage_scope(self._owner_scope, **dimensions),
            statuses=(CanonicalRunStatus(status.value),) if status is not None else (),
            page=PageRequest(limit=window),
        )
        page = await self._repository.query(query)
        return [project_canonical_run_record(item) for item in page.items[offset : offset + limit]]

    async def record_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one exact canonical run artifact occurrence.

        Examples:
            Count an occurrence:
                ```python
                await store.record_artifact(run_id, artifact_id=artifact_id, occurrence_id=occurrence_id)
                ```
            Replay an occurrence:
                ```python
                await store.record_artifact(run_id, artifact_id=artifact_id, occurrence_id=occurrence_id)
                ```

        Args:
            run_id: Exact stable run identity.
            artifact_id: Content identity retained only by the frozen service preview.
            occurrence_id: Exact provider idempotency identity.
            created_at: Optional UTC occurrence time.

        Returns:
            None: The occurrence was counted, replayed, or its run was absent.

        Notes:
            Canonical counters use occurrence identity; no identity is fabricated here.
        """
        _nonempty("artifact_id", artifact_id)
        current = await self._repository.get(
            _operation_scope(self._owner_scope, run_id=run_id), run_id
        )
        if current is None:
            return
        try:
            await self._repository.record_artifact(
                current.scope,
                run_id,
                artifact_id,
                occurrence_id,
                created_at or self._clock(),
            )
        except StorageNotFoundError:
            return


class CanonicalRunResultStore(RunResultStore):
    """Project frozen durable outputs onto canonical result/run repositories."""

    def __init__(
        self,
        *,
        repository: RunResultRepository,
        runs: RunRepository,
        owner_scope: StorageScope,
    ) -> None:
        """Bind result operations to one coherent provider bundle.

        Examples:
            Bind bundle repositories:
                ```python
                store = CanonicalRunResultStore(repository=bundle.run_results, runs=bundle.runs, owner_scope=scope)
                ```
            Bind fake repositories:
                ```python
                store = CanonicalRunResultStore(repository=results, runs=runs, owner_scope=scope)
                ```

        Args:
            repository: Exact canonical result repository.
            runs: Exact owning run repository from the same bundle.
            owner_scope: Trusted provider ownership scope.

        Returns:
            None: The provider-backed projection is ready.

        Notes:
            Result/run atomicity remains provider-owned; this object has no close.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._runs = runs
        self._owner_scope = owner_scope

    async def save(self, run_id: str, result: RunResult) -> None:
        """Create or revise one successful canonical durable result.

        Examples:
            Save direct output:
                ```python
                await store.save(run_id, result)
                ```
            Save recovered output:
                ```python
                await store.save(run_id, recovered)
                ```

        Args:
            run_id: Exact owning run identity.
            result: Complete frozen successful result.

        Returns:
            None: The provider committed the result and run marker atomically.

        Notes:
            The owning run must already be durably successful in the same scope, and
            outputs must remain an object at the frozen service boundary.
        """
        if result.run_id != run_id:
            raise ValueError("run_id must match result.run_id")
        run = await self._runs.get(_operation_scope(self._owner_scope, run_id=run_id), run_id)
        if run is None:
            raise StorageNotFoundError(run_id)
        if run.graph_id != result.graph_id:
            raise ValueError("result graph_id must match its owning run")
        current = await self._repository.get(run.scope, run_id)
        revision = 1 if current is None else current.revision + 1
        outputs = _plain(result.outputs)
        if not isinstance(outputs, dict):
            raise TypeError("Run result outputs must be an object")
        record = CanonicalRunResultRecord(
            run_id=run_id,
            graph_id=result.graph_id,
            scope=run.scope,
            status=CanonicalRunStatus(result.status.value),
            outputs=outputs,
            revision=revision,
            created_at=result.created_at,
            updated_at=result.updated_at,
            source=result.source,
            snapshot_revision=result.snapshot_rev,
        )
        await self._repository.compare_and_set(record, revision - 1)

    async def get(self, run_id: str) -> RunResult | None:
        """Read one current provider-authorized durable result.

        Examples:
            Read output:
                ```python
                result = await store.get("run-1")
                ```
            Detect absence:
                ```python
                assert await store.get("missing") is None
                ```

        Args:
            run_id: Exact stable run identity.

        Returns:
            RunResult | None: Frozen result projection or `None`.

        Notes:
            Outputs are read only from the canonical result repository.
        """
        record = await self._repository.get(
            _operation_scope(self._owner_scope, run_id=run_id), run_id
        )
        return _result_to_service(record) if record is not None else None

    async def delete(self, run_id: str) -> None:
        """Delete one current result through exact provider revision CAS.

        Examples:
            Delete output:
                ```python
                await store.delete("run-1")
                ```
            Delete absent output:
                ```python
                await store.delete("missing")
                ```

        Args:
            run_id: Exact stable run identity.

        Returns:
            None: The result was deleted or already absent.

        Notes:
            Provider deletion atomically clears and advances the owning run marker.
        """
        current = await self._repository.get(
            _operation_scope(self._owner_scope, run_id=run_id), run_id
        )
        if current is not None:
            await self._repository.delete(current.scope, run_id, current.revision)


class CanonicalSessionStore(SessionStore):
    """Project the frozen SessionStore onto one canonical session repository."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        owner_scope: StorageScope,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind session operations to one provider-authoritative owner.

        Examples:
            Bind a bundle repository:
                ```python
                store = CanonicalSessionStore(repository=bundle.sessions, owner_scope=scope, clock=clock)
                ```
            Bind a fake repository:
                ```python
                store = CanonicalSessionStore(repository=fake, owner_scope=scope, clock=clock)
                ```

        Args:
            repository: Exact canonical session repository.
            owner_scope: Trusted provider ownership scope.
            clock: Timezone-aware UTC transition clock.

        Returns:
            None: The provider-backed projection is ready.

        Notes:
            Construction performs no I/O and the bundle owns repository lifecycle.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope
        self._clock = clock

    async def create(
        self,
        *,
        session_id: str | None = None,
        kind: SessionKind,
        user_id: str | None = None,
        org_id: str | None = None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        """Create one canonical session with optional caller identity.

        Examples:
            Create a generated identity:
                ```python
                session = await store.create(kind=SessionKind.chat)
                ```
            Create an explicit identity:
                ```python
                session = await store.create(session_id="session-1", kind=SessionKind.chat)
                ```

        Args:
            session_id: Optional exact caller-owned identity.
            kind: Frozen runtime session kind.
            user_id: Optional exact user scope.
            org_id: Optional exact organization scope.
            title: Optional initial title.
            source: Non-empty session source.
            external_ref: Optional external reference.

        Returns:
            Session: Frozen public session projection.

        Notes:
            Frozen identity replay returns the provider winner; collisions fail directly.
        """
        resolved = session_id or f"sess_{uuid4().hex[:8]}"
        scope = merge_storage_scope(
            self._owner_scope,
            **_present(user_id=user_id, org_id=org_id, session_id=resolved),
        )
        if session_id:
            current = await self._get_record(resolved)
            if current is not None:
                _validate_session_create_identity(
                    current,
                    kind=kind,
                    scope=scope,
                    source=source,
                    external_ref=external_ref,
                )
                return _session_to_service(current)
        now = self._clock()
        metadata = {_SERVICE_CONTEXT: {"title_source": "manual" if (title or "").strip() else None}}
        record = CanonicalSessionRecord(
            session_id=resolved,
            kind=CanonicalSessionKind(kind.value),
            scope=scope,
            revision=1,
            created_at=now,
            updated_at=now,
            title=title,
            source=source,
            external_reference=external_ref,
            metadata=metadata,
        )
        try:
            stored = await self._repository.create(record)
        except StorageIntegrityError:
            if not session_id:
                raise
            winner = await self._get_record(resolved)
            if winner is None:
                raise
            _validate_session_create_identity(
                winner,
                kind=kind,
                scope=scope,
                source=source,
                external_ref=external_ref,
            )
            stored = winner
        return _session_to_service(stored)

    async def get(self, session_id: str) -> Session | None:
        """Read one provider-authorized session projection.

        Examples:
            Read a session:
                ```python
                session = await store.get("session-1")
                ```
            Detect absence:
                ```python
                assert await store.get("missing") is None
                ```

        Args:
            session_id: Exact stable session identity.

        Returns:
            Session | None: Frozen public projection or `None`.

        Notes:
            Lookup never broadens beyond the trusted owner scope.
        """
        record = await self._repository.get(
            _operation_scope(self._owner_scope, session_id=session_id), session_id
        )
        return _session_to_service(record) if record is not None else None

    async def storage_scope(self, session_id: str) -> StorageScope | None:
        """Read the canonical storage authority for one session.

        Session-bound artifact and memory operations use this exact persisted
        scope instead of reconstructing ownership from an actor identity.

        Examples:
            Resolve a stored session scope:
                ```python
                scope = await store.storage_scope("session-1")
                assert scope.session_id == "session-1"
                ```

            Resolve an unknown session:
                ```python
                assert await store.storage_scope("missing") is None
                ```

        Args:
            session_id: Exact stable session identity.

        Returns:
            StorageScope | None: Persisted session scope, or `None` when absent.

        Notes:
            The returned immutable scope is never widened with request identity.
        """

        record = await self._get_record(session_id)
        return record.scope if record is not None else None

    async def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        """List one bounded provider-side session window.

        Examples:
            List user sessions:
                ```python
                sessions = await store.list_for_user(user_id="user-1")
                ```
            List chats:
                ```python
                sessions = await store.list_for_user(user_id="user-1", kind=SessionKind.chat)
                ```

        Args:
            user_id: Optional exact user scope.
            org_id: Optional exact organization scope.
            kind: Optional exact session kind.
            limit: Positive requested result count.
            offset: Non-negative bounded compatibility offset.

        Returns:
            list[Session]: Recent provider-authorized session projections.

        Notes:
            The compatibility window is capped at 1,000 and never scans unbounded data.
        """
        window = _window(limit, offset)
        query = SessionQuery(
            scope=merge_storage_scope(
                self._owner_scope, **_present(user_id=user_id, org_id=org_id)
            ),
            kinds=(CanonicalSessionKind(kind.value),) if kind is not None else (),
            page=PageRequest(limit=window),
        )
        page = await self._repository.query(query)
        return [_session_to_service(item) for item in page.items[offset : offset + limit]]

    async def touch(self, session_id: str, *, updated_at: datetime | None = None) -> None:
        """Advance one session update timestamp through revision CAS.

        Examples:
            Touch now:
                ```python
                await store.touch("session-1")
                ```
            Touch at a known time:
                ```python
                await store.touch("session-1", updated_at=now)
                ```

        Args:
            session_id: Exact stable session identity.
            updated_at: Optional UTC update time.

        Returns:
            None: The session was absent, unchanged, or committed.

        Notes:
            Provider time remains monotonic; earlier touches become no-ops.
        """
        current = await self._get_record(session_id)
        if current is None:
            return
        resolved = max(current.updated_at, updated_at or self._clock())
        if resolved == current.updated_at:
            return
        await self._repository.compare_and_set(
            replace(current, revision=current.revision + 1, updated_at=resolved),
            current.revision,
        )

    async def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        title_source: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        """Update mutable session presentation metadata through revision CAS.

        Examples:
            Rename a session:
                ```python
                session = await store.update("session-1", title="New title")
                ```
            Update an external reference:
                ```python
                session = await store.update("session-1", external_ref="external-1")
                ```

        Args:
            session_id: Exact stable session identity.
            title: Optional title replacement when non-null.
            title_source: Optional title provenance paired with title.
            external_ref: Optional external-reference replacement when non-null.

        Returns:
            Session | None: Updated public projection or `None` when absent.

        Notes:
            A provider revision advances for identical explicit updates; title source
            accepts only the frozen `manual` and `auto` values.
        """
        current = await self._get_record(session_id)
        if current is None:
            return None
        metadata = _plain_mapping(current.metadata)
        service = metadata.setdefault(_SERVICE_CONTEXT, {})
        if not isinstance(service, dict):
            raise ValueError("Canonical session service metadata is malformed")
        if title is not None:
            resolved_title_source = title_source or "manual"
            if resolved_title_source not in {"manual", "auto"}:
                raise ValueError("title_source must be 'manual' or 'auto'")
            service["title_source"] = resolved_title_source
        proposed = replace(
            current,
            revision=current.revision + 1,
            updated_at=max(current.updated_at, self._clock()),
            title=current.title if title is None else title,
            external_reference=(
                current.external_reference if external_ref is None else external_ref
            ),
            metadata=metadata,
        )
        stored = await self._repository.compare_and_set(proposed, current.revision)
        return _session_to_service(stored)

    async def delete(self, session_id: str) -> None:
        """Delete one current session through exact provider revision CAS.

        Examples:
            Delete a session:
                ```python
                await store.delete("session-1")
                ```
            Delete an absent session:
                ```python
                await store.delete("missing")
                ```

        Args:
            session_id: Exact stable session identity.

        Returns:
            None: The session was deleted or already absent.

        Notes:
            Provider-owned occurrence receipts cascade; runs and artifacts remain.
        """
        current = await self._get_record(session_id)
        if current is not None:
            await self._repository.delete(current.scope, session_id, current.revision)

    async def record_artifact(
        self,
        session_id: str,
        *,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one exact canonical session artifact occurrence.

        Examples:
            Count an occurrence:
                ```python
                await store.record_artifact(session_id, occurrence_id=occurrence_id)
                ```
            Replay an occurrence:
                ```python
                await store.record_artifact(session_id, occurrence_id=occurrence_id)
                ```

        Args:
            session_id: Exact stable session identity.
            occurrence_id: Exact provider idempotency identity.
            created_at: Optional UTC occurrence time.

        Returns:
            None: The occurrence was counted, replayed, or its session was absent.

        Notes:
            No content identity, timestamp identity, or random fallback is fabricated.
        """
        current = await self._get_record(session_id)
        if current is None:
            return
        try:
            await self._repository.record_artifact(
                current.scope,
                session_id,
                occurrence_id,
                created_at or self._clock(),
            )
        except StorageNotFoundError:
            return

    async def _get_record(self, session_id: str) -> CanonicalSessionRecord | None:
        return await self._repository.get(
            _operation_scope(self._owner_scope, session_id=session_id), session_id
        )


@dataclass(frozen=True, slots=True)
class CanonicalControlStores:
    """Frozen service projections bound to one coherent canonical bundle."""

    runs: CanonicalRunStore
    run_results: CanonicalRunResultStore
    sessions: CanonicalSessionStore


def bind_canonical_control_stores(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
) -> CanonicalControlStores:
    """Bind frozen control services to exact fields from one open bundle.

    Examples:
        Bind production composition inputs:
            ```python
            stores = bind_canonical_control_stores(bundle=bundle, owner_scope=scope, clock=clock)
            ```
        Bind a fake bundle:
            ```python
            stores = bind_canonical_control_stores(bundle=fake_bundle, owner_scope=scope, clock=clock)
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        clock: Timezone-aware UTC transition clock.

    Returns:
        CanonicalControlStores: Exact run/result/session service projections.

    Notes:
        Binding performs no selection, I/O, fallback, activation, or close operation.
    """
    validate_storage_owner_scope(owner_scope)
    return CanonicalControlStores(
        runs=CanonicalRunStore(repository=bundle.runs, owner_scope=owner_scope, clock=clock),
        run_results=CanonicalRunResultStore(
            repository=bundle.run_results,
            runs=bundle.runs,
            owner_scope=owner_scope,
        ),
        sessions=CanonicalSessionStore(
            repository=bundle.sessions,
            owner_scope=owner_scope,
            clock=clock,
        ),
    )


def _run_to_canonical(
    record: RunRecord,
    owner_scope: StorageScope,
    *,
    revision: int,
) -> CanonicalRunRecord:
    if record.app_id is not None:
        _nonempty("app_id", record.app_id)
    scope = merge_storage_scope(
        owner_scope,
        **_present(
            user_id=record.user_id,
            org_id=record.org_id,
            session_id=record.session_id,
            run_id=record.run_id,
            graph_id=record.graph_id,
            agent_id=record.agent_id,
        ),
    )
    service = {
        "origin": record.origin.value,
        "visibility": record.visibility.value,
        "importance": record.importance.value,
    }
    compatibility: dict[str, Any] = {}
    if record.app_id is not None:
        compatibility[_DEPRECATED_APP_ID] = {
            "value": record.app_id,
            "deprecated": True,
            "scheduled_removal": "future breaking release",
        }
    return CanonicalRunRecord(
        run_id=record.run_id,
        graph_id=record.graph_id,
        kind=record.kind,
        status=CanonicalRunStatus(record.status.value),
        scope=scope,
        revision=revision,
        started_at=record.started_at,
        finished_at=record.finished_at,
        tags=tuple(record.tags),
        error=record.error,
        metadata=_metadata(_run_public_metadata(record.meta), service, compatibility),
        artifact_count=record.artifact_count,
        first_artifact_at=record.first_artifact_at,
        last_artifact_at=record.last_artifact_at,
        recent_artifact_ids=tuple(record.recent_artifact_ids),
        result_available=record.result_available,
        result_updated_at=record.result_updated_at,
    )


def project_canonical_run_record(record: CanonicalRunRecord) -> RunRecord:
    """Project one canonical provider run into the stable runtime record.

    Intro:
        Decodes provider-neutral metadata and explicit compatibility metadata without
        exposing the repository record or inferring deprecated App identity.

    Examples:
        Project a queried run:
            ```python
            public = project_canonical_run_record(record)
            ```

        Read the stable status value:
            ```python
            assert project_canonical_run_record(record).status.value == "running"
            ```

    Args:
        record: Canonical provider run authorized by its repository query.

    Returns:
        RunRecord: Detached stable runtime projection of the canonical record.

    Notes:
        `app_id` is decoded only from explicitly deprecated optional compatibility
        metadata and never from provider scope, tags, or public metadata.
    """
    public, service, compatibility = _metadata_parts(record.metadata, "run")
    return RunRecord(
        run_id=record.run_id,
        graph_id=record.graph_id,
        kind=record.kind,
        status=RunStatus(record.status.value),
        started_at=record.started_at,
        finished_at=record.finished_at,
        tags=list(record.tags),
        user_id=record.scope.user_id,
        org_id=record.scope.org_id,
        error=record.error,
        meta=public,
        session_id=record.scope.session_id,
        origin=RunOrigin(str(service.get("origin") or RunOrigin.app.value)),
        visibility=RunVisibility(str(service.get("visibility") or RunVisibility.normal.value)),
        importance=RunImportance(str(service.get("importance") or RunImportance.normal.value)),
        agent_id=record.scope.agent_id,
        app_id=_deprecated_app_id(compatibility),
        artifact_count=record.artifact_count,
        first_artifact_at=record.first_artifact_at,
        last_artifact_at=record.last_artifact_at,
        recent_artifact_ids=list(record.recent_artifact_ids),
        result_available=record.result_available,
        result_updated_at=record.result_updated_at,
    )


def _result_to_service(record: CanonicalRunResultRecord) -> RunResult:
    outputs = _plain(record.outputs)
    if not isinstance(outputs, dict):
        raise ValueError("Canonical run result outputs must be an object")
    return RunResult(
        run_id=record.run_id,
        graph_id=record.graph_id,
        session_id=record.scope.session_id,
        status=RunStatus(record.status.value),
        outputs=outputs,
        created_at=record.created_at,
        updated_at=record.updated_at,
        source=record.source,
        snapshot_rev=record.snapshot_revision,
    )


def _session_to_service(record: CanonicalSessionRecord) -> Session:
    metadata = _plain_mapping(record.metadata)
    service = metadata.get(_SERVICE_CONTEXT, {})
    if not isinstance(service, dict):
        raise ValueError("Canonical session service metadata is malformed")
    return Session(
        session_id=record.session_id,
        kind=SessionKind(record.kind.value),
        title=record.title,
        title_source=service.get("title_source"),
        user_id=record.scope.user_id,
        org_id=record.scope.org_id,
        source=record.source,
        external_ref=record.external_reference,
        created_at=record.created_at,
        updated_at=record.updated_at,
        artifact_count=record.artifact_count,
        last_artifact_at=record.last_artifact_at,
    )


def _metadata_parts(
    metadata: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plain = _plain_mapping(metadata)
    public = plain.get(_PUBLIC_METADATA, {})
    service = plain.get(_SERVICE_CONTEXT, {})
    compatibility = plain.get(_COMPATIBILITY_METADATA, {})
    if not isinstance(public, dict) or not isinstance(service, dict):
        raise ValueError(f"Canonical {label} service metadata is malformed")
    if not isinstance(compatibility, dict):
        raise ValueError(f"Canonical {label} compatibility metadata is malformed")
    if label == "run":
        public = _run_public_metadata(public)
    return public, service, compatibility


def _metadata(
    public: Mapping[str, Any],
    service: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = {
        _PUBLIC_METADATA: _plain_mapping(public),
        _SERVICE_CONTEXT: _plain_mapping(service),
    }
    if compatibility:
        metadata[_COMPATIBILITY_METADATA] = _plain_mapping(compatibility)
    return metadata


def _deprecated_app_id(compatibility: Mapping[str, Any]) -> str | None:
    unknown = sorted(set(compatibility) - {_DEPRECATED_APP_ID})
    if unknown:
        raise ValueError("Canonical run compatibility metadata reserves: " + ", ".join(unknown))
    value = compatibility.get(_DEPRECATED_APP_ID)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Canonical run App compatibility metadata is malformed")
    app_id = value.get("value")
    if (
        value.get("deprecated") is not True
        or value.get("scheduled_removal") != "future breaking release"
        or not isinstance(app_id, str)
        or not app_id.strip()
    ):
        raise ValueError("Canonical run App compatibility metadata is malformed")
    return app_id


def _run_public_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    public = _plain_mapping(value)
    reserved = sorted(_RESERVED_PUBLIC_METADATA.intersection(public))
    if reserved:
        raise ValueError("Run public metadata reserves: " + ", ".join(reserved))
    return public


def _validate_session_create_identity(
    record: CanonicalSessionRecord,
    *,
    kind: SessionKind,
    scope: StorageScope,
    source: str,
    external_ref: str | None,
) -> None:
    expected = (
        CanonicalSessionKind(kind.value),
        scope.user_id,
        scope.org_id,
        source,
        external_ref,
    )
    actual = (
        record.kind,
        record.scope.user_id,
        record.scope.org_id,
        record.source,
        record.external_reference,
    )
    if actual != expected:
        raise ValueError(f"Session identity collision: {record.session_id}")


def _operation_scope(owner_scope: StorageScope, **dimensions: str) -> StorageScope:
    return merge_storage_scope(owner_scope, **dimensions)


def _present(**dimensions: str | None) -> dict[str, str]:
    return {name: value for name, value in dimensions.items() if value is not None}


def _window(limit: int, offset: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    window = limit + offset
    if window > _MAX_WINDOW:
        raise ValueError(f"offset plus limit must not exceed {_MAX_WINDOW}")
    return window


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    plain = _plain(value)
    if not isinstance(plain, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("mapping projection failed")
    return plain


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value

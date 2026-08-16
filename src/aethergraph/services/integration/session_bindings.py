"""Durable provider-neutral external-session binding storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Literal, Protocol

from aethergraph.contracts.integration import (
    ExternalIdentity,
    ExternalSessionBinding,
    IntegrationRoute,
)
from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.storage.contracts import (
    ExternalSessionBindingRecord,
    ExternalSessionBindingRepository,
    ExternalSessionBindingRequest,
    StorageIntegrityError,
    StorageScope,
)


class SessionBindingError(RuntimeError):
    """Structured failure raised for invalid or incompatible session bindings."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.binding_thread_required",
            "integration.binding_build_mismatch",
        ],
        message: str,
    ) -> None:
        """Create one stable external-session binding failure.

        Examples:
            Reject a missing required thread:
            ```python
            SessionBindingError(
                code="integration.binding_thread_required",
                message="This route requires a thread identity.",
            )
            ```

            Reject an attempt to move a session to another build:
            ```python
            SessionBindingError(
                code="integration.binding_build_mismatch",
                message="Existing session is pinned to another build.",
            )
            ```

        Args:
            code: Stable machine-readable binding failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Binding failures never create a replacement session implicitly.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class BindingResolution:
    """Result of atomically resolving or creating one external binding."""

    binding: ExternalSessionBinding
    created: bool


class ExternalSessionBindingStore(Protocol):
    """Provider-neutral persistence contract for durable session bindings."""

    async def get_or_create(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
    ) -> BindingResolution:
        """Resolve or create one exact route-scoped binding.

        Examples:
            Resolve a conversation:
            ```python
            result = await store.get_or_create(
                route=route,
                external_identity=identity,
                build_id="build-1",
                binding_id="binding-1",
                ag_session_id="session-1",
                now=now,
            )
            ```

            Inspect creation ownership:
            ```python
            if result.created:
                await create_ag_session(result.binding.ag_session_id)
            ```

        Args:
            route: Exact resolved route and session-scope policy.
            external_identity: Authenticated external conversation identity.
            build_id: Immutable build identity for the bound AG session.
            binding_id: Candidate binding identifier.
            ag_session_id: Candidate AG session identifier.
            now: Authoritative acceptance timestamp.

        Returns:
            BindingResolution: Persisted binding and creation ownership.

        Notes:
            Implementations must make scope creation atomic.
        """
        ...

    async def get(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        """Read one exact route-scoped binding.

        Examples:
            Read an existing binding:
            ```python
            binding = await store.get(route=route, external_identity=identity)
            ```

            Detect an unbound identity:
            ```python
            assert await store.get(route=route, external_identity=identity) is None
            ```

        Args:
            route: Exact resolved route and session-scope policy.
            external_identity: Authenticated external conversation identity.

        Returns:
            ExternalSessionBinding | None: Persisted binding when present.

        Notes:
            Implementations must use the same scope calculation as creation.
        """
        ...


class CanonicalExternalSessionBindingStore:
    """Project route-authored Host bindings onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: ExternalSessionBindingRepository,
        owner_scope: StorageScope,
    ) -> None:
        """Bind external sessions to one provider-authoritative owner.

        The service computes the opaque route session key and merges it with trusted
        provider ownership. Provider records receive no Host route or external-
        identity DTO and this projection retains no physical path.

        Examples:
            Bind an opened bundle repository:
                ```python
                store = CanonicalExternalSessionBindingStore(
                    repository=bundle.external_session_bindings,
                    owner_scope=owner_scope,
                )
                ```

            Bind a deterministic test repository:
                ```python
                store = CanonicalExternalSessionBindingStore(
                    repository=fake_repository,
                    owner_scope=StorageScope(project_id="project-1"),
                )
                ```

        Args:
            repository: Canonical external-session repository from one bundle.
            owner_scope: Exact trusted Host ownership scope.

        Returns:
            None: The provider-backed service projection is ready without I/O.

        Notes:
            App/client identity, provider selection, and fallback are absent.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope

    async def get_or_create(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
    ) -> BindingResolution:
        """Resolve or atomically create one route-scoped external binding.

        Candidate binding and session identities are used only on creation. Existing
        provider-authoritative identities are resubmitted during last-seen updates,
        including after a concurrent creator wins.

        Examples:
            Create a first binding:
                ```python
                result = await store.get_or_create(
                    route=route,
                    external_identity=identity,
                    build_id="build-1",
                    binding_id="binding-1",
                    ag_session_id="session-1",
                    now=now,
                )
                ```

            Resolve an existing binding:
                ```python
                existing = await store.get_or_create(
                    route=route,
                    external_identity=identity,
                    build_id="build-1",
                    binding_id="unused-candidate",
                    ag_session_id="unused-session",
                    now=later,
                )
                ```

        Args:
            route: Exact immutable Host route and session policy.
            external_identity: Authenticated external conversation identity.
            build_id: Host build that must remain pinned.
            binding_id: Candidate binding identity used only on creation.
            ag_session_id: Candidate AG session identity used only on creation.
            now: Authoritative acceptance timestamp.

        Returns:
            BindingResolution: Frozen Host binding and creation ownership.

        Notes:
            Concurrent resolution retries only the same canonical repository record;
            it never selects another provider or creates a replacement binding.
        """
        scope = _binding_scope(self._owner_scope, route, external_identity)
        existing = await self._repository.get(scope, route.route_id)
        if existing is not None and existing.build_id != build_id:
            _raise_build_mismatch(existing.build_id, build_id)
        request = _canonical_binding_request(
            scope=scope,
            route=route,
            build_id=build_id,
            binding_id=existing.binding_id if existing is not None else binding_id,
            ag_session_id=existing.ag_session_id if existing is not None else ag_session_id,
            now=max(existing.last_seen_at, now) if existing is not None else now,
        )
        try:
            result = await self._repository.get_or_create(request)
        except StorageIntegrityError:
            winner = await self._repository.get(scope, route.route_id)
            if winner is None:
                raise
            if winner.build_id != build_id:
                _raise_build_mismatch(winner.build_id, build_id)
            result = await self._repository.get_or_create(
                _canonical_binding_request(
                    scope=scope,
                    route=route,
                    build_id=build_id,
                    binding_id=winner.binding_id,
                    ag_session_id=winner.ag_session_id,
                    now=max(winner.last_seen_at, now),
                )
            )
        return BindingResolution(
            binding=_host_binding(result.record, external_identity),
            created=result.created,
        )

    async def get(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        """Read one exact route-scoped binding from the canonical repository.

        The same route-authored opaque key used during creation is recomputed from
        the authenticated identity before the exact provider lookup.

        Examples:
            Read an existing binding:
                ```python
                binding = await store.get(route=route, external_identity=identity)
                ```

            Detect an unbound identity:
                ```python
                assert await store.get(route=route, external_identity=new_identity) is None
                ```

        Args:
            route: Exact immutable Host route and session policy.
            external_identity: Authenticated external conversation identity.

        Returns:
            ExternalSessionBinding | None: Frozen Host projection or `None`.

        Notes:
            A miss is final and does not probe another identity or provider.
        """
        scope = _binding_scope(self._owner_scope, route, external_identity)
        record = await self._repository.get(scope, route.route_id)
        return _host_binding(record, external_identity) if record is not None else None


def _scope_key(*, route: IntegrationRoute, external_identity: ExternalIdentity) -> str:
    scope = route.session_policy.scope
    include_thread = scope in {"conversation_thread", "conversation_thread_user"}
    include_user = scope in {"conversation_user", "conversation_thread_user"}
    if include_thread and external_identity.thread_id is None:
        raise SessionBindingError(
            code="integration.binding_thread_required",
            message=f"Route {route.route_id!r} requires an external thread identity.",
        )
    fields = {
        "tenant_id": external_identity.tenant_id,
        "conversation_id": external_identity.conversation_id,
        "thread_id": external_identity.thread_id if include_thread else None,
        "user_id": external_identity.user_id if include_user else None,
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _binding_scope(
    owner_scope: StorageScope,
    route: IntegrationRoute,
    external_identity: ExternalIdentity,
) -> StorageScope:
    return merge_storage_scope(
        owner_scope,
        scope_key=_scope_key(route=route, external_identity=external_identity),
    )


def _canonical_binding_request(
    *,
    scope: StorageScope,
    route: IntegrationRoute,
    build_id: str,
    binding_id: str,
    ag_session_id: str,
    now: datetime,
) -> ExternalSessionBindingRequest:
    return ExternalSessionBindingRequest(
        binding_id=binding_id,
        route_id=route.route_id,
        build_id=build_id,
        ag_session_id=ag_session_id,
        scope=scope,
        now=now,
    )


def _host_binding(
    record: ExternalSessionBindingRecord,
    external_identity: ExternalIdentity,
) -> ExternalSessionBinding:
    return ExternalSessionBinding(
        binding_id=record.binding_id,
        route_id=record.route_id,
        external_identity=external_identity,
        ag_session_id=record.ag_session_id,
        build_id=record.build_id,
        created_at=record.created_at,
        last_seen_at=record.last_seen_at,
    )


def _raise_build_mismatch(current: str, requested: str) -> None:
    raise SessionBindingError(
        code="integration.binding_build_mismatch",
        message=f"External session is pinned to build {current!r}, not {requested!r}.",
    )

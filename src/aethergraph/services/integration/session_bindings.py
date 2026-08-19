"""Canonical provisioning for integration sessions and their external bindings."""

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
    ExternalSessionBindingRequest,
    IntegrationSessionRepository,
    SessionKind,
    SessionRecord,
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
        """Create one stable integration-session provisioning failure.

        Invalid route identity and build changes are reported through stable codes
        without creating a competing session or binding.

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
            code: Stable machine-readable provisioning failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Provisioning failures never create a replacement session implicitly.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class IntegrationSessionResolution:
    """Result of atomically provisioning a session and external binding."""

    binding: ExternalSessionBinding
    session: SessionRecord
    session_created: bool
    binding_created: bool


class IntegrationSessionStore(Protocol):
    """Provider-neutral service contract for integration session provisioning."""

    async def provision(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
        title: str | None = None,
    ) -> IntegrationSessionResolution:
        """Provision one canonical session and its route-scoped binding.

        The operation guarantees that a successful binding always references an
        existing canonical session in the same control-store transaction.

        Examples:
            Provision a first conversation:
                ```python
                result = await store.provision(
                    route=route,
                    external_identity=identity,
                    build_id="build-1",
                    binding_id="binding-1",
                    ag_session_id="session-1",
                    now=now,
                )
                ```

            Re-provision an existing conversation safely:
                ```python
                existing = await store.provision(
                    route=route,
                    external_identity=identity,
                    build_id="build-1",
                    binding_id="unused-binding",
                    ag_session_id="unused-session",
                    now=later,
                )
                ```

        Args:
            route: Exact resolved route and session-scope policy.
            external_identity: Authenticated external conversation identity.
            build_id: Immutable build identity for the bound AG session.
            binding_id: Candidate binding identifier used only on creation.
            ag_session_id: Candidate session identifier used only on creation.
            now: Authoritative provisioning timestamp.
            title: Optional title used only when the canonical session is created.

        Returns:
            IntegrationSessionResolution: Binding plus session and binding ownership.

        Notes:
            Implementations do not expose a binding-only creation path.
        """
        ...

    async def get_binding(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        """Read one exact route-scoped binding.

        This read uses the same canonical scope projection as provisioning and does
        not probe alternate providers or identities.

        Examples:
            Read an existing binding:
                ```python
                binding = await store.get_binding(route=route, external_identity=identity)
                ```

            Detect an unbound identity:
                ```python
                assert await store.get_binding(
                    route=route,
                    external_identity=identity,
                ) is None
                ```

        Args:
            route: Exact resolved route and session-scope policy.
            external_identity: Authenticated external conversation identity.

        Returns:
            ExternalSessionBinding | None: Persisted binding when present.

        Notes:
            A binding returned here was created only through compound provisioning.
        """
        ...


class CanonicalIntegrationSessionStore:
    """Project Host integration identities onto canonical session persistence."""

    def __init__(
        self,
        *,
        repository: IntegrationSessionRepository,
        owner_scope: StorageScope,
    ) -> None:
        """Bind integration sessions to one provider-authoritative owner.

        The service centralizes route scope, canonical session metadata, and the
        compound repository transaction. It retains no physical storage path.

        Examples:
            Bind an opened bundle repository:
                ```python
                store = CanonicalIntegrationSessionStore(
                    repository=bundle.integration_sessions,
                    owner_scope=owner_scope,
                )
                ```

            Bind a deterministic test repository:
                ```python
                store = CanonicalIntegrationSessionStore(
                    repository=fake_repository,
                    owner_scope=StorageScope(project_id="project-1"),
                )
                ```

        Args:
            repository: Canonical compound session repository from one bundle.
            owner_scope: Exact trusted Host ownership scope.

        Returns:
            None.

        Notes:
            App identity, provider selection, and fallback are absent.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope

    async def provision(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
        build_id: str,
        binding_id: str,
        ag_session_id: str,
        now: datetime,
        title: str | None = None,
    ) -> IntegrationSessionResolution:
        """Provision one canonical session and route-scoped binding atomically.

        Existing provider-authoritative identities win over candidate identifiers.
        Concurrent creators converge through the same compound repository operation.

        Examples:
            Create a first integration session:
                ```python
                result = await store.provision(
                    route=route,
                    external_identity=identity,
                    build_id="build-1",
                    binding_id="binding-1",
                    ag_session_id="session-1",
                    now=now,
                )
                ```

            Repair a previously orphaned binding:
                ```python
                repaired = await store.provision(
                    route=route,
                    external_identity=identity,
                    build_id="build-1",
                    binding_id="unused-binding",
                    ag_session_id="unused-session",
                    now=later,
                )
                ```

        Args:
            route: Exact immutable Host route and session policy.
            external_identity: Authenticated external conversation identity.
            build_id: Host build that must remain pinned.
            binding_id: Candidate binding identity used only on creation.
            ag_session_id: Candidate session identity used only on creation.
            now: Authoritative provisioning timestamp.
            title: Optional title used only for a newly created session.

        Returns:
            IntegrationSessionResolution: Frozen binding and creation ownership.

        Notes:
            Success guarantees that the returned binding's session exists before
            downstream ingress, artifacts, or capability registration can run.
        """
        scope = _binding_scope(self._owner_scope, route, external_identity)
        existing = await self._repository.get_binding(scope, route.route_id)
        if existing is not None and existing.build_id != build_id:
            _raise_build_mismatch(existing.build_id, build_id)
        resolved_binding_id = existing.binding_id if existing is not None else binding_id
        resolved_session_id = existing.ag_session_id if existing is not None else ag_session_id
        request = _canonical_binding_request(
            scope=scope,
            route=route,
            build_id=build_id,
            binding_id=resolved_binding_id,
            ag_session_id=resolved_session_id,
            now=max(existing.last_seen_at, now) if existing is not None else now,
        )
        session = _canonical_session(
            owner_scope=self._owner_scope,
            route=route,
            external_identity=external_identity,
            session_id=resolved_session_id,
            now=now,
            title=title,
        )
        try:
            result = await self._repository.provision(request, session)
        except StorageIntegrityError:
            winner = await self._repository.get_binding(scope, route.route_id)
            if winner is None:
                raise
            if winner.build_id != build_id:
                _raise_build_mismatch(winner.build_id, build_id)
            result = await self._repository.provision(
                _canonical_binding_request(
                    scope=scope,
                    route=route,
                    build_id=build_id,
                    binding_id=winner.binding_id,
                    ag_session_id=winner.ag_session_id,
                    now=max(winner.last_seen_at, now),
                ),
                _canonical_session(
                    owner_scope=self._owner_scope,
                    route=route,
                    external_identity=external_identity,
                    session_id=winner.ag_session_id,
                    now=now,
                    title=title,
                ),
            )
        return IntegrationSessionResolution(
            binding=_host_binding(result.binding, external_identity),
            session=result.session,
            session_created=result.session_created,
            binding_created=result.binding_created,
        )

    async def get_binding(
        self,
        *,
        route: IntegrationRoute,
        external_identity: ExternalIdentity,
    ) -> ExternalSessionBinding | None:
        """Read one exact route-scoped binding from canonical persistence.

        The authenticated identity is projected to the route's exact opaque scope
        before the repository lookup.

        Examples:
            Read an existing binding:
                ```python
                binding = await store.get_binding(route=route, external_identity=identity)
                ```

            Detect an unbound identity:
                ```python
                assert await store.get_binding(
                    route=route,
                    external_identity=new_identity,
                ) is None
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
        record = await self._repository.get_binding(scope, route.route_id)
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


def _canonical_session(
    *,
    owner_scope: StorageScope,
    route: IntegrationRoute,
    external_identity: ExternalIdentity,
    session_id: str,
    now: datetime,
    title: str | None,
) -> SessionRecord:
    include_user = route.session_policy.scope in {
        "conversation_user",
        "conversation_thread_user",
    }
    dimensions = {"session_id": session_id}
    if owner_scope.org_id is None:
        dimensions["org_id"] = external_identity.tenant_id
    if include_user and owner_scope.user_id is None:
        dimensions["user_id"] = external_identity.user_id
    scope = merge_storage_scope(owner_scope, **dimensions)
    external_reference = (
        f"agent-endpoint:{route.endpoint_id}"
        if route.endpoint_id is not None
        else f"integration:{route.route_id}"
    )
    return SessionRecord(
        session_id=session_id,
        kind=SessionKind.CHAT,
        scope=scope,
        revision=1,
        created_at=now,
        updated_at=now,
        title=title,
        source=route.integration_kind.value,
        external_reference=external_reference,
        metadata={},
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

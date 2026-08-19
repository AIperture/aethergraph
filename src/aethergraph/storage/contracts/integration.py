"""Canonical Host integration idempotency and session-binding contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from .control import SessionRecord
from .records import FrozenJson, _freeze_mapping, _nonempty, _utc
from .scope import StorageScope


class IngressClaimStatus(StrEnum):
    """Canonical state of one ingress idempotency identity."""

    PENDING = "pending"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True, kw_only=True)
class IngressClaimRequest:
    """Canonical claim input computed from one validated ingress envelope."""

    deployment_id: str
    integration_id: str
    idempotency_key: str
    external_event_id: str
    envelope_digest: str
    digest_algorithm: str
    scope: StorageScope
    claimed_at: datetime

    def __post_init__(self) -> None:
        _validate_ingress_identity(
            deployment_id=self.deployment_id,
            integration_id=self.integration_id,
            idempotency_key=self.idempotency_key,
            external_event_id=self.external_event_id,
            envelope_digest=self.envelope_digest,
            digest_algorithm=self.digest_algorithm,
            claimed_at=self.claimed_at,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class IngressClaimRecord:
    """Revisioned ingress claim and optional single-assignment receipt."""

    deployment_id: str
    integration_id: str
    idempotency_key: str
    external_event_id: str
    envelope_digest: str
    digest_algorithm: str
    scope: StorageScope
    claimed_at: datetime
    status: IngressClaimStatus
    revision: int
    receipt: Mapping[str, FrozenJson] = field(default_factory=dict)
    completed_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_ingress_identity(
            deployment_id=self.deployment_id,
            integration_id=self.integration_id,
            idempotency_key=self.idempotency_key,
            external_event_id=self.external_event_id,
            envelope_digest=self.envelope_digest,
            digest_algorithm=self.digest_algorithm,
            claimed_at=self.claimed_at,
        )
        if not isinstance(self.status, IngressClaimStatus):
            raise TypeError("status must be an IngressClaimStatus")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        receipt = _freeze_mapping(self.receipt, path="receipt")
        object.__setattr__(self, "receipt", receipt)
        if self.status is IngressClaimStatus.PENDING:
            if receipt or self.completed_at is not None:
                raise ValueError("pending ingress claims must not have a receipt")
        else:
            if not receipt or self.completed_at is None:
                raise ValueError("completed ingress claims require a receipt and completed_at")
            _utc("completed_at", self.completed_at)
            if self.completed_at < self.claimed_at:
                raise ValueError("completed_at must not precede claimed_at")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")


def _validate_ingress_identity(
    *,
    deployment_id: str,
    integration_id: str,
    idempotency_key: str,
    external_event_id: str,
    envelope_digest: str,
    digest_algorithm: str,
    claimed_at: datetime,
) -> None:
    for name, value in (
        ("deployment_id", deployment_id),
        ("integration_id", integration_id),
        ("idempotency_key", idempotency_key),
        ("external_event_id", external_event_id),
        ("envelope_digest", envelope_digest),
        ("digest_algorithm", digest_algorithm),
    ):
        _nonempty(name, value)
    _utc("claimed_at", claimed_at)


@dataclass(frozen=True, slots=True)
class IngressClaimResult:
    """Atomic ingress-claim result and current authoritative record."""

    record: IngressClaimRecord
    acquired: bool

    def __post_init__(self) -> None:
        if not isinstance(self.acquired, bool):
            raise TypeError("acquired must be a boolean")
        if self.acquired and self.record.status is not IngressClaimStatus.PENDING:
            raise ValueError("only a pending ingress claim may be newly acquired")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalSessionBindingRequest:
    """Candidate for atomic external-scope to AG-session binding."""

    binding_id: str
    route_id: str
    build_id: str
    ag_session_id: str
    scope: StorageScope
    now: datetime

    def __post_init__(self) -> None:
        _validate_binding_identity(
            binding_id=self.binding_id,
            route_id=self.route_id,
            build_id=self.build_id,
            ag_session_id=self.ag_session_id,
            scope=self.scope,
        )
        _utc("now", self.now)


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalSessionBindingRecord:
    """Revisioned build-pinned external scope to AG session binding."""

    binding_id: str
    route_id: str
    build_id: str
    ag_session_id: str
    scope: StorageScope
    revision: int
    created_at: datetime
    last_seen_at: datetime
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_binding_identity(
            binding_id=self.binding_id,
            route_id=self.route_id,
            build_id=self.build_id,
            ag_session_id=self.ag_session_id,
            scope=self.scope,
        )
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("created_at", self.created_at)
        _utc("last_seen_at", self.last_seen_at)
        if self.last_seen_at < self.created_at:
            raise ValueError("last_seen_at must not precede created_at")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


def _validate_binding_identity(
    *,
    binding_id: str,
    route_id: str,
    build_id: str,
    ag_session_id: str,
    scope: StorageScope,
) -> None:
    for name, value in (
        ("binding_id", binding_id),
        ("route_id", route_id),
        ("build_id", build_id),
        ("ag_session_id", ag_session_id),
    ):
        _nonempty(name, value)
    scope.require("scope_key")
    if scope.run_id is not None or scope.node_id is not None:
        raise ValueError("external session binding scope must not contain run_id or node_id")


@dataclass(frozen=True, slots=True)
class IntegrationSessionProvisioningResult:
    """Atomic canonical-session and external-binding provisioning result."""

    session: SessionRecord
    binding: ExternalSessionBindingRecord
    session_created: bool
    binding_created: bool

    def __post_init__(self) -> None:
        if self.session.session_id != self.binding.ag_session_id:
            raise ValueError("provisioned session and binding identities must match")
        for name in ("session_created", "binding_created"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")


class IngressIdempotencyRepository(Protocol):
    """Transactional ingress claims keyed by deployment and integration identity."""

    async def claim(self, request: IngressClaimRequest) -> IngressClaimResult:
        """Atomically acquire or inspect one canonical ingress identity.

        Deployment, integration, idempotency key, and external event uniqueness are
        checked with the immutable envelope digest in one transaction.

        Examples:
            Acquire new ingress work:
                ```python
                result = await ingress.claim(request)
                ```

            Replay a completed receipt:
                ```python
                if result.record.status is IngressClaimStatus.COMPLETED:
                    return result.record.receipt
                ```

        Args:
            request: Canonical identity, digest, scope, and claim timestamp.

        Returns:
            IngressClaimResult: New ownership or the current pending/completed record.

        Notes:
            Key or external-event reuse with different identity/digest raises
            `StorageIntegrityError`; no alternate key is tried.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        deployment_id: str,
        integration_id: str,
        idempotency_key: str,
    ) -> IngressClaimRecord | None:
        """Read one ingress claim or receipt by exact scoped identity.

        The provider performs a direct lookup without considering external event
        aliases after a miss.

        Examples:
            Inspect a claim:
                ```python
                record = await ingress.get(scope, deployment_id, integration_id, key)
                ```

            Detect absence:
                ```python
                assert await ingress.get(scope, deployment_id, integration_id, "missing") is None
                ```

        Args:
            scope: Canonical Host owner scope constraining access.
            deployment_id: Exact Host deployment identity.
            integration_id: Exact configured integration identity.
            idempotency_key: Exact provider ingress idempotency key.

        Returns:
            IngressClaimRecord | None: Current claim/receipt or `None` when absent.

        Notes:
            The external event identity is independently unique but is not a fallback
            lookup path for this method.
        """
        ...

    async def complete(
        self,
        record: IngressClaimRecord,
        expected_revision: int,
    ) -> IngressClaimRecord:
        """Atomically assign the single terminal receipt to an acquired claim.

        The complete record must preserve every immutable claim field and advance to
        the exact next revision.

        Examples:
            Complete accepted ingress:
                ```python
                stored = await ingress.complete(completed, pending.revision)
                ```

            Complete a stable rejection:
                ```python
                stored = await ingress.complete(rejected, pending.revision)
                ```

        Args:
            record: Complete canonical next revision containing terminal receipt.
            expected_revision: Current pending revision required for completion.

        Returns:
            IngressClaimRecord: Newly committed terminal receipt record.

        Notes:
            Completion is single assignment. Recompletion or stale ownership raises
            `StorageConflictError`; persisted receipts never carry duplicate-response state.
        """
        ...


class IntegrationSessionRepository(Protocol):
    """Atomic canonical-session and external-binding persistence boundary."""

    async def provision(
        self,
        request: ExternalSessionBindingRequest,
        session: SessionRecord,
    ) -> IntegrationSessionProvisioningResult:
        """Provision one compatible canonical session and external binding.

        The provider creates or validates both records in one transaction and repairs
        an existing orphan binding at its authoritative session identity.

        Examples:
            Provision one conversation:
                ```python
                result = await sessions.provision(request, session)
                ```

            Detect an idempotent replay:
                ```python
                assert not (result.session_created or result.binding_created)
                ```

        Args:
            request: Candidate identities, canonical external scope, and timestamp.
            session: Candidate canonical session using the requested AG session ID.

        Returns:
            IntegrationSessionProvisioningResult: Authoritative records and creation flags.

        Notes:
            Immutable conflicts raise `StorageIntegrityError`; competing candidates
            roll back without leaving orphan sessions.
        """
        ...

    async def get_binding(
        self,
        scope: StorageScope,
        route_id: str,
    ) -> ExternalSessionBindingRecord | None:
        """Read one binding by exact route and canonical external scope key.

        AG computes the route-authored scope key before the provider call, keeping
        integration session-policy semantics outside storage.

        Examples:
            Read a conversation binding:
                ```python
                binding = await sessions.get_binding(external_scope, "route-1")
                ```

            Detect an unbound conversation:
                ```python
                assert await sessions.get_binding(new_scope, "route-1") is None
                ```

        Args:
            scope: Canonical Host owner scope containing the exact opaque scope key.
            route_id: Exact manifest route identity.

        Returns:
            ExternalSessionBindingRecord | None: Current binding or `None`.

        Notes:
            Provider code does not import or interpret Host route/session-policy DTOs.
        """
        ...

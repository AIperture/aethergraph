"""Provider selection, open request, bundle lifecycle, and provider protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from .capabilities import StorageCapabilities
from .continuations import ContinuationLeaseRepository, ContinuationRepository
from .control import RunRepository, RunResultRepository, SessionRepository
from .integration import ExternalSessionBindingRepository, IngressIdempotencyRepository
from .observations import ObservationRepository
from .scope import StorageScope
from .stores import ArtifactRepository, BlobStore, EventStore, SearchBackend, StateStore
from .supporting import DocumentStore, KeyValueStore
from .triggers import TriggerRepository


class StorageOpenMode(StrEnum):
    """Explicit access mode for one provider bundle."""

    READ_WRITE = "read_write"
    READ_ONLY = "read_only"


class StorageClock(Protocol):
    """Clock boundary supplied to providers for durable timestamps and leases."""

    def now(self) -> datetime:
        """Return the current timezone-aware timestamp.

        The provider calls this boundary instead of reading the system clock directly
        when durable ordering, TTL, or lease behavior is under test.

        Examples:
            Read the production clock:
                ```python
                timestamp = clock.now()
                ```

            Read a deterministic test clock:
                ```python
                assert fake_clock.now() == expected
                ```

        Args:
            None.

        Returns:
            datetime: Current timezone-aware timestamp.

        Notes:
            Implementations must not return naive timestamps.
        """
        ...


class StorageSecretResolver(Protocol):
    """Resolve opaque secret references without serializing resolved credentials."""

    async def resolve(self, reference: str) -> str | bytes:
        """Resolve one exact opaque secret reference.

        Resolution may contact an external secret boundary. The returned credential
        stays in process and must not be copied into a manifest or provider config.

        Examples:
            Resolve a continuation HMAC secret:
                ```python
                secret = await resolver.resolve("secret://continuations")
                ```

            Resolve an external database credential:
                ```python
                password = await resolver.resolve("vault://storage/password")
                ```

        Args:
            reference: Exact opaque secret reference selected by trusted composition.

        Returns:
            str | bytes: Resolved credential material for immediate provider use.

        Notes:
            Providers must not persist the resolved return value.
        """
        ...


@dataclass(frozen=True, slots=True)
class StorageProviderSelection:
    """Exact provider name and immutable validated-configuration input."""

    provider: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("provider must be a non-empty exact name")
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))


@dataclass(frozen=True, slots=True)
class StorageOpenRequest:
    """Complete trusted input for opening exactly one provider bundle."""

    workspace_id: str
    workspace_root: Path
    owner_scope: StorageScope
    selection: StorageProviderSelection
    mode: StorageOpenMode
    expected_format_version: int
    clock: StorageClock
    secrets: StorageSecretResolver

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_id, str) or not self.workspace_id.strip():
            raise ValueError("workspace_id must be a non-empty string")
        if not isinstance(self.workspace_root, Path) or not self.workspace_root.is_absolute():
            raise ValueError("workspace_root must be an authorized absolute Path")
        if isinstance(self.expected_format_version, bool) or self.expected_format_version < 1:
            raise ValueError("expected_format_version must be a positive integer")


@dataclass(frozen=True, slots=True)
class StorageHealth:
    """Provider readiness result returned without exposing provider-private handles."""

    ready: bool
    detail: str = ""


class StorageBundle(Protocol):
    """Lifecycle surface of one coherent provider-owned store collection."""

    provider_name: str
    capabilities: StorageCapabilities
    format_version: int
    mode: StorageOpenMode
    events: EventStore
    memory_events: EventStore
    state: StateStore
    blobs: BlobStore
    artifacts: ArtifactRepository
    search: SearchBackend
    kv: KeyValueStore
    documents: DocumentStore
    auth_grants: KeyValueStore
    auth_invites: KeyValueStore
    registry_manifests: DocumentStore
    runs: RunRepository
    run_results: RunResultRepository
    sessions: SessionRepository
    continuations: ContinuationRepository
    continuation_leases: ContinuationLeaseRepository
    triggers: TriggerRepository
    observations: ObservationRepository
    ingress_idempotency: IngressIdempotencyRepository
    external_session_bindings: ExternalSessionBindingRepository

    async def health(self) -> StorageHealth:
        """Return current readiness for the already-open bundle.

        The check covers provider-owned shared resources and does not open another
        bundle or select another provider.

        Examples:
            Check a local bundle:
                ```python
                status = await bundle.health()
                ```

            Fail a readiness gate:
                ```python
                if not (await bundle.health()).ready:
                    raise RuntimeError("storage unavailable")
                ```

        Args:
            None.

        Returns:
            StorageHealth: Current readiness and a bounded diagnostic detail.

        Notes:
            Focused typed store fields are added as their S1 protocols are normalized;
            untyped service lookup is intentionally absent.
        """
        ...

    async def close(self) -> None:
        """Close every provider-owned resource exactly once.

        The bundle owns connection pools, background maintenance, and shared handles.
        Callers may invoke close repeatedly; implementations remain idempotent.

        Examples:
            Close after runtime shutdown:
                ```python
                await bundle.close()
                ```

            Close safely after partial startup:
                ```python
                try:
                    await start_services(bundle)
                finally:
                    await bundle.close()
                ```

        Args:
            None.

        Returns:
            None: All bundle-owned resources are closed or were already closed.

        Notes:
            Services must not close individual stores from the bundle.
        """
        ...


class StorageProvider(Protocol):
    """Open one coherent bundle for an exact provider selection."""

    name: str

    def validate_config(self, selection: StorageProviderSelection) -> None:
        """Validate provider-specific configuration before opening resources.

        Validation is deterministic and side-effect free. Unknown keys or malformed
        values fail directly instead of selecting local defaults.

        Examples:
            Validate local configuration:
                ```python
                provider.validate_config(selection)
                ```

            Reject external configuration before open:
                ```python
                with pytest.raises(StorageConfigurationError):
                    provider.validate_config(invalid_selection)
                ```

        Args:
            selection: Exact selected provider name and immutable raw config.

        Returns:
            None: The selection is valid for this provider.

        Notes:
            Implementations raise typed storage configuration errors on failure.
        """
        ...

    async def open(self, request: StorageOpenRequest) -> StorageBundle:
        """Open one bundle in the request's explicit access mode.

        The provider validates format and capabilities, owns all created resources,
        and never opens or falls back to a different provider.

        Examples:
            Open a writable workspace:
                ```python
                bundle = await provider.open(request)
                ```

            Open a historical workspace read-only:
                ```python
                readonly = replace(request, mode=StorageOpenMode.READ_ONLY)
                bundle = await provider.open(readonly)
                ```

        Args:
            request: Complete trusted provider-open request.

        Returns:
            StorageBundle: One coherent provider-owned bundle.

        Notes:
            Unsupported mode, format, or capability requirements fail closed.
        """
        ...

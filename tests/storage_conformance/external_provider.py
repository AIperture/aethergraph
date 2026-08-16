"""Deterministic filesystem-free external provider used by shared conformance tests."""

from __future__ import annotations

from dataclasses import dataclass

from test_storage_focused_protocols import (
    _ArtifactRepository,
    _BlobStore,
    _EventStore,
    _SearchBackend,
    _StateStore,
)

from aethergraph.storage.contracts import (
    StorageBundle,
    StorageCapabilities,
    StorageCapability,
    StorageConfigurationError,
    StorageHealth,
    StorageHealthError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
)
from storage_conformance.runtime_repositories import (
    InMemoryContinuationLeaseRepository,
    InMemoryContinuationRepository,
    InMemoryDeliveryCursorAllocator,
    InMemoryDocumentStore,
    InMemoryExternalSessionBindingRepository,
    InMemoryInboundEventRepository,
    InMemoryIngressIdempotencyRepository,
    InMemoryKeyValueStore,
    InMemoryObservationRepository,
    InMemoryRunRepository,
    InMemoryRunResultRepository,
    InMemoryRuntimeOutputSink,
    InMemorySemanticEventRepository,
    InMemorySessionRepository,
    InMemoryTriggerRepository,
)

EXTERNAL_PROVIDER_NAME = "test.external"


@dataclass(slots=True)
class SharedResource:
    close_calls: int = 0
    closed: bool = False


class DeterministicExternalBundle:
    provider_name = EXTERNAL_PROVIDER_NAME
    capabilities = StorageCapabilities.of(
        StorageCapability.DURABLE,
        StorageCapability.TRANSACTIONS,
        StorageCapability.ATOMIC_COMPARE_AND_SET,
        StorageCapability.ORDERED_APPEND,
        StorageCapability.MONOTONIC_CURSORS,
        StorageCapability.SHARED_DELIVERY_CURSOR,
        StorageCapability.TTL,
        StorageCapability.LEASES,
        StorageCapability.BLOB_STREAMING,
        StorageCapability.BLOB_RANGE_READ,
        StorageCapability.SEARCH_STRUCTURAL,
        StorageCapability.SEARCH_SEMANTIC,
        StorageCapability.SEARCH_LEXICAL,
        StorageCapability.SEARCH_HYBRID,
        StorageCapability.READ_ONLY_OPEN,
        StorageCapability.HEALTH,
    )
    format_version = 1

    def __init__(
        self,
        mode: StorageOpenMode,
        *,
        clock,
        ready: bool,
        close_failures: int,
    ) -> None:
        self.mode = mode
        self.resource = SharedResource()
        self.ready = ready
        self.health_calls = 0
        self.close_failures = close_failures
        self.events = _EventStore()
        self.memory_events = _EventStore()
        self.state = _StateStore()
        self.blobs = _BlobStore()
        self.artifacts = _ArtifactRepository(self.blobs)
        self.search = _SearchBackend()
        self.kv = InMemoryKeyValueStore(clock)
        self.documents = InMemoryDocumentStore(clock)
        self.auth_grants = InMemoryKeyValueStore(clock)
        self.auth_invites = InMemoryKeyValueStore(clock)
        self.registry_manifests = InMemoryDocumentStore(clock)
        self.runs = InMemoryRunRepository()
        self.run_results = InMemoryRunResultRepository()
        self.sessions = InMemorySessionRepository()
        self.continuations = InMemoryContinuationRepository()
        self.continuation_leases = InMemoryContinuationLeaseRepository()
        self.triggers = InMemoryTriggerRepository()
        self.observations = InMemoryObservationRepository(clock)
        self.ingress_idempotency = InMemoryIngressIdempotencyRepository()
        self.external_session_bindings = InMemoryExternalSessionBindingRepository()
        self.inbound_events = InMemoryInboundEventRepository()
        delivery_cursors = InMemoryDeliveryCursorAllocator()
        self.semantic_events = InMemorySemanticEventRepository(delivery_cursors)
        self.runtime_output = InMemoryRuntimeOutputSink(delivery_cursors)

    async def health(self) -> StorageHealth:
        self.health_calls += 1
        if self.resource.closed:
            return StorageHealth(ready=False, detail="closed")
        return StorageHealth(ready=self.ready, detail="ready" if self.ready else "unavailable")

    async def close(self) -> None:
        if self.resource.closed:
            return
        self.resource.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise StorageHealthError("injected external flush failure")
        self.resource.closed = True


class DeterministicExternalProvider:
    name = EXTERNAL_PROVIDER_NAME

    def __init__(self, *, ready: bool = True, close_failures: int = 0) -> None:
        self.ready = ready
        self.close_failures = close_failures
        self.open_calls = 0
        self.bundles: list[DeterministicExternalBundle] = []

    def validate_config(self, selection: StorageProviderSelection) -> None:
        if selection.provider != self.name:
            raise StorageConfigurationError("external selection does not match provider")
        unknown = set(selection.config) - {"endpoint", "credential_ref"}
        if unknown:
            raise StorageConfigurationError(f"unknown external provider keys: {sorted(unknown)}")
        if selection.config.get("endpoint") != "memory://external-conformance":
            raise StorageConfigurationError("external endpoint is invalid")
        credential_ref = selection.config.get("credential_ref")
        if not isinstance(credential_ref, str) or not credential_ref.startswith("secret://"):
            raise StorageConfigurationError("external credential_ref is invalid")

    def open(self, request: StorageOpenRequest) -> StorageBundle:
        self.validate_config(request.selection)
        if request.expected_format_version != self.format_version:
            raise StorageConfigurationError("external format version is unsupported")
        self.open_calls += 1
        bundle = DeterministicExternalBundle(
            request.mode,
            clock=request.clock,
            ready=self.ready,
            close_failures=self.close_failures,
        )
        self.bundles.append(bundle)
        return bundle  # type: ignore[return-value]

    @property
    def format_version(self) -> int:
        return DeterministicExternalBundle.format_version

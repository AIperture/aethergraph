from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    ArtifactRepository,
    BlobStore,
    ContinuationLeaseRepository,
    ContinuationRepository,
    DocumentStore,
    EventStore,
    InboundEventRepository,
    IngressIdempotencyRepository,
    IntegrationSessionRepository,
    KeyValueStore,
    ObservationRepository,
    RunRepository,
    RunResultRepository,
    RuntimeOutputSink,
    SearchBackend,
    SemanticEventRepository,
    SessionRepository,
    StateStore,
    StorageBundle,
    StorageCapabilities,
    StorageCapability,
    StorageConfigurationError,
    StorageHealth,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
    TriggerRepository,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry

BUNDLE_STORE_TYPES = {
    "events": EventStore,
    "memory_events": EventStore,
    "state": StateStore,
    "blobs": BlobStore,
    "artifacts": ArtifactRepository,
    "search": SearchBackend,
    "kv": KeyValueStore,
    "documents": DocumentStore,
    "auth_grants": KeyValueStore,
    "auth_invites": KeyValueStore,
    "registry_manifests": DocumentStore,
    "runs": RunRepository,
    "run_results": RunResultRepository,
    "sessions": SessionRepository,
    "continuations": ContinuationRepository,
    "continuation_leases": ContinuationLeaseRepository,
    "triggers": TriggerRepository,
    "observations": ObservationRepository,
    "ingress_idempotency": IngressIdempotencyRepository,
    "integration_sessions": IntegrationSessionRepository,
    "inbound_events": InboundEventRepository,
    "semantic_events": SemanticEventRepository,
    "runtime_output": RuntimeOutputSink,
}


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 14, 12, tzinfo=UTC)


class _Secrets:
    async def resolve(self, reference: str) -> str:
        return f"resolved:{reference}"


@dataclass(frozen=True, slots=True)
class _StoreHandle:
    name: str
    connection: _SharedConnection


@dataclass(slots=True)
class _SharedConnection:
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class _ExternalBundle:
    provider_name = "company.external"
    capabilities = StorageCapabilities.of(
        StorageCapability.DURABLE,
        StorageCapability.TRANSACTIONS,
        StorageCapability.HEALTH,
    )
    format_version = 1

    def __init__(self, mode: StorageOpenMode) -> None:
        self.mode = mode
        self.connection = _SharedConnection()
        self.closed = False
        for name in BUNDLE_STORE_TYPES:
            setattr(self, name, _StoreHandle(name=name, connection=self.connection))

    async def health(self) -> StorageHealth:
        return StorageHealth(ready=not self.closed, detail="closed" if self.closed else "ready")

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.connection.close()


class _ExternalProvider:
    name = "company.external"
    open_calls = 0

    def validate_config(self, selection: StorageProviderSelection) -> None:
        if selection.provider != self.name:
            raise StorageConfigurationError("selection does not match provider")
        unknown = set(selection.config) - {"endpoint", "credential_ref"}
        if unknown:
            raise StorageConfigurationError(f"unknown external provider keys: {sorted(unknown)}")

    def open(self, request: StorageOpenRequest) -> StorageBundle:
        self.validate_config(request.selection)
        if request.expected_format_version != 1:
            raise StorageConfigurationError("unsupported format version")
        type(self).open_calls += 1
        return _ExternalBundle(request.mode)  # type: ignore[return-value]


def _request(tmp_path: Path, provider: str = "company.external") -> StorageOpenRequest:
    return StorageOpenRequest(
        workspace_id="workspace-1",
        workspace_root=tmp_path.resolve(),
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        selection=StorageProviderSelection(
            provider=provider,
            config={"endpoint": "postgresql://storage", "credential_ref": "secret://storage"},
        ),
        mode=StorageOpenMode.READ_ONLY,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )


def test_bundle_exposes_every_store_as_an_exact_typed_field() -> None:
    hints = get_type_hints(StorageBundle)

    assert {name: hints[name] for name in BUNDLE_STORE_TYPES} == BUNDLE_STORE_TYPES
    assert set(hints) == {
        "provider_name",
        "capabilities",
        "format_version",
        "mode",
        *BUNDLE_STORE_TYPES,
    }
    assert not hasattr(StorageBundle, "stores")
    assert not hasattr(StorageBundle, "get_store")
    assert not hasattr(StorageBundle, "__getitem__")


@pytest.mark.asyncio
async def test_external_provider_opens_one_coherent_lifecycle_owned_bundle(tmp_path: Path) -> None:
    provider = StorageProviderRegistry({"company.external": _ExternalProvider}).create(
        "company.external"
    )
    request = _request(tmp_path)
    provider.validate_config(request.selection)

    bundle = provider.open(request)

    assert bundle.provider_name == "company.external"
    assert bundle.mode is StorageOpenMode.READ_ONLY
    assert bundle.format_version == request.expected_format_version
    assert (await bundle.health()).ready is True
    handles = [getattr(bundle, name) for name in BUNDLE_STORE_TYPES]
    assert [handle.name for handle in handles] == list(BUNDLE_STORE_TYPES)
    assert len({id(handle.connection) for handle in handles}) == 1

    await bundle.close()
    await bundle.close()

    assert bundle.connection.close_calls == 1  # type: ignore[attr-defined]
    assert (await bundle.health()).ready is False


@pytest.mark.asyncio
async def test_external_selection_failure_never_creates_or_opens_local_provider(
    tmp_path: Path,
) -> None:
    local_factory_calls = 0

    def local_factory() -> _ExternalProvider:
        nonlocal local_factory_calls
        local_factory_calls += 1
        return _ExternalProvider()

    registry = StorageProviderRegistry(
        {
            "company.external": _ExternalProvider,
            "local.sqlite": local_factory,
        }
    )
    provider = registry.create("company.external")

    with pytest.raises(StorageConfigurationError, match="unknown external provider keys"):
        invalid = _request(tmp_path)
        invalid = StorageOpenRequest(
            workspace_id=invalid.workspace_id,
            workspace_root=invalid.workspace_root,
            owner_scope=invalid.owner_scope,
            selection=StorageProviderSelection(
                provider="company.external",
                config={"endpoint": "postgresql://storage", "fallback": "local.sqlite"},
            ),
            mode=invalid.mode,
            expected_format_version=invalid.expected_format_version,
            clock=invalid.clock,
            secrets=invalid.secrets,
        )
        provider.open(invalid)

    assert local_factory_calls == 0

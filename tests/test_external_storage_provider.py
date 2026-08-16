from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from storage_conformance.external_provider import (
    EXTERNAL_PROVIDER_NAME,
    DeterministicExternalProvider,
)
from storage_conformance.suite import (
    check_artifact_repository_conformance,
    check_blob_store_conformance,
    check_event_store_conformance,
    check_search_backend_conformance,
    check_state_store_conformance,
)

from aethergraph.contracts.services.state_stores import GraphSnapshot, StateEvent
from aethergraph.services.state_stores.canonical_store import CanonicalGraphStateStore
from aethergraph.storage.composition import StorageComposition
from aethergraph.storage.contracts import (
    DuplicateStorageProviderError,
    StorageCapability,
    StorageCapabilityError,
    StorageConfigurationError,
    StorageHealthError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
    UnknownStorageProviderError,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 16, 12, tzinfo=UTC)


class _Secrets:
    async def resolve(self, reference: str) -> str:
        return f"resolved:{reference}"


def _request(
    tmp_path: Path,
    *,
    provider: str = EXTERNAL_PROVIDER_NAME,
    config: dict[str, object] | None = None,
) -> StorageOpenRequest:
    return StorageOpenRequest(
        workspace_id="external-workspace",
        workspace_root=tmp_path.resolve(),
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        selection=StorageProviderSelection(
            provider=provider,
            config=config
            or {
                "endpoint": "memory://external-conformance",
                "credential_ref": "secret://external-storage",
            },
        ),
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )


def _composition(
    provider: DeterministicExternalProvider,
    *,
    required: frozenset[StorageCapability] = frozenset(),
) -> StorageComposition:
    registry = StorageProviderRegistry()
    registry.register(EXTERNAL_PROVIDER_NAME, lambda: provider)
    return StorageComposition(registry, required)


@pytest.mark.asyncio
async def test_external_provider_passes_shared_store_and_composition_conformance(
    tmp_path: Path,
) -> None:
    provider = DeterministicExternalProvider()
    composition = _composition(
        provider,
        required=frozenset(
            {
                StorageCapability.DURABLE,
                StorageCapability.ATOMIC_COMPARE_AND_SET,
                StorageCapability.ORDERED_APPEND,
                StorageCapability.BLOB_STREAMING,
                StorageCapability.SEARCH_HYBRID,
                StorageCapability.HEALTH,
            }
        ),
    )

    bundle = await composition.open(_request(tmp_path))
    await check_event_store_conformance(bundle.events)
    await check_event_store_conformance(bundle.memory_events)
    await check_state_store_conformance(bundle.state)
    await check_blob_store_conformance(bundle.blobs)
    await check_artifact_repository_conformance(bundle.artifacts, bundle.blobs)
    await check_search_backend_conformance(bundle.search)

    graph_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="runtime-run-1",
        graph_id="runtime-graph-1",
    )
    graph_state = CanonicalGraphStateStore(
        state_store=bundle.state,
        event_store=bundle.events,
        run_repository=bundle.runs,
    )
    snapshot = GraphSnapshot(
        run_id="runtime-run-1",
        graph_id="runtime-graph-1",
        rev=1,
        created_at=1.0,
        spec_hash="spec-1",
        state={"nodes": {}},
        started_at=datetime(2026, 8, 16, 12, tzinfo=UTC),
    )
    event = StateEvent(
        run_id="runtime-run-1",
        graph_id="runtime-graph-1",
        rev=2,
        ts=datetime(2026, 8, 16, 12, 1, tzinfo=UTC).timestamp(),
        kind="STATUS",
        payload={"node_id": "node-1", "status": "RUNNING"},
    )
    await graph_state.save_snapshot(graph_scope, snapshot)
    await graph_state.append_event(graph_scope, event)
    assert await graph_state.load_latest_snapshot(graph_scope, "runtime-run-1") == snapshot
    assert await graph_state.load_events_since(graph_scope, "runtime-run-1", 1) == [event]

    assert provider.open_calls == 1
    assert (await composition.health()).ready is True
    assert tuple(tmp_path.iterdir()) == ()

    await composition.close()
    await composition.close()
    assert provider.bundles[0].resource.close_calls == 1
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.asyncio
async def test_external_selection_errors_do_not_construct_local_storage(tmp_path: Path) -> None:
    local_factory_calls = 0

    def local_factory() -> DeterministicExternalProvider:
        nonlocal local_factory_calls
        local_factory_calls += 1
        return DeterministicExternalProvider()

    registry = StorageProviderRegistry({"local.sqlite": local_factory})
    with pytest.raises(UnknownStorageProviderError, match=EXTERNAL_PROVIDER_NAME):
        await StorageComposition(registry).open(_request(tmp_path))

    registry.register(EXTERNAL_PROVIDER_NAME, DeterministicExternalProvider)
    with pytest.raises(DuplicateStorageProviderError, match=EXTERNAL_PROVIDER_NAME):
        registry.register(EXTERNAL_PROVIDER_NAME, DeterministicExternalProvider)

    assert local_factory_calls == 0
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.asyncio
async def test_external_invalid_config_fails_without_open_or_files(tmp_path: Path) -> None:
    provider = DeterministicExternalProvider()
    composition = _composition(provider)

    with pytest.raises(StorageConfigurationError, match="unknown external provider keys"):
        await composition.open(
            _request(
                tmp_path,
                config={
                    "endpoint": "memory://external-conformance",
                    "credential_ref": "secret://external-storage",
                    "fallback": "local.sqlite",
                },
            )
        )

    assert provider.open_calls == 0
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "required", "error"),
    [
        (
            DeterministicExternalProvider(),
            frozenset({StorageCapability.TTL}),
            StorageCapabilityError,
        ),
        (
            DeterministicExternalProvider(ready=False),
            frozenset(),
            StorageHealthError,
        ),
    ],
)
async def test_external_missing_capability_and_health_failure_close_directly(
    tmp_path: Path,
    provider: DeterministicExternalProvider,
    required: frozenset[StorageCapability],
    error: type[Exception],
) -> None:
    composition = _composition(provider, required=required)

    with pytest.raises(error):
        await composition.open(_request(tmp_path))

    assert provider.open_calls == 1
    assert provider.bundles[0].resource.close_calls == 1
    assert provider.bundles[0].resource.closed is True
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.asyncio
async def test_external_close_failure_is_retryable_without_reselection(tmp_path: Path) -> None:
    provider = DeterministicExternalProvider(close_failures=1)
    composition = _composition(provider)
    await composition.open(_request(tmp_path))

    with pytest.raises(StorageHealthError, match="injected external flush failure"):
        await composition.close()
    assert provider.open_calls == 1
    assert (await composition.health()).ready is True

    await composition.close()
    assert provider.open_calls == 1
    assert provider.bundles[0].resource.close_calls == 2
    assert tuple(tmp_path.iterdir()) == ()

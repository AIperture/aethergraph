from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
from typing import get_type_hints

import pytest
from storage_conformance.external_provider import (
    EXTERNAL_PROVIDER_NAME,
    DeterministicExternalBundle,
    DeterministicExternalProvider,
)

from aethergraph.config.storage_provider import StorageProviderSettings
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.services.container.canonical_storage import (
    CanonicalStorageServices,
    bind_canonical_storage_services,
)
from aethergraph.storage.builtin_local import build_builtin_local_storage_registry
from aethergraph.storage.contracts import (
    ContinuationDraft,
    ContinuationLeaseRequest,
    DocumentQuery,
    ExternalSessionBindingRequest,
    InboundEventDraft,
    IngressClaimRequest,
    ObservationDraft,
    RunRecord,
    RunResultRecord,
    RunStatus,
    RuntimeOutputFrame,
    RuntimeOutputStream,
    SemanticEventDraft,
    SemanticEventKind,
    SemanticEventQuery,
    SessionKind,
    SessionRecord,
    StorageBundle,
    StorageCapabilities,
    StorageCapability,
    StorageCapabilityError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
    TriggerClaimRequest,
    TriggerKind,
    TriggerRecord,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry
from aethergraph.storage.runtime_requirements import (
    RUNTIME_STORAGE_CAPABILITIES,
    create_runtime_storage_composition,
)

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)
OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def now(self) -> datetime:
        return NOW


class _Secrets:
    async def resolve(self, reference: str) -> str:
        raise AssertionError(f"provider must not resolve {reference!r} during open")


class _IncompleteExternalProvider(DeterministicExternalProvider):
    def __init__(self, missing: StorageCapability) -> None:
        super().__init__()
        self._missing = missing

    def open(self, request: StorageOpenRequest) -> StorageBundle:
        bundle = super().open(request)
        bundle.capabilities = StorageCapabilities(bundle.capabilities.supported - {self._missing})
        return bundle  # type: ignore[return-value]


def _external_request(root: Path) -> StorageOpenRequest:
    return StorageOpenRequest(
        workspace_id="external-runtime-qualification",
        workspace_root=root.resolve(),
        owner_scope=OWNER,
        selection=StorageProviderSelection(
            provider=EXTERNAL_PROVIDER_NAME,
            config={
                "endpoint": "memory://external-conformance",
                "credential_ref": "secret://external-storage",
            },
        ),
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )


def _local_request(root: Path) -> StorageOpenRequest:
    return StorageOpenRequest(
        workspace_id="local-runtime-qualification",
        workspace_root=root.resolve(),
        owner_scope=OWNER,
        selection=StorageProviderSettings(provider="local.sqlite").to_selection(),
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )


def _bind_all_runtime_services(bundle: StorageBundle) -> tuple[object, ...]:
    services = bind_canonical_storage_services(
        bundle=bundle,
        owner_scope=OWNER,
        clock=_Clock().now,
        observation_policy=ObservationPolicy(capture_mode="off"),
    )
    execution_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="run-qualification",
        graph_id="graph-qualification",
        node_id="node-qualification",
        agent_id="agent-qualification",
    )
    return (
        services.key_value,
        services.metering,
        services.control,
        services.continuations,
        services.continuation_leases,
        services.triggers,
        services.auth,
        services.registration_manifests,
        services.integration,
        services.observations,
        services.runtime_output,
        services.viz,
        services.memory_factory,
        services.artifact_factory,
        services.graph_state,
        services.agent_state(execution_scope),
        services.inspection(),
    )


def test_external_bundle_implements_every_typed_repository_surface() -> None:
    bundle = DeterministicExternalBundle(
        StorageOpenMode.READ_WRITE,
        clock=_Clock(),
        ready=True,
        close_failures=0,
    )
    bundle_types = get_type_hints(StorageBundle)
    excluded = {"provider_name", "capabilities", "format_version", "mode"}

    checked: dict[str, tuple[str, ...]] = {}
    for field_name, protocol in bundle_types.items():
        if field_name in excluded:
            continue
        methods = tuple(
            name
            for name, member in inspect.getmembers(protocol, inspect.isfunction)
            if not name.startswith("_")
        )
        assert methods, f"{field_name} protocol unexpectedly has no methods"
        implementation = getattr(bundle, field_name)
        missing = tuple(
            name for name in methods if not callable(getattr(implementation, name, None))
        )
        assert missing == (), f"{field_name} is missing {missing}"
        checked[field_name] = methods

    assert set(checked) == set(bundle_types) - excluded


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [StorageCapability.TTL, StorageCapability.LEASES])
async def test_runtime_admission_rejects_incomplete_external_before_health_or_publication(
    tmp_path: Path,
    missing: StorageCapability,
) -> None:
    provider = _IncompleteExternalProvider(missing)
    registry = StorageProviderRegistry({EXTERNAL_PROVIDER_NAME: lambda: provider})
    composition = create_runtime_storage_composition(registry)

    prepared = composition.prepare(_external_request(tmp_path))
    with pytest.raises(StorageCapabilityError, match=missing.value):
        await composition.start()

    assert provider.open_calls == 1
    assert prepared is provider.bundles[0]
    assert provider.bundles[0].health_calls == 0
    assert provider.bundles[0].resource.close_calls == 1
    assert provider.bundles[0].resource.closed is True
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.asyncio
async def test_local_and_external_runtime_composition_bind_the_complete_service_surface(
    tmp_path: Path,
) -> None:
    local_root = tmp_path / "local"
    external_root = tmp_path / "external"
    local_root.mkdir()
    external_root.mkdir()
    local_selection = StorageProviderSettings(provider="local.sqlite").to_selection()
    local_registry = build_builtin_local_storage_registry(
        selection=local_selection,
        workspace_id="local-runtime-qualification",
        auth_signing_secret="qualification-auth-secret",
    )
    external_provider = DeterministicExternalProvider()
    external_registry = StorageProviderRegistry({EXTERNAL_PROVIDER_NAME: lambda: external_provider})
    local = create_runtime_storage_composition(local_registry)
    external = create_runtime_storage_composition(external_registry)

    local_bundle = await local.open(_local_request(local_root))
    external_bundle = await external.open(_external_request(external_root))
    assert len(_bind_all_runtime_services(local_bundle)) == 17
    assert len(_bind_all_runtime_services(external_bundle)) == 17
    assert local_bundle.capabilities.supported >= RUNTIME_STORAGE_CAPABILITIES
    assert external_bundle.capabilities.supported >= RUNTIME_STORAGE_CAPABILITIES
    assert tuple(external_root.iterdir()) == ()

    await local.close()
    await external.close()
    assert external_provider.open_calls == 1
    assert external_provider.bundles[0].health_calls == 1
    assert external_provider.bundles[0].resource.close_calls == 1
    assert tuple(external_root.iterdir()) == ()


@pytest.mark.asyncio
async def test_external_runtime_repositories_execute_representative_operations_without_files(
    tmp_path: Path,
) -> None:
    provider = DeterministicExternalProvider()
    registry = StorageProviderRegistry({EXTERNAL_PROVIDER_NAME: lambda: provider})
    composition = create_runtime_storage_composition(registry)
    bundle = await composition.open(_external_request(tmp_path))

    for store, namespace in (
        (bundle.kv, "runtime"),
        (bundle.auth_grants, "auth.grants"),
        (bundle.auth_invites, "auth.invites"),
    ):
        record = await store.compare_and_set(OWNER, namespace, "key-1", 0, {"ok": True})
        assert await store.get(OWNER, namespace, "key-1") == record
    for store, namespace in (
        (bundle.documents, "runtime.documents"),
        (bundle.registry_manifests, "registry.manifests"),
    ):
        record = await store.compare_and_set(
            OWNER,
            namespace,
            "document-1",
            0,
            {"kind": "qualified"},
            1,
        )
        assert (await store.query(DocumentQuery(scope=OWNER, namespace=namespace))).items == (
            record,
        )

    run_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="run-1",
        graph_id="graph-1",
    )
    run = RunRecord(
        run_id="run-1",
        graph_id="graph-1",
        kind="graph",
        status=RunStatus.RUNNING,
        scope=run_scope,
        revision=1,
        started_at=NOW,
    )
    assert await bundle.runs.create(run) == run
    result = RunResultRecord(
        run_id="run-1",
        graph_id="graph-1",
        scope=run_scope,
        status=RunStatus.SUCCEEDED,
        outputs={"answer": 42},
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        source="runtime",
    )
    assert await bundle.run_results.compare_and_set(result, 0) == result
    session_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        session_id="session-1",
    )
    session = SessionRecord(
        session_id="session-1",
        kind=SessionKind.CHAT,
        scope=session_scope,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
    )
    assert await bundle.sessions.create(session) == session

    continuation_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
    )
    continuation = await bundle.continuations.create(
        ContinuationDraft(
            continuation_id="continuation-1",
            kind="approval",
            scope=continuation_scope,
            created_at=NOW,
        )
    )
    assert await bundle.continuations.resolve_token(continuation.token) == continuation.record
    lease = await bundle.continuation_leases.claim(
        ContinuationLeaseRequest(
            fire_id="fire-1",
            continuation_id="continuation-1",
            scope=continuation_scope,
            scheduled_for=NOW,
            worker_id="worker-1",
            now=NOW,
            lease_until=NOW + timedelta(minutes=1),
        )
    )
    assert lease is not None and lease.record.fire_id == "fire-1"
    trigger_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        graph_id="graph-1",
    )
    trigger = TriggerRecord(
        trigger_id="trigger-1",
        graph_id="graph-1",
        scope=trigger_scope,
        kind=TriggerKind.ONE_SHOT,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        run_at=NOW,
        next_fire_at=NOW,
    )
    await bundle.triggers.create(trigger)
    claims = await bundle.triggers.claim_due(
        TriggerClaimRequest(
            now=NOW,
            worker_id="worker-1",
            lease_until=NOW + timedelta(minutes=1),
            limit=1,
            scope=trigger_scope,
        )
    )
    assert len(claims) == 1

    observation = await bundle.observations.append_many(
        (
            ObservationDraft(
                observation_id="observation-1",
                category="trace",
                name="qualified",
                summary="qualified external observation",
                occurred_at=NOW,
                scope=run_scope,
                trace_id="trace-1",
            ),
        )
    )
    assert await bundle.observations.get(run_scope, "observation-1") == observation[0]
    ingress = await bundle.ingress_idempotency.claim(
        IngressClaimRequest(
            deployment_id="deployment-1",
            integration_id="integration-1",
            idempotency_key="idempotency-1",
            external_event_id="external-event-1",
            envelope_digest="digest-1",
            digest_algorithm="sha256",
            scope=OWNER,
            claimed_at=NOW,
        )
    )
    assert ingress.acquired is True
    binding_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        session_id="session-1",
        scope_key="route-1",
    )
    binding = await bundle.external_session_bindings.get_or_create(
        ExternalSessionBindingRequest(
            binding_id="binding-1",
            route_id="route-1",
            build_id="build-1",
            ag_session_id="session-1",
            scope=binding_scope,
            now=NOW,
        )
    )
    assert binding.created is True
    inbound = await bundle.inbound_events.append(
        InboundEventDraft(
            event_id="inbound-1",
            deployment_id="deployment-1",
            route_id="route-1",
            integration_id="integration-1",
            external_event_id="external-event-1",
            received_at=NOW,
            scope=session_scope,
        )
    )
    assert await bundle.inbound_events.get(OWNER, inbound.event_id) == inbound
    semantic = await bundle.semantic_events.append(
        SemanticEventDraft(
            event_id="semantic-1",
            deployment_id="deployment-1",
            turn_id="turn-1",
            sequence=1,
            producer="runtime",
            occurred_at=NOW,
            kind=SemanticEventKind.INPUT_ACCEPTED,
            scope=session_scope,
        )
    )
    semantic_page = await bundle.semantic_events.query(
        SemanticEventQuery(deployment_id="deployment-1", scope=session_scope)
    )
    assert semantic_page.items == (semantic,)
    frame = RuntimeOutputFrame(
        output_id="output-1",
        execution_id="execution-1",
        scope=continuation_scope,
        stream=RuntimeOutputStream.STDOUT,
        sequence=1,
        text="qualified",
        source="runtime",
    )
    bundle.runtime_output.emit(frame)
    await bundle.runtime_output.flush_execution("execution-1")
    await bundle.runtime_output.flush_run("run-1")

    await composition.close()
    assert tuple(tmp_path.iterdir()) == ()


def test_runtime_storage_composition_factory_docstring_and_exact_capabilities() -> None:
    docstring = inspect.getdoc(create_runtime_storage_composition)

    assert docstring is not None
    required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
    positions = tuple(docstring.index(section) for section in required)
    assert positions == tuple(sorted(positions))
    assert docstring.count("```python") >= 2
    assert {StorageCapability.TTL, StorageCapability.LEASES} <= RUNTIME_STORAGE_CAPABILITIES
    assert StorageCapability.SEARCH_SEMANTIC not in RUNTIME_STORAGE_CAPABILITIES
    assert StorageCapability.SEARCH_HYBRID not in RUNTIME_STORAGE_CAPABILITIES


def test_canonical_storage_service_binding_is_scoped_bundle_hidden_and_documented() -> None:
    bundle = DeterministicExternalBundle(
        StorageOpenMode.READ_WRITE,
        clock=_Clock(),
        ready=True,
        close_failures=0,
    )
    services = bind_canonical_storage_services(
        bundle=bundle,
        owner_scope=OWNER,
        clock=_Clock().now,
        observation_policy=ObservationPolicy(capture_mode="off"),
    )

    assert not hasattr(services, "bundle")
    scoped = services.agent_state(
        StorageScope(run_id="run-1", graph_id="graph-1", node_id="node-1")
    )
    assert scoped.scope == StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
    )
    with pytest.raises(ValueError, match="conflicts with owner_scope project_id"):
        services.agent_state(StorageScope(project_id="other", run_id="run-1"))

    for public_api in (
        bind_canonical_storage_services,
        CanonicalStorageServices.agent_state,
        CanonicalStorageServices.inspection,
    ):
        docstring = inspect.getdoc(public_api)
        assert docstring is not None
        required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
        positions = tuple(docstring.index(section) for section in required)
        assert positions == tuple(sorted(positions))
        assert docstring.count("```python") >= 2

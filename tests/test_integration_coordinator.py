from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from aethergraph.contracts.integration import (
    ExternalIdentity,
    ExternalSessionBinding,
    HostManifest,
    IngressAttachment,
    IngressChoice,
    IngressEnvelope,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    OriginAddress,
    SemanticEventKind,
)
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore
from aethergraph.services.integration import (
    BindingResolution,
    EventLogInboundEventStore,
    IntegrationIngressCoordinator,
    InteractionResolutionError,
    InteractionResolver,
    ManifestRouteResolver,
    ResourceIngress,
    ResourceIngressError,
    ResourceIngressPolicy,
    SQLiteIngressIdempotencyStore,
    VerifiedIntegrationContext,
    install_integration_ingress,
)
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog

_NOW = datetime(2026, 8, 3, tzinfo=UTC)
_DIGEST = "b" * 64


def _route(*, enabled: bool = True, attachments: bool = True) -> IntegrationRoute:
    return IntegrationRoute(
        route_id="route-slack",
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        entry_agent_id="agent.support",
        enabled=enabled,
        match_policy=IntegrationMatchPolicy(),
        session_policy=IntegrationSessionPolicy(scope="conversation_thread"),
        required_capabilities=IntegrationCapabilities(
            event_kinds=(SemanticEventKind.MESSAGE_COMPLETED,),
            streaming=False,
            interactions=True,
            attachments=attachments,
            cancellation=True,
        ),
    )


def _manifest(route: IntegrationRoute) -> HostManifest:
    return HostManifest(
        deployment_id="deployment-1",
        build_id="build-1",
        source_digest=_DIGEST,
        build_root="C:/compiled/build-1",
        entrypoint_module="compiled.runtime",
        entrypoint_symbol="register",
        graph_id="graph.support",
        entry_agent_id="agent.support",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        integration_routes=(route,),
        workspace_identity="workspace-1",
        manifest_digest=_DIGEST,
    )


def _identity() -> ExternalIdentity:
    return ExternalIdentity(
        tenant_id="team-T1",
        conversation_id="channel-C1",
        thread_id="thread-1",
        user_id="user-U1",
    )


def _envelope(
    *,
    event_id: str = "event-1",
    text: str | None = "Hello",
    choice: IngressChoice | None = None,
    attachments: tuple[IngressAttachment, ...] = (),
) -> IngressEnvelope:
    return IngressEnvelope(
        integration_id="slack-main",
        route_hint="route-slack",
        external_identity=_identity(),
        external_event_id=event_id,
        idempotency_key=event_id,
        received_at=_NOW,
        text=text,
        choice=choice,
        attachments=attachments,
        origin_address=OriginAddress(
            channel_key="slack:team/T1:chan/C1:thread/thread-1",
            capability_profile_id="slack-v1",
        ),
    )


def _verified() -> VerifiedIntegrationContext:
    return VerifiedIntegrationContext(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id="team-T1",
    )


def _binding() -> ExternalSessionBinding:
    return ExternalSessionBinding(
        binding_id="binding-1",
        route_id="route-slack",
        external_identity=_identity(),
        ag_session_id="session-1",
        build_id="build-1",
        created_at=_NOW,
        last_seen_at=_NOW,
    )


class _BindingStore:
    async def get_or_create(self, **kwargs) -> BindingResolution:
        return BindingResolution(binding=_binding(), created=False)


class _RootDispatcher:
    def __init__(self) -> None:
        self.calls = []

    async def start(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "run-root-1"


class _ResumeRouter:
    def __init__(self) -> None:
        self.calls = []

    async def resume(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _coordinator(
    *,
    tmp_path,
    route: IntegrationRoute,
    continuation_store,
    event_log,
    root_dispatcher,
    resume_router,
) -> IntegrationIngressCoordinator:
    manifest = _manifest(route)
    return IntegrationIngressCoordinator(
        manifest=manifest,
        route_resolver=ManifestRouteResolver(manifest),
        idempotency_store=SQLiteIngressIdempotencyStore(tmp_path / "integration.db"),
        binding_store=_BindingStore(),
        resource_ingress=ResourceIngress(container=SimpleNamespace()),
        interaction_resolver=InteractionResolver(continuation_store),
        inbound_events=EventLogInboundEventStore(event_log),
        resume_router=resume_router,
        root_dispatcher=root_dispatcher,
    )


@pytest.mark.asyncio
async def test_host_installer_binds_one_manifest_coordinator(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    container = SimpleNamespace(
        root=str(tmp_path),
        integration_ingress=None,
        host_manifest=None,
        cont_store=InMemoryContinuationStore(secret=b"test-secret"),
        eventlog=event_log,
        resume_router=_ResumeRouter(),
        run_manager=SimpleNamespace(),
    )
    manifest = _manifest(_route())

    coordinator = install_integration_ingress(container=container, manifest=manifest)

    assert container.integration_ingress is coordinator
    assert container.host_manifest is manifest
    assert coordinator.manifest is manifest
    assert (tmp_path / "integration" / "operations.db").is_file()
    with pytest.raises(RuntimeError, match="already installed"):
        install_integration_ingress(container=container, manifest=manifest)
    await event_log.close()


@pytest.mark.asyncio
async def test_coordinator_starts_one_root_turn_and_replays_receipt(tmp_path) -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    root = _RootDispatcher()
    resume = _ResumeRouter()
    coordinator = _coordinator(
        tmp_path=tmp_path,
        route=_route(),
        continuation_store=continuation_store,
        event_log=event_log,
        root_dispatcher=root,
        resume_router=resume,
    )

    receipt = await coordinator.accept(verified=_verified(), envelope=_envelope())
    duplicate = await coordinator.accept(verified=_verified(), envelope=_envelope())

    assert receipt.action == "root_turn_started"
    assert receipt.turn_id == "run-root-1"
    assert receipt.event_cursor == 1
    assert duplicate == receipt.model_copy(update={"duplicate": True})
    assert len(root.calls) == 1
    assert resume.calls == []
    await event_log.close()


@pytest.mark.asyncio
async def test_coordinator_resumes_exact_public_interaction_id(tmp_path) -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    continuation = Continuation(
        run_id="run-waiting-1",
        node_id="node-choice",
        kind="choice",
        token="internal-secret-token",
        prompt={"title": "Proceed?", "options": ["Yes", "No"]},
        session_id="session-1",
        payload={"_interaction_id": "interaction-public-1"},
    )
    await continuation_store.save(continuation)
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    root = _RootDispatcher()
    resume = _ResumeRouter()
    coordinator = _coordinator(
        tmp_path=tmp_path,
        route=_route(),
        continuation_store=continuation_store,
        event_log=event_log,
        root_dispatcher=root,
        resume_router=resume,
    )
    envelope = _envelope(
        text=None,
        choice=IngressChoice(
            interaction_id="interaction-public-1",
            option_ids=("Yes",),
        ),
    )

    receipt = await coordinator.accept(verified=_verified(), envelope=envelope)

    assert receipt.action == "continuation_resumed"
    assert receipt.turn_id == "run-waiting-1"
    assert root.calls == []
    assert resume.calls[0]["token"] == "internal-secret-token"
    assert resume.calls[0]["payload"]["interaction_id"] == "interaction-public-1"
    assert resume.calls[0]["payload"]["choice"] == "Yes"
    await event_log.close()


@pytest.mark.asyncio
async def test_interaction_resolver_rejects_ambiguous_bound_session_text() -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    for index in range(2):
        await continuation_store.save(
            Continuation(
                run_id=f"run-{index}",
                node_id=f"node-{index}",
                kind="user_input",
                token=f"token-{index}",
                session_id="session-1",
                payload={"_interaction_id": f"interaction-{index}"},
            )
        )

    with pytest.raises(InteractionResolutionError) as exc_info:
        await InteractionResolver(continuation_store).resolve(
            binding=_binding(),
            envelope=_envelope(),
        )
    assert exc_info.value.code == "integration.interaction_ambiguous"


@pytest.mark.asyncio
async def test_coordinator_persists_disabled_route_rejection(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    root = _RootDispatcher()
    coordinator = _coordinator(
        tmp_path=tmp_path,
        route=_route(enabled=False),
        continuation_store=InMemoryContinuationStore(secret=b"test-secret"),
        event_log=event_log,
        root_dispatcher=root,
        resume_router=_ResumeRouter(),
    )

    receipt = await coordinator.accept(verified=_verified(), envelope=_envelope())
    duplicate = await coordinator.accept(verified=_verified(), envelope=_envelope())

    assert receipt.accepted is False
    assert receipt.rejection_code == "integration.route_disabled"
    assert duplicate.duplicate is True
    assert root.calls == []
    await event_log.close()


@pytest.mark.asyncio
async def test_resource_ingress_validates_existing_artifact_reference() -> None:
    artifact = SimpleNamespace(
        artifact_id="artifact-1",
        name="report.pdf",
        mime="application/pdf",
        bytes=100,
        uri="artifact://artifact-1",
        labels={"filename": "report.pdf"},
    )

    class _ArtifactIndex:
        async def get(self, artifact_id):
            return artifact if artifact_id == "artifact-1" else None

    ingress = ResourceIngress(
        container=SimpleNamespace(artifact_index=_ArtifactIndex()),
        policy=ResourceIngressPolicy(allowed_content_types=("application/pdf",)),
    )
    envelope = _envelope(
        text=None,
        attachments=(
            IngressAttachment(
                attachment_id="attachment-1",
                source_kind="artifact",
                source_id="artifact-1",
                filename="report.pdf",
                content_type="application/pdf",
                size_bytes=100,
            ),
        ),
    )

    resources = await ingress.materialize(
        verified=_verified(),
        route=_route(),
        binding=_binding(),
        envelope=envelope,
    )

    assert len(resources) == 1
    assert resources[0].artifact_id == "artifact-1"


@pytest.mark.asyncio
async def test_resource_ingress_rejects_attachments_without_route_capability() -> None:
    ingress = ResourceIngress(container=SimpleNamespace())
    envelope = _envelope(
        text=None,
        attachments=(
            IngressAttachment(
                attachment_id="attachment-1",
                source_kind="artifact",
                source_id="artifact-1",
                filename="report.pdf",
                content_type="application/pdf",
                size_bytes=100,
            ),
        ),
    )

    with pytest.raises(ResourceIngressError) as exc_info:
        await ingress.materialize(
            verified=_verified(),
            route=_route(attachments=False),
            binding=_binding(),
            envelope=envelope,
        )
    assert exc_info.value.code == "integration.attachments_unsupported"

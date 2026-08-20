from __future__ import annotations

from datetime import UTC, datetime
import logging
from types import SimpleNamespace

import pytest

from aethergraph.config.config import AppSettings
from aethergraph.contracts.integration import (
    SEMANTIC_EVENT_PROTOCOL_VERSION,
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
from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.channel.resources import InputResource
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.continuations.continuation import ContinuationDraft, Correlator
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore
from aethergraph.services.integration import (
    IntegrationIngressCoordinator,
    IntegrationSessionResolution,
    InteractionResolutionError,
    InteractionResolver,
    ManifestRouteResolver,
    ResourceIngress,
    ResourceIngressError,
    ResourceIngressPolicy,
    SemanticEventEmitter,
    VerifiedAttachment,
    VerifiedIntegrationContext,
    install_integration_ingress,
)
import aethergraph.services.integration.dispatch as dispatch_module
from aethergraph.services.integration.dispatch import AGRootTurnDispatcher
from aethergraph.storage.contracts import (
    SessionKind,
    SessionRecord,
    StorageNotFoundError,
    StorageScope,
)
from tests._canonical_storage_fakes import (
    make_integration_persistence,
    make_semantic_event_store,
)
from tests._integration_fixtures import contract_compatibility

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


@pytest.mark.asyncio
async def test_root_dispatch_adopts_resources_before_external_admission(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class _Registry:
        def get_meta(self, **_kwargs):
            return {
                "backing": {"type": "graphfn", "name": "graph.support"},
                "run_visibility": "inline",
                "run_importance": "ephemeral",
            }

    class _Artifacts:
        async def attach_existing(self, artifact_id, **kwargs):
            events.append(("attached", (artifact_id, kwargs)))

    class _ArtifactFactory:
        def for_execution(self, scope, **_kwargs):
            events.append(("scope", scope))
            return _Artifacts()

    class _RunManager:
        async def submit_run(self, **kwargs):
            record = SimpleNamespace(run_id="run-root-1", started_at=_NOW)
            await kwargs["admission_callback"](record)
            events.append(("execution_scheduled", record.run_id))
            return record

    class _Monitor:
        def observe(self, **kwargs):
            events.append(("observed", kwargs["run_id"]))

    monkeypatch.setattr(dispatch_module, "scoped_registry", lambda _identity: _Registry())
    dispatcher = AGRootTurnDispatcher(
        SimpleNamespace(
            run_manager=_RunManager(),
            artifact_factory=_ArtifactFactory(),
        ),
        turn_monitor=_Monitor(),
    )

    async def persist_admission(run_id: str) -> None:
        events.append(("external_admission", run_id))

    run_id = await dispatcher.start(
        verified=_verified(),
        route=_route(),
        binding=_binding(),
        envelope=_envelope(),
        resources=(
            InputResource(
                kind="file",
                source="slack",
                id="target_buffer",
                artifact_id="artifact-input",
            ),
        ),
        admission_callback=persist_admission,
    )

    assert run_id == "run-root-1"
    assert [event[0] for event in events] == [
        "scope",
        "attached",
        "external_admission",
        "execution_scheduled",
        "observed",
    ]
    scope = events[0][1]
    assert isinstance(scope, StorageScope)
    assert scope.run_id == "run-root-1"
    assert scope.session_id == "session-1"
    assert scope.graph_id == "graph.support"
    assert scope.agent_id == "agent.support"


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
        release_compatibility=contract_compatibility(),
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


def _session() -> SessionRecord:
    return SessionRecord(
        session_id="session-1",
        kind=SessionKind.CHAT,
        scope=StorageScope(
            tenant_id="team-T1",
            project_id="project-1",
            session_id="session-1",
        ),
        revision=1,
        created_at=_NOW,
        updated_at=_NOW,
    )


class _SessionStore:
    async def provision(self, **kwargs) -> IntegrationSessionResolution:
        return IntegrationSessionResolution(
            binding=_binding(),
            session=_session(),
            session_created=False,
            binding_created=False,
        )


class _RootDispatcher:
    def __init__(self) -> None:
        self.calls = []

    async def start(self, **kwargs) -> str:
        self.calls.append(kwargs)
        callback = kwargs.get("admission_callback")
        if callback is not None:
            await callback("run-root-1")
        return "run-root-1"


class _FailingRootDispatcher:
    def __init__(self) -> None:
        self.calls = []

    async def start(self, **kwargs) -> str:
        self.calls.append(kwargs)
        raise ValueError("Agent not found: agent.support")


class _ResumeRouter:
    def __init__(self) -> None:
        self.calls = []

    async def resume_continuation(self, continuation, payload) -> None:
        self.calls.append({"continuation": continuation, "payload": payload})


async def _create_wait(
    store: InMemoryContinuationStore,
    *,
    run_id: str,
    node_id: str,
    kind: str,
    session_id: str,
    interaction_id: str,
    prompt=None,
):
    return (
        await store.create(
            ContinuationDraft(
                run_id=run_id,
                node_id=node_id,
                kind=kind,
                prompt=prompt,
                session_id=session_id,
                payload={"_interaction_id": interaction_id},
                correlators=(
                    Correlator(
                        scheme="interaction",
                        channel="public",
                        message=interaction_id,
                    ),
                ),
            )
        )
    ).record


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
    persistence = make_integration_persistence()
    return IntegrationIngressCoordinator(
        manifest=manifest,
        route_resolver=ManifestRouteResolver(manifest),
        idempotency_store=persistence.idempotency,
        session_store=_SessionStore(),
        resource_ingress=ResourceIngress(container=SimpleNamespace()),
        interaction_resolver=InteractionResolver(continuation_store),
        inbound_events=persistence.inbound_events,
        semantic_emitter=SemanticEventEmitter(
            deployment_id=manifest.deployment_id,
            store=event_log,
            semantic_event_protocol_version=manifest.semantic_event_protocol_version,
        ),
        resume_router=resume_router,
        root_dispatcher=root_dispatcher,
    )


@pytest.mark.asyncio
async def test_host_installer_binds_one_manifest_coordinator(tmp_path) -> None:
    class _Channels:
        def __init__(self) -> None:
            self.adapters = {"slack": SimpleNamespace(capabilities=set(), send=None)}

        def register_adapter(self, prefix, adapter) -> None:
            self.adapters[prefix] = adapter

    persistence = make_integration_persistence()
    event_log = persistence.semantic_events
    container = SimpleNamespace(
        root=str(tmp_path),
        integration_ingress=None,
        host_manifest=None,
        cont_store=InMemoryContinuationStore(secret=b"test-secret"),
        storage_services=SimpleNamespace(integration=persistence),
        resume_router=_ResumeRouter(),
        run_manager=SimpleNamespace(),
        channels=_Channels(),
    )
    manifest = _manifest(_route())

    coordinator = install_integration_ingress(container=container, manifest=manifest)

    assert container.integration_ingress is coordinator
    assert container.host_manifest is manifest
    assert coordinator.manifest is manifest
    assert container.semantic_events is not None
    assert container.semantic_turn_monitor is not None
    assert "endpoint" in container.channels.adapters
    assert not (tmp_path / "integration").exists()
    with pytest.raises(RuntimeError, match="already installed"):
        install_integration_ingress(container=container, manifest=manifest)
    await event_log.close()


@pytest.mark.asyncio
async def test_host_installer_enables_canonical_semantic_projector(tmp_path) -> None:
    class _Channels:
        def __init__(self) -> None:
            self.adapters = {"slack": SimpleNamespace(capabilities=set(), send=None)}

        def register_adapter(self, prefix, adapter) -> None:
            self.adapters[prefix] = adapter

    persistence = make_integration_persistence()
    event_log = persistence.semantic_events
    container = SimpleNamespace(
        root=str(tmp_path),
        integration_ingress=None,
        host_manifest=None,
        cont_store=InMemoryContinuationStore(secret=b"test-secret"),
        storage_services=SimpleNamespace(integration=persistence),
        resume_router=_ResumeRouter(),
        run_manager=SimpleNamespace(),
        channels=_Channels(),
    )
    manifest = _manifest(_route())

    coordinator = install_integration_ingress(container=container, manifest=manifest)

    assert coordinator.semantic_emitter.semantic_event_protocol_version == (
        SEMANTIC_EVENT_PROTOCOL_VERSION
    )
    assert container.integration_ingress is coordinator
    assert container.host_manifest is manifest
    assert not (tmp_path / "integration").exists()
    await event_log.close()


@pytest.mark.asyncio
async def test_coordinator_starts_one_root_turn_and_replays_receipt(tmp_path) -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    event_log = make_semantic_event_store()
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

    admitted: list[str] = []

    async def persist_admission(run_id: str) -> None:
        admitted.append(run_id)

    receipt = await coordinator.accept(
        verified=_verified(),
        envelope=_envelope(),
        root_admission_callback=persist_admission,
    )
    duplicate = await coordinator.accept(verified=_verified(), envelope=_envelope())

    assert receipt.action == "root_turn_started"
    assert receipt.turn_id == "run-root-1"
    assert receipt.event_cursor == 1
    assert duplicate == receipt.model_copy(update={"duplicate": True})
    assert len(root.calls) == 1
    assert admitted == ["run-root-1"]
    assert resume.calls == []
    semantic = await coordinator.semantic_events.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert len(semantic) == 1
    assert semantic[0].event.kind is SemanticEventKind.INPUT_ACCEPTED
    assert semantic[0].event.payload.input_id == "event-1"
    await event_log.close()


@pytest.mark.asyncio
async def test_coordinator_completes_idempotency_after_unexpected_dispatch_failure(
    tmp_path,
) -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    event_log = make_semantic_event_store()
    root = _FailingRootDispatcher()
    coordinator = _coordinator(
        tmp_path=tmp_path,
        route=_route(),
        continuation_store=continuation_store,
        event_log=event_log,
        root_dispatcher=root,
        resume_router=_ResumeRouter(),
    )

    with pytest.raises(ValueError, match="Agent not found"):
        await coordinator.accept(verified=_verified(), envelope=_envelope())
    duplicate = await coordinator.accept(verified=_verified(), envelope=_envelope())

    assert duplicate.accepted is False
    assert duplicate.duplicate is True
    assert duplicate.action == "rejected"
    assert duplicate.rejection_code == "integration.dispatch_failed"
    assert duplicate.event_cursor == 1
    assert len(root.calls) == 1
    semantic = await coordinator.semantic_events.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert [item.event.kind for item in semantic] == [SemanticEventKind.INPUT_ACCEPTED]
    await event_log.close()


@pytest.mark.asyncio
async def test_coordinator_resumes_exact_public_interaction_id(tmp_path) -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    continuation = await _create_wait(
        continuation_store,
        run_id="run-waiting-1",
        node_id="node-choice",
        kind="choice",
        prompt={"title": "Proceed?", "options": ["Yes", "No"]},
        session_id="session-1",
        interaction_id="interaction-public-1",
    )
    event_log = make_semantic_event_store()
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
    assert resume.calls[0]["continuation"].continuation_id == continuation.continuation_id
    assert resume.calls[0]["payload"]["interaction_id"] == "interaction-public-1"
    assert resume.calls[0]["payload"]["choice"] == "Yes"
    await event_log.close()


@pytest.mark.asyncio
async def test_interaction_resolver_rejects_ambiguous_bound_session_text() -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    for index in range(2):
        await _create_wait(
            continuation_store,
            run_id=f"run-{index}",
            node_id=f"node-{index}",
            kind="user_input",
            session_id="session-1",
            interaction_id=f"interaction-{index}",
        )

    with pytest.raises(InteractionResolutionError) as exc_info:
        await InteractionResolver(continuation_store).resolve(
            binding=_binding(),
            envelope=_envelope(),
        )
    assert exc_info.value.code == "integration.interaction_ambiguous"


@pytest.mark.asyncio
async def test_interaction_resolver_resolves_exact_public_identity() -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    continuation = await _create_wait(
        continuation_store,
        run_id="run-1",
        node_id="node-1",
        kind="user_input",
        session_id="session-1",
        interaction_id="interaction-public-1",
    )

    resolved = await InteractionResolver(continuation_store).resolve_exact(
        session_id="session-1",
        interaction_id="interaction-public-1",
        expected_kinds={"user_input"},
    )

    assert resolved.interaction_id == "interaction-public-1"
    assert resolved.continuation.continuation_id == continuation.continuation_id


@pytest.mark.asyncio
async def test_interaction_resolver_rejects_exact_identity_from_other_session() -> None:
    continuation_store = InMemoryContinuationStore(secret=b"test-secret")
    await _create_wait(
        continuation_store,
        run_id="run-1",
        node_id="node-1",
        kind="choice",
        session_id="session-1",
        interaction_id="interaction-public-1",
    )

    with pytest.raises(InteractionResolutionError) as exc_info:
        await InteractionResolver(continuation_store).resolve_exact(
            session_id="session-2",
            interaction_id="interaction-public-1",
            expected_kinds={"approval", "choice"},
        )

    assert exc_info.value.code == "integration.interaction_session_mismatch"


@pytest.mark.asyncio
async def test_coordinator_persists_disabled_route_rejection(tmp_path) -> None:
    event_log = make_semantic_event_store()
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
async def test_resource_ingress_validates_existing_artifact_reference(
    monkeypatch,
) -> None:
    artifact = Artifact(
        artifact_id="artifact-1",
        kind="upload",
        mime="application/pdf",
        bytes=100,
        uri="artifact://artifact-1",
        labels={"filename": "report.pdf"},
    )

    class _ArtifactService:
        async def get_by_id(self, artifact_id):
            return artifact if artifact_id == "artifact-1" else None

    class _UnexpectedStager:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("existing artifact references must not be copied")

    monkeypatch.setattr(
        "aethergraph.services.integration.resources.ResourceStager",
        _UnexpectedStager,
    )
    ingress = ResourceIngress(
        container=SimpleNamespace(artifact_service=_ArtifactService()),
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
        session_scope=_session().scope,
        envelope=envelope,
    )

    assert len(resources) == 1
    assert resources[0].artifact_id == "artifact-1"


@pytest.mark.asyncio
async def test_resource_ingress_materializes_verified_provider_bytes_once(
    monkeypatch,
) -> None:
    payload = b"exact current buffer\n"
    staged: list[bytes] = []

    class _Stager:
        def __init__(self, **_kwargs) -> None:
            pass

        async def stage_bytes(self, data, **_kwargs):
            staged.append(bytes(data))
            return InputResource(
                kind="upload",
                source="studio",
                status="materialized",
                id="buffer-1",
                name="weather.py",
                mime="text/x-python",
                size=len(data),
                artifact_id="artifact-buffer",
            )

    monkeypatch.setattr(
        "aethergraph.services.integration.resources.ResourceStager",
        _Stager,
    )
    ingress = ResourceIngress(container=SimpleNamespace())
    envelope = _envelope(
        text=None,
        attachments=(
            IngressAttachment(
                attachment_id="buffer-1",
                source_kind="provider_file",
                source_id="buffer-1",
                filename="weather.py",
                content_type="text/x-python",
                size_bytes=len(payload),
            ),
        ),
    )
    verified = VerifiedIntegrationContext(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id="team-T1",
        attachments=(VerifiedAttachment("buffer-1", payload),),
    )

    resources = await ingress.materialize(
        verified=verified,
        route=_route(),
        binding=_binding(),
        session_scope=_session().scope,
        envelope=envelope,
    )

    assert staged == [payload]
    assert resources[0].artifact_id == "artifact-buffer"


@pytest.mark.asyncio
async def test_resource_ingress_materializes_provider_bytes_with_canonical_container(
    tmp_path,
) -> None:
    payload = b"exact current buffer\n"
    base_logger = logging.getLogger("aethergraph")
    previous_logging = (
        list(base_logger.handlers),
        base_logger.level,
        base_logger.propagate,
        base_logger.disabled,
    )
    container = build_default_container(
        root=str(tmp_path),
        cfg=AppSettings(workspace=str(tmp_path)),
        channel_adapters={},
        owner_scope=StorageScope(project_id="project-1"),
    )
    await container.start_storage()
    try:
        await container.session_store.create(
            session_id="session-1",
            kind=SessionKind.CHAT,
            source="integration-test",
        )
        session_scope = await container.session_store.storage_scope("session-1")
        assert session_scope is not None
        ingress = ResourceIngress(container=container)
        resources = await ingress.materialize(
            verified=VerifiedIntegrationContext(
                integration_id="slack-main",
                integration_kind=IntegrationKind.SLACK,
                external_tenant_id="team-T1",
                attachments=(VerifiedAttachment("buffer-1", payload),),
            ),
            route=_route(),
            binding=_binding(),
            session_scope=session_scope,
            envelope=_envelope(
                text=None,
                attachments=(
                    IngressAttachment(
                        attachment_id="buffer-1",
                        source_kind="provider_file",
                        source_id="buffer-1",
                        filename="weather.py",
                        content_type="text/x-python",
                        size_bytes=len(payload),
                    ),
                ),
            ),
        )

        assert len(resources) == 1
        artifact_id = resources[0].artifact_id
        assert artifact_id is not None
        assert await container.artifact_service.get_by_id(artifact_id) is not None
        referenced = await ingress.materialize(
            verified=_verified(),
            route=_route(),
            binding=_binding(),
            session_scope=session_scope,
            envelope=_envelope(
                event_id="event-existing-artifact",
                text=None,
                attachments=(
                    IngressAttachment(
                        attachment_id="existing-buffer",
                        source_kind="artifact",
                        source_id=artifact_id,
                        filename="weather.py",
                        content_type="text/x-python",
                        size_bytes=len(payload),
                    ),
                ),
            ),
        )
        assert referenced[0].artifact_id == artifact_id
    finally:
        await container.close_storage()
        handlers, level, propagate, disabled = previous_logging
        base_logger.handlers[:] = handlers
        base_logger.setLevel(level)
        base_logger.propagate = propagate
        base_logger.disabled = disabled


@pytest.mark.asyncio
async def test_resource_ingress_reports_unavailable_canonical_artifact_service(
    caplog,
) -> None:
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

    logger = logging.getLogger("aethergraph.integration.resources")
    logger.addHandler(caplog.handler)
    try:
        with (
            caplog.at_level(logging.ERROR, logger=logger.name),
            pytest.raises(ResourceIngressError) as exc_info,
        ):
            await ingress.materialize(
                verified=_verified(),
                route=_route(),
                binding=_binding(),
                session_scope=_session().scope,
                envelope=envelope,
            )
    finally:
        logger.removeHandler(caplog.handler)

    assert exc_info.value.code == "integration.artifact_service_unavailable"
    record = next(
        record
        for record in caplog.records
        if record.integration_error_code == "integration.artifact_service_unavailable"
    )
    assert record.route_id == "route-slack"
    assert record.attachment_id == "attachment-1"
    assert record.artifact_id == "artifact-1"


@pytest.mark.asyncio
async def test_resource_ingress_surfaces_storage_scope_rejection(
    caplog,
    monkeypatch,
) -> None:
    payload = b"exact current buffer\n"

    class _RejectingStager:
        def __init__(self, **_kwargs) -> None:
            pass

        async def stage_bytes(self, _data, **_kwargs):
            raise StorageNotFoundError("session-1")

    monkeypatch.setattr(
        "aethergraph.services.integration.resources.ResourceStager",
        _RejectingStager,
    )
    ingress = ResourceIngress(container=SimpleNamespace())
    envelope = _envelope(
        text=None,
        attachments=(
            IngressAttachment(
                attachment_id="buffer-1",
                source_kind="provider_file",
                source_id="buffer-1",
                filename="weather.py",
                content_type="text/x-python",
                size_bytes=len(payload),
            ),
        ),
    )
    verified = VerifiedIntegrationContext(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id="team-T1",
        attachments=(VerifiedAttachment("buffer-1", payload),),
    )

    logger = logging.getLogger("aethergraph.integration.resources")
    logger.addHandler(caplog.handler)
    try:
        with (
            caplog.at_level(logging.ERROR, logger=logger.name),
            pytest.raises(ResourceIngressError) as exc_info,
        ):
            await ingress.materialize(
                verified=verified,
                route=_route(),
                binding=_binding(),
                session_scope=_session().scope,
                envelope=envelope,
            )
    finally:
        logger.removeHandler(caplog.handler)

    assert exc_info.value.code == "integration.attachment_storage_scope_rejected"
    record = next(
        record for record in caplog.records if record.name == "aethergraph.integration.resources"
    )
    assert record.integration_error_code == "integration.attachment_storage_scope_rejected"
    assert record.route_id == "route-slack"
    assert record.attachment_id == "buffer-1"
    assert record.storage_error_type == "StorageNotFoundError"


@pytest.mark.asyncio
async def test_resource_ingress_rejects_mismatched_provider_bytes() -> None:
    ingress = ResourceIngress(container=SimpleNamespace())
    envelope = _envelope(
        text=None,
        attachments=(
            IngressAttachment(
                attachment_id="buffer-1",
                source_kind="provider_file",
                source_id="buffer-1",
                filename="weather.py",
                content_type="text/x-python",
                size_bytes=4,
            ),
        ),
    )
    verified = VerifiedIntegrationContext(
        integration_id="slack-main",
        integration_kind=IntegrationKind.SLACK,
        external_tenant_id="team-T1",
        attachments=(VerifiedAttachment("other-buffer", b"data"),),
    )

    with pytest.raises(ResourceIngressError) as exc_info:
        await ingress.materialize(
            verified=verified,
            route=_route(),
            binding=_binding(),
            session_scope=_session().scope,
            envelope=envelope,
        )

    assert exc_info.value.code == "integration.attachment_bytes_missing"


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
            session_scope=_session().scope,
            envelope=envelope,
        )
    assert exc_info.value.code == "integration.attachments_unsupported"

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from aethergraph.contracts.integration import (
    ArtifactAvailablePayload,
    ExternalIdentity,
    HostManifest,
    IngressAttachment,
    IngressChoice,
    IngressEnvelope,
    IntegrationCapabilities,
    IntegrationKind,
    IntegrationMatchPolicy,
    IntegrationRoute,
    IntegrationSessionPolicy,
    InteractionOption,
    InteractionRequestedPayload,
    InteractionResolvedPayload,
    MessageCompletedPayload,
    MessageDeltaPayload,
    MessageStartedPayload,
    OriginAddress,
    PhaseChangedPayload,
    ProgressChangedPayload,
    SemanticEvent,
    SemanticEventKind,
    StructuredOutputPayload,
    ToolActivityPayload,
    TurnOutcomePayload,
    WarningRaisedPayload,
)
from aethergraph.services.channel.resources import InputResource
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.services.integration import (
    EventLogInboundEventStore,
    EventLogSemanticEventStore,
    IntegrationIngressCoordinator,
    InteractionResolver,
    ManifestRouteResolver,
    SemanticEventEmitter,
    SQLiteExternalSessionBindingStore,
    SQLiteIngressIdempotencyStore,
    VerifiedIntegrationContext,
)
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from tests._integration_fixtures import contract_compatibility

_NOW = datetime(2026, 8, 3, tzinfo=UTC)
_DIGEST = "d" * 64
_ALL_EVENTS = tuple(SemanticEventKind)


@dataclass(frozen=True, slots=True)
class _TransportCase:
    name: str
    kind: IntegrationKind
    integration_id: str
    route_id: str
    endpoint_id: str | None
    origin_prefix: str
    cancellation: bool


_CASES = (
    _TransportCase(
        name="ag_ui",
        kind=IntegrationKind.AG_UI,
        integration_id="ag-ui",
        route_id="route-ag-ui",
        endpoint_id="studio-ui",
        origin_prefix="endpoint:studio-ui",
        cancellation=True,
    ),
    _TransportCase(
        name="slack",
        kind=IntegrationKind.SLACK,
        integration_id="slack-main",
        route_id="route-slack",
        endpoint_id=None,
        origin_prefix="slack:team/T1",
        cancellation=False,
    ),
    _TransportCase(
        name="telegram",
        kind=IntegrationKind.TELEGRAM,
        integration_id="telegram-main",
        route_id="route-telegram",
        endpoint_id=None,
        origin_prefix="tg:chat",
        cancellation=False,
    ),
    _TransportCase(
        name="generic_endpoint",
        kind=IntegrationKind.AGENT_ENDPOINT,
        integration_id="endpoint-main",
        route_id="route-endpoint",
        endpoint_id="bespoke-ui",
        origin_prefix="endpoint:bespoke-ui",
        cancellation=True,
    ),
)


def _route(case: _TransportCase) -> IntegrationRoute:
    return IntegrationRoute(
        route_id=case.route_id,
        endpoint_id=case.endpoint_id,
        integration_id=case.integration_id,
        integration_kind=case.kind,
        entry_agent_id="agent.vertical_fixture",
        enabled=True,
        match_policy=IntegrationMatchPolicy(),
        session_policy=IntegrationSessionPolicy(scope="conversation_user"),
        required_capabilities=IntegrationCapabilities(
            event_kinds=_ALL_EVENTS,
            streaming=True,
            interactions=True,
            attachments=True,
            cancellation=case.cancellation,
        ),
    )


def _manifest() -> HostManifest:
    return HostManifest(
        deployment_id="deployment-vertical-v1",
        build_id="build-vertical-v1",
        source_digest=_DIGEST,
        build_root="C:/compiled/build-vertical-v1",
        entrypoint_module="compiled.vertical_fixture",
        entrypoint_symbol="register",
        graph_id="vertical_fixture.entry",
        entry_agent_id="agent.vertical_fixture",
        environment_snapshot_digest=_DIGEST,
        runtime_profile_digest=_DIGEST,
        application_settings_digest=_DIGEST,
        release_compatibility=contract_compatibility(),
        integration_routes=tuple(_route(case) for case in _CASES),
        workspace_identity="workspace-vertical-v1",
        manifest_digest=_DIGEST,
    )


def _identity(case: _TransportCase, conversation: str = "conversation-1") -> ExternalIdentity:
    return ExternalIdentity(
        tenant_id=f"tenant-{case.name}",
        conversation_id=f"{case.name}-{conversation}",
        thread_id=f"{conversation}-thread",
        user_id=f"{case.name}-user",
    )


def _verified(case: _TransportCase) -> VerifiedIntegrationContext:
    return VerifiedIntegrationContext(
        integration_id=case.integration_id,
        integration_kind=case.kind,
        external_tenant_id=f"tenant-{case.name}",
    )


def _envelope(
    case: _TransportCase,
    *,
    event_id: str,
    conversation: str = "conversation-1",
    text: str | None = "hello",
    choice: IngressChoice | None = None,
    attachments: tuple[IngressAttachment, ...] = (),
) -> IngressEnvelope:
    return IngressEnvelope(
        integration_id=case.integration_id,
        route_hint=case.route_id if case.endpoint_id is None else None,
        endpoint_id=case.endpoint_id,
        external_identity=_identity(case, conversation),
        external_event_id=event_id,
        idempotency_key=event_id,
        received_at=_NOW + timedelta(seconds=sum(ord(char) for char in event_id)),
        text=text,
        choice=choice,
        attachments=attachments,
        origin_address=OriginAddress(
            channel_key=f"{case.origin_prefix}:{conversation}",
            capability_profile_id=f"{case.name}-v1",
        ),
    )


class _ResourceIngress:
    async def materialize(self, *, verified, route, binding, envelope):
        assert verified.integration_kind is route.integration_kind
        return tuple(
            InputResource(
                kind="artifact",
                source=route.integration_kind.value,
                status="materialized",
                id=item.attachment_id,
                name=item.filename,
                mime=item.content_type,
                size=item.size_bytes,
                artifact_id=item.source_id,
            )
            for item in envelope.attachments
        )


class _RootDispatcher:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return f"run-{len(self.calls)}"


class _ResumeRouter:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def resume(self, **kwargs) -> None:
        self.calls.append(kwargs)


@dataclass(slots=True)
class _Harness:
    root: Path
    manifest: HostManifest
    continuations: FSContinuationStore
    dispatcher: _RootDispatcher
    resumes: _ResumeRouter
    event_log: SqliteEventLog
    coordinator: IntegrationIngressCoordinator

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        dispatcher: _RootDispatcher | None = None,
        resumes: _ResumeRouter | None = None,
    ) -> _Harness:
        manifest = _manifest()
        continuations = FSContinuationStore(root / "continuations", secret=b"vertical-secret")
        event_log = SqliteEventLog(str(root / "events.db"))
        dispatcher = dispatcher or _RootDispatcher()
        resumes = resumes or _ResumeRouter()
        coordinator = IntegrationIngressCoordinator(
            manifest=manifest,
            route_resolver=ManifestRouteResolver(manifest),
            idempotency_store=SQLiteIngressIdempotencyStore(root / "operations.db"),
            binding_store=SQLiteExternalSessionBindingStore(root / "operations.db"),
            resource_ingress=_ResourceIngress(),
            interaction_resolver=InteractionResolver(continuations),
            inbound_events=EventLogInboundEventStore(event_log),
            semantic_emitter=SemanticEventEmitter(
                deployment_id=manifest.deployment_id,
                store=EventLogSemanticEventStore(event_log),
                semantic_event_protocol_version=manifest.semantic_event_protocol_version,
            ),
            resume_router=resumes,
            root_dispatcher=dispatcher,
        )
        return cls(
            root=root,
            manifest=manifest,
            continuations=continuations,
            dispatcher=dispatcher,
            resumes=resumes,
            event_log=event_log,
            coordinator=coordinator,
        )

    async def restart(self) -> _Harness:
        await self.event_log.close()
        return self.open(
            self.root,
            dispatcher=self.dispatcher,
            resumes=self.resumes,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
async def test_each_transport_shares_root_session_duplicate_attachment_and_restart(
    tmp_path: Path,
    case: _TransportCase,
) -> None:
    harness = _Harness.open(tmp_path / case.name)
    try:
        first_envelope = _envelope(case, event_id=f"{case.name}-root")
        first = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=first_envelope,
        )
        second = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(case, event_id=f"{case.name}-second", text="follow up"),
        )
        duplicate = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=first_envelope,
        )
        attachment = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(
                case,
                event_id=f"{case.name}-attachment",
                text=None,
                attachments=(
                    IngressAttachment(
                        attachment_id="attachment-1",
                        source_kind="artifact",
                        source_id="artifact-1",
                        filename="brief.txt",
                        content_type="text/plain",
                        size_bytes=5,
                    ),
                ),
            ),
        )

        assert first.session_id == second.session_id == attachment.session_id
        assert duplicate == first.model_copy(update={"duplicate": True})
        assert len(harness.dispatcher.calls) == 3
        assert harness.dispatcher.calls[0]["route"].entry_agent_id == "agent.vertical_fixture"
        assert harness.dispatcher.calls[0]["envelope"].origin_address.channel_key.startswith(
            case.origin_prefix
        )
        assert harness.dispatcher.calls[2]["resources"][0].artifact_id == "artifact-1"

        restarted = await harness.restart()
        harness = restarted
        replayed = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=first_envelope,
        )
        after_restart = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(case, event_id=f"{case.name}-restart", text="after restart"),
        )
        assert replayed == first.model_copy(update={"duplicate": True})
        assert after_restart.session_id == first.session_id
        assert len(harness.dispatcher.calls) == 4

        history = await harness.coordinator.semantic_events.list_session(
            deployment_id=harness.manifest.deployment_id,
            session_id=first.session_id,
        )
        assert [record.event.kind for record in history] == [
            SemanticEventKind.INPUT_ACCEPTED,
            SemanticEventKind.INPUT_ACCEPTED,
            SemanticEventKind.INPUT_ACCEPTED,
            SemanticEventKind.INPUT_ACCEPTED,
        ]
        replay = await harness.coordinator.semantic_events.list_session(
            deployment_id=harness.manifest.deployment_id,
            session_id=first.session_id,
            after_cursor=history[1].cursor,
        )
        assert replay == history[2:]
    finally:
        await harness.event_log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
async def test_each_transport_resumes_choice_and_free_text_without_cross_delivery(
    tmp_path: Path,
    case: _TransportCase,
) -> None:
    harness = _Harness.open(tmp_path / case.name)
    try:
        root = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(case, event_id=f"{case.name}-bind"),
        )
        await harness.continuations.save(
            Continuation(
                run_id="run-choice",
                node_id="node-choice",
                kind="choice",
                token=f"{case.name}-choice-token",
                prompt={"title": "Ship?", "options": ["approve", "reject"]},
                session_id=root.session_id,
                payload={"_interaction_id": f"{case.name}-choice"},
            )
        )
        choice = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(
                case,
                event_id=f"{case.name}-choice-event",
                text=None,
                choice=IngressChoice(
                    interaction_id=f"{case.name}-choice",
                    option_ids=("approve",),
                ),
            ),
        )
        await harness.continuations.mark_closed(f"{case.name}-choice-token")
        await harness.continuations.save(
            Continuation(
                run_id="run-text",
                node_id="node-text",
                kind="user_input",
                token=f"{case.name}-text-token",
                session_id=root.session_id,
                payload={"_interaction_id": f"{case.name}-text"},
            )
        )
        free_text = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(case, event_id=f"{case.name}-text-event", text="revise it"),
        )

        assert choice.action == free_text.action == "continuation_resumed"
        assert choice.session_id == free_text.session_id == root.session_id
        assert harness.resumes.calls[0]["token"] == f"{case.name}-choice-token"
        assert harness.resumes.calls[0]["payload"]["choice"] == "approve"
        assert harness.resumes.calls[1]["token"] == f"{case.name}-text-token"
        assert harness.resumes.calls[1]["payload"]["text"] == "revise it"

        other = await harness.coordinator.accept(
            verified=_verified(case),
            envelope=_envelope(
                case,
                event_id=f"{case.name}-other-session",
                conversation="conversation-2",
            ),
        )
        assert other.session_id != root.session_id
        assert harness.dispatcher.calls[-1]["binding"].ag_session_id == other.session_id
    finally:
        await harness.event_log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
async def test_each_transport_keeps_concurrent_sessions_and_capabilities_exact(
    tmp_path: Path,
    case: _TransportCase,
) -> None:
    harness = _Harness.open(tmp_path / case.name)
    try:
        receipts = await asyncio.gather(
            *(
                harness.coordinator.accept(
                    verified=_verified(case),
                    envelope=_envelope(
                        case,
                        event_id=f"{case.name}-concurrent-{index}",
                        conversation=f"conversation-{index}",
                    ),
                )
                for index in range(6)
            )
        )
        assert len({receipt.session_id for receipt in receipts}) == 6
        route = next(
            route
            for route in harness.manifest.integration_routes
            if route.route_id == case.route_id
        )
        assert route.required_capabilities.event_kinds == _ALL_EVENTS
        assert route.required_capabilities.streaming is True
        assert route.required_capabilities.interactions is True
        assert route.required_capabilities.attachments is True
        assert route.required_capabilities.cancellation is case.cancellation
    finally:
        await harness.event_log.close()


@pytest.mark.asyncio
async def test_shared_fixture_semantics_cover_streaming_activity_interactions_and_terminals(
    tmp_path: Path,
) -> None:
    harness = _Harness.open(tmp_path)
    try:
        sessions: dict[str, str] = {}
        for case in _CASES:
            receipt = await harness.coordinator.accept(
                verified=_verified(case),
                envelope=_envelope(case, event_id=f"{case.name}-semantic-root"),
            )
            sessions[case.name] = receipt.session_id
            payloads = (
                (
                    SemanticEventKind.MESSAGE_STARTED,
                    MessageStartedPayload(message_id="message-stream"),
                ),
                (
                    SemanticEventKind.MESSAGE_DELTA,
                    MessageDeltaPayload(message_id="message-stream", delta="hel"),
                ),
                (
                    SemanticEventKind.MESSAGE_DELTA,
                    MessageDeltaPayload(message_id="message-stream", delta="lo"),
                ),
                (
                    SemanticEventKind.MESSAGE_COMPLETED,
                    MessageCompletedPayload(message_id="message-stream", text="hello"),
                ),
                (
                    SemanticEventKind.PHASE_CHANGED,
                    PhaseChangedPayload(phase="research", status="active", label="Researching"),
                ),
                (
                    SemanticEventKind.PROGRESS_CHANGED,
                    ProgressChangedPayload(
                        progress_id="progress-1",
                        status="running",
                        label="Reading",
                        current=1,
                        total=2,
                    ),
                ),
                (
                    SemanticEventKind.TOOL_ACTIVITY,
                    ToolActivityPayload(
                        tool_call_id="tool-1", tool_name="lookup", status="completed"
                    ),
                ),
                (
                    SemanticEventKind.INTERACTION_REQUESTED,
                    InteractionRequestedPayload(
                        interaction_id="approval-1",
                        request_kind="approval",
                        prompt="Ship?",
                        options=(
                            InteractionOption(option_id="approve", label="Approve"),
                            InteractionOption(option_id="reject", label="Reject"),
                        ),
                    ),
                ),
                (
                    SemanticEventKind.INTERACTION_RESOLVED,
                    InteractionResolvedPayload(
                        interaction_id="approval-1",
                        resolution_kind="approved",
                        option_ids=("approve",),
                    ),
                ),
                (
                    SemanticEventKind.ARTIFACT_AVAILABLE,
                    ArtifactAvailablePayload(
                        artifact_id="artifact-1",
                        filename="report.txt",
                        content_type="text/plain",
                        size_bytes=5,
                    ),
                ),
                (
                    SemanticEventKind.STRUCTURED_OUTPUT,
                    StructuredOutputPayload(
                        output_name="fixture.result", value={"status": "ready"}
                    ),
                ),
                (
                    SemanticEventKind.WARNING_RAISED,
                    WarningRaisedPayload(
                        code="fixture.warning", message="Deliberate non-terminal warning."
                    ),
                ),
                (
                    SemanticEventKind.TURN_OUTCOME,
                    TurnOutcomePayload(
                        outcome="completed",
                        code="completed",
                        summary="Fixture completed.",
                        resumable=False,
                        engine_turn_id=f"{case.name}-engine-success",
                    ),
                ),
            )
            records = []
            for sequence, (kind, payload) in enumerate(payloads):
                records.append(
                    await harness.coordinator.semantic_events.append(
                        SemanticEvent(
                            event_id=f"{case.name}-success-{sequence}",
                            deployment_id=harness.manifest.deployment_id,
                            session_id=receipt.session_id,
                            turn_id=f"{case.name}-success",
                            sequence=sequence,
                            producer="fixture.vertical",
                            timestamp=_NOW + timedelta(milliseconds=sequence),
                            kind=kind,
                            payload=payload,
                        )
                    )
                )
            normal = await harness.coordinator.semantic_events.append(
                SemanticEvent(
                    event_id=f"{case.name}-normal",
                    deployment_id=harness.manifest.deployment_id,
                    session_id=receipt.session_id,
                    turn_id=f"{case.name}-normal",
                    sequence=0,
                    producer="fixture.vertical",
                    timestamp=_NOW + timedelta(seconds=1),
                    kind=SemanticEventKind.MESSAGE_COMPLETED,
                    payload=MessageCompletedPayload(
                        message_id="message-normal", text="normal reply"
                    ),
                )
            )
            failed = await harness.coordinator.semantic_events.append(
                SemanticEvent(
                    event_id=f"{case.name}-failed",
                    deployment_id=harness.manifest.deployment_id,
                    session_id=receipt.session_id,
                    turn_id=f"{case.name}-failed",
                    sequence=0,
                    producer="fixture.vertical",
                    timestamp=_NOW + timedelta(seconds=2),
                    kind=SemanticEventKind.TURN_OUTCOME,
                    payload=TurnOutcomePayload(
                        outcome="failed",
                        code="fixture.failure",
                        summary="Deliberate terminal failure.",
                        resumable=False,
                        engine_turn_id=f"{case.name}-engine-failed",
                    ),
                )
            )
            replay = await harness.coordinator.semantic_events.list_session(
                deployment_id=harness.manifest.deployment_id,
                session_id=receipt.session_id,
                after_cursor=records[-1].cursor,
            )
            assert [item.cursor for item in replay] == [normal.cursor, failed.cursor]
            assert [item.event.kind for item in replay] == [
                SemanticEventKind.MESSAGE_COMPLETED,
                SemanticEventKind.TURN_OUTCOME,
            ]

        assert len(set(sessions.values())) == len(_CASES)
        assert {call["route"].entry_agent_id for call in harness.dispatcher.calls} == {
            harness.manifest.entry_agent_id
        }
        assert {call["route"].route_id for call in harness.dispatcher.calls} == {
            case.route_id for case in _CASES
        }
    finally:
        await harness.event_log.close()

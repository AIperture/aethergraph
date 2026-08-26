from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.contracts.integration import (
    SEMANTIC_EVENT_PROTOCOL_VERSION,
    ArtifactAvailablePayload,
    InteractionRequestedPayload,
    MessageCompletedPayload,
    SemanticEvent,
    SemanticEventKind,
    StructuredOutputPayload,
    ToolActivityPayload,
    TurnOutcomePayload,
    WarningRaisedPayload,
)
from aethergraph.contracts.services.channel import Button, ChannelAction, OutEvent
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.integration import (
    SemanticDeliveryError,
    SemanticEventChannelAdapter,
    SemanticEventEmitter,
    SemanticTurnMonitor,
)
from tests._canonical_storage_fakes import make_semantic_event_store


def _meta() -> dict[str, str]:
    return {
        "run_id": "run-1",
        "session_id": "session-1",
        "agent_id": "agent.support",
    }


@pytest.mark.asyncio
async def test_semantic_adapter_persists_ordered_message_and_interaction(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    first = await adapter.send(
        OutEvent(
            type="agent.message",
            channel="endpoint:sessions/public-1",
            text="Hello",
            meta=_meta(),
        )
    )
    second = await adapter.send(
        OutEvent(
            type="session.need_approval",
            channel="endpoint:sessions/public-1",
            text="Ship?",
            buttons=[Button(label="Ship", value="ship")],
            meta={
                **_meta(),
                "interaction_id": "interaction-public-1",
                "interaction_kind": "choice",
            },
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert first == {"event_cursor": 1, "event_cursors": [1]}
    assert second == {"event_cursor": 2, "event_cursors": [2]}
    assert [item.event.sequence for item in history] == [0, 1]
    assert [item.event.kind for item in history] == [
        SemanticEventKind.MESSAGE_COMPLETED,
        SemanticEventKind.INTERACTION_REQUESTED,
    ]
    assert isinstance(history[0].event.payload, MessageCompletedPayload)
    assert isinstance(history[1].event.payload, InteractionRequestedPayload)
    assert history[1].event.payload.interaction_id == "interaction-public-1"
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_rejects_missing_turn_identity(tmp_path) -> None:
    event_log = make_semantic_event_store()
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=event_log,
        )
    )

    with pytest.raises(SemanticDeliveryError, match="session_id"):
        await adapter.send(
            OutEvent(
                type="agent.message",
                channel="endpoint:sessions/public-1",
                text="Hello",
                meta={"run_id": "run-1"},
            )
        )
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_rejects_unsupported_channel_event(tmp_path) -> None:
    event_log = make_semantic_event_store()
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=event_log,
        )
    )

    with pytest.raises(SemanticDeliveryError, match="Unsupported Channel event type"):
        await adapter.send(
            OutEvent(
                type="not.a.semantic.event",  # type: ignore[arg-type]
                channel="endpoint:sessions/public-1",
                meta=_meta(),
            )
        )
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_persists_named_structured_output(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="structured.output",
            channel="endpoint:sessions/public-1",
            rich={
                "output_name": "workflow.status",
                "value": {"operation": "clear"},
            },
            meta=_meta(),
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert history[0].event.kind == SemanticEventKind.STRUCTURED_OUTPUT
    assert isinstance(history[0].event.payload, StructuredOutputPayload)
    assert history[0].event.payload.output_name == "workflow.status"
    assert history[0].event.payload.value == {"operation": "clear"}
    await event_log.close()


@pytest.mark.asyncio
async def test_channel_bus_interaction_reaches_semantic_delivery(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )
    bus = ChannelBus(adapters={"endpoint": adapter})

    await bus.notify(
        Continuation(
            continuation_id="cont-1",
            revision=1,
            run_id="run-1",
            node_id="node-1",
            kind="choice",
            channel="endpoint:sessions/public-1",
            prompt={"title": "Proceed?", "choices": [{"id": "yes", "label": "Yes"}]},
            session_id="session-1",
            payload={"_interaction_id": "interaction-public-1"},
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert history[0].event.turn_id == "run-1"
    assert history[0].event.producer == "node-1"
    assert history[0].event.kind == SemanticEventKind.INTERACTION_REQUESTED
    assert history[0].event.payload.interaction_id == "interaction-public-1"
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_preserves_authored_buttons(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="agent.message",
            channel="endpoint:sessions/public-1",
            text="Choose",
            actions=[
                ChannelAction(
                    kind="external_link",
                    label="Open report",
                    href="https://example.test/report",
                    style="primary",
                )
            ],
            meta=_meta(),
        )
    )

    payload = (
        await store.list_session(
            deployment_id="deployment-1",
            session_id="session-1",
        )
    )[0].event.payload
    assert isinstance(payload, MessageCompletedPayload)
    assert payload.text == "Choose"
    assert len(payload.actions) == 1
    assert payload.actions[0].kind == "external_link"
    assert payload.actions[0].href == "https://example.test/report"
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_rejects_url_only_assistant_file(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    with pytest.raises(SemanticDeliveryError, match="canonical artifacts"):
        await adapter.send(
            OutEvent(
                type="agent.message",
                channel="endpoint:sessions/public-1",
                text="Report",
                file={"filename": "report.pdf", "url": "https://example.test/report.pdf"},
                meta=_meta(),
            )
        )
    assert await store.list_session(deployment_id="deployment-1", session_id="session-1") == ()
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_announces_artifact_before_referencing_message(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    result = await adapter.send(
        OutEvent(
            type="agent.message",
            channel="endpoint:sessions/public-1",
            text="Rendered image",
            file={
                "artifact_id": "artifact-1",
                "filename": "probe.png",
                "mimetype": "image/png",
                "size": 67,
            },
            meta=_meta(),
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert result == {"event_cursor": 2, "event_cursors": [1, 2]}
    assert [item.event.kind for item in history] == [
        SemanticEventKind.ARTIFACT_AVAILABLE,
        SemanticEventKind.MESSAGE_COMPLETED,
    ]
    available = history[0].event.payload
    assert isinstance(available, ArtifactAvailablePayload)
    assert available.artifact_id == "artifact-1"
    assert available.filename == "probe.png"
    assert available.content_type == "image/png"
    assert available.size_bytes == 67
    message = history[1].event.payload
    assert isinstance(message, MessageCompletedPayload)
    assert tuple(item.artifact_id for item in message.attachments) == ("artifact-1",)
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_preserves_text_or_files_interaction_kind(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="session.need_input",
            channel="endpoint:sessions/public-1",
            text="Reply or attach evidence",
            meta={
                **_meta(),
                "interaction_id": "interaction-mixed-1",
                "interaction_kind": "user_input_or_files",
                "accept": ["text/plain", "image/png"],
            },
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    requested = history[0].event.payload
    assert isinstance(requested, InteractionRequestedPayload)
    assert requested.request_kind == "text_or_files"
    assert requested.accepted_content_types == ("text/plain", "image/png")
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_preserves_structured_output_upsert_identity(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="structured.output",
            channel="endpoint:sessions/public-1",
            rich={
                "output_name": "agstudio.workbench.suggestion",
                "value": {"suggestion_id": "sug-1", "revision": 2},
            },
            meta=_meta(),
            upsert_key="suggestion:sug-1",
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert history[0].event.extensions["aethergraph.upsert_key"] == ("suggestion:sug-1")
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_projects_tool_activity_with_upsert_identity(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="agent.tool.activity",
            channel="endpoint:sessions/public-1",
            text="Project inspected.",
            rich={
                "tool_call_id": "call-1",
                "tool_name": "inspect_project",
                "status": "completed",
                "message": "Project inspected.",
            },
            meta=_meta(),
            upsert_key="tool:call-1",
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    event = history[0].event
    assert event.kind == SemanticEventKind.TOOL_ACTIVITY
    assert isinstance(event.payload, ToolActivityPayload)
    assert event.payload.tool_call_id == "call-1"
    assert event.payload.status == "completed"
    assert event.extensions["aethergraph.upsert_key"] == "tool:call-1"
    await event_log.close()


@pytest.mark.asyncio
async def test_tool_failure_preserves_structured_error_in_one_activity_path(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=store,
            semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_VERSION,
        )
    )

    await adapter.send(
        OutEvent(
            type="agent.tool.activity",
            channel="endpoint:sessions/public-1",
            text="Apply the design before retrying.",
            rich={
                "tool_call_id": "call-1",
                "tool_name": "run_design",
                "status": "failed",
                "message": "Apply the design before retrying.",
                "error": {
                    "kind": "rejected",
                    "code": "design_not_applied",
                    "summary": "Apply the design before retrying.",
                    "retryable": True,
                    "details": {"design_id": "design-1"},
                    "repair_hints": ["Apply the current design."],
                    "allowed_actions": ["apply_design"],
                    "reference": "tool-error-1",
                },
            },
            meta=_meta(),
            upsert_key="tool:call-1",
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    event = history[0].event
    assert isinstance(event, SemanticEvent)
    assert event.kind == SemanticEventKind.TOOL_ACTIVITY
    assert isinstance(event.payload, ToolActivityPayload)
    assert event.payload.error is not None
    assert event.payload.error.code == "design_not_applied"
    assert event.payload.error.allowed_actions == ("apply_design",)
    await event_log.close()


@pytest.mark.asyncio
async def test_turn_monitor_appends_terminal_event_after_channel_history(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    emitter = SemanticEventEmitter(deployment_id="deployment-1", store=store)
    await emitter.emit(
        OutEvent(
            type="agent.message",
            channel="endpoint:sessions/public-1",
            text="Done",
            meta=_meta(),
        )
    )

    class _RunManager:
        async def wait_run(self, run_id, *, return_outputs=False):
            assert run_id == "run-1"
            assert return_outputs is True
            return (
                SimpleNamespace(status=RunStatus.succeeded),
                {
                    "agent_outcome": {
                        "outcome": "completed",
                        "code": "completed",
                        "summary": "Execution completed.",
                        "resumable": False,
                        "engine_turn_id": "engine-turn-1",
                        "reply_disposition": "message_required",
                    }
                },
            )

    monitor = SemanticTurnMonitor(run_manager=_RunManager(), emitter=emitter)
    await monitor._observe(
        run_id="run-1",
        session_id="session-1",
        route_id="studio-assistant",
        integration_id="agstudio",
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert [item.event.sequence for item in history] == [0, 1]
    assert history[1].event.kind == SemanticEventKind.TURN_OUTCOME
    assert isinstance(history[1].event.payload, TurnOutcomePayload)
    assert history[1].event.payload.outcome == "completed"
    assert history[1].event.payload.reply_disposition == "message_required"
    assert history[1].event.extensions == {
        "aethergraph.integration_id": "agstudio",
        "aethergraph.route_id": "studio-assistant",
    }
    await event_log.close()


@pytest.mark.asyncio
async def test_turn_monitor_uses_engine_outcome_after_infrastructure_success(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    emitter = SemanticEventEmitter(
        deployment_id="deployment-1",
        store=store,
        semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_VERSION,
    )
    await emitter.emit_semantic(
        session_id="session-1",
        turn_id="run-1",
        producer="aethergraph.engine",
        kind=SemanticEventKind.WARNING_RAISED,
        payload=WarningRaisedPayload(
            code="partial_context",
            message="One optional source was unavailable.",
        ),
    )

    class _RunManager:
        async def wait_run(self, run_id, *, return_outputs=False):
            assert run_id == "run-1"
            assert return_outputs is True
            return (
                SimpleNamespace(status=RunStatus.succeeded),
                {
                    "agent_outcome": {
                        "outcome": "failed",
                        "code": "composition_error",
                        "summary": "The response could not be composed.",
                        "resumable": False,
                        "engine_turn_id": "engine-turn-7",
                        "reply_disposition": "no_message",
                        "runtime_error": True,
                        "diagnostics": {"phase": "composition"},
                    }
                },
            )

    monitor = SemanticTurnMonitor(run_manager=_RunManager(), emitter=emitter)
    await monitor._observe(
        run_id="run-1",
        session_id="session-1",
        route_id="studio-assistant",
        integration_id="agstudio",
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert [item.event.kind for item in history] == [
        SemanticEventKind.WARNING_RAISED,
        SemanticEventKind.TURN_OUTCOME,
    ]
    payload = history[1].event.payload
    assert isinstance(payload, TurnOutcomePayload)
    assert payload.outcome == "failed"
    assert payload.engine_turn_id == "engine-turn-7"
    assert payload.reply_disposition == "no_message"
    assert "runtime_error" not in payload.model_dump()
    await event_log.close()


@pytest.mark.asyncio
async def test_new_turn_outcome_emission_requires_explicit_reply_disposition() -> None:
    event_log = make_semantic_event_store()
    emitter = SemanticEventEmitter(
        deployment_id="deployment-1",
        store=event_log,
        semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_VERSION,
    )
    try:
        with pytest.raises(
            SemanticDeliveryError,
            match="explicit reply_disposition",
        ):
            await emitter.emit_semantic(
                session_id="session-1",
                turn_id="run-historical-shape",
                producer="aethergraph.engine",
                kind=SemanticEventKind.TURN_OUTCOME,
                payload=TurnOutcomePayload(
                    outcome="completed",
                    code="completed",
                    summary="Missing current delivery contract.",
                    resumable=False,
                    engine_turn_id="engine-turn-1",
                ),
            )
    finally:
        await event_log.close()


@pytest.mark.asyncio
async def test_turn_monitor_does_not_invent_outcome_for_infrastructure_failure(
    tmp_path,
) -> None:
    event_log = make_semantic_event_store()
    store = event_log

    class _RunManager:
        async def wait_run(self, run_id, *, return_outputs=False):
            assert return_outputs is True
            return SimpleNamespace(status=RunStatus.failed), None

    monitor = SemanticTurnMonitor(
        run_manager=_RunManager(),
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=store,
            semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_VERSION,
        ),
    )
    await monitor._observe(
        run_id="run-1",
        session_id="session-1",
        route_id="studio-assistant",
        integration_id="agstudio",
    )

    assert (
        await store.list_session(
            deployment_id="deployment-1",
            session_id="session-1",
        )
        == ()
    )
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_preserves_rich_message_as_named_output(tmp_path) -> None:
    event_log = make_semantic_event_store()
    store = event_log
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )
    rich = {
        "kind": "component",
        "payload": {"component_type": "ag.ui.run_card.v1", "props": {"run_id": "run-1"}},
    }

    await adapter.send(
        OutEvent(
            type="agent.message",
            channel="endpoint:sessions/public-1",
            text="Live run",
            rich=rich,
            meta=_meta(),
        )
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    payload = history[0].event.payload
    assert history[0].event.kind == SemanticEventKind.STRUCTURED_OUTPUT
    assert isinstance(payload, StructuredOutputPayload)
    assert payload.output_name == "channel.rich"
    assert payload.value == {"text": "Live run", "rich": rich}
    await event_log.close()

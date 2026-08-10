from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.contracts.integration import (
    SEMANTIC_EVENT_PROTOCOL_V2,
    InteractionRequestedPayload,
    MessageCompletedPayload,
    SemanticEventKind,
    SemanticEventKindV2,
    SemanticEventV2,
    StructuredOutputPayload,
    ToolActivityPayload,
    ToolActivityPayloadV2,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnOutcomePayload,
    WarningRaisedPayload,
)
from aethergraph.contracts.services.channel import Button, OutEvent
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.services.channel.channel_bus import ChannelBus
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.integration import (
    EventLogSemanticEventStore,
    SemanticDeliveryError,
    SemanticEventChannelAdapter,
    SemanticEventEmitter,
    SemanticTurnMonitor,
)
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog


def _meta() -> dict[str, str]:
    return {
        "run_id": "run-1",
        "session_id": "session-1",
        "agent_id": "agent.support",
    }


@pytest.mark.asyncio
async def test_semantic_adapter_persists_ordered_message_and_interaction(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
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
    assert first == {"event_cursor": 1}
    assert second == {"event_cursor": 2}
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=EventLogSemanticEventStore(event_log),
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=EventLogSemanticEventStore(event_log),
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )
    bus = ChannelBus(adapters={"endpoint": adapter})

    await bus.notify(
        Continuation(
            run_id="run-1",
            node_id="node-1",
            token="private-token",
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="link.buttons",
            channel="endpoint:sessions/public-1",
            text="Choose",
            buttons=[
                Button(
                    label="Open report",
                    value="report",
                    url="https://example.test/report",
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
    assert isinstance(payload, StructuredOutputPayload)
    assert payload.output_name == "channel.link.buttons"
    assert payload.value == {
        "text": "Choose",
        "buttons": [
            {
                "label": "Open report",
                "value": "report",
                "url": "https://example.test/report",
                "style": "primary",
            }
        ],
        "file": None,
    }
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_projects_remote_file_as_structured_output(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store)
    )

    await adapter.send(
        OutEvent(
            type="agent.message",
            channel="endpoint:sessions/public-1",
            text="Report",
            file={"filename": "report.pdf", "url": "https://example.test/report.pdf"},
            meta=_meta(),
        )
    )

    payload = (
        await store.list_session(
            deployment_id="deployment-1",
            session_id="session-1",
        )
    )[0].event.payload
    assert isinstance(payload, StructuredOutputPayload)
    assert payload.output_name == "channel.attachment"
    assert payload.value == {
        "text": "Report",
        "file": {
            "filename": "report.pdf",
            "url": "https://example.test/report.pdf",
        },
    }
    await event_log.close()


@pytest.mark.asyncio
async def test_semantic_adapter_preserves_structured_output_upsert_identity(
    tmp_path,
) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
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
async def test_v2_tool_failure_preserves_structured_error_in_one_activity_path(
    tmp_path,
) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    adapter = SemanticEventChannelAdapter(
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=store,
            semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_V2,
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
    assert isinstance(event, SemanticEventV2)
    assert event.kind == SemanticEventKindV2.TOOL_ACTIVITY
    assert isinstance(event.payload, ToolActivityPayloadV2)
    assert event.payload.error is not None
    assert event.payload.error.code == "design_not_applied"
    assert event.payload.error.allowed_actions == ("apply_design",)
    await event_log.close()


@pytest.mark.asyncio
async def test_v1_tool_failure_remains_closed_when_channel_carries_v2_error(
    tmp_path,
) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    emitter = SemanticEventEmitter(deployment_id="deployment-1", store=store)

    await emitter.emit(
        OutEvent(
            type="agent.tool.activity",
            channel="endpoint:sessions/public-1",
            rich={
                "tool_call_id": "call-1",
                "tool_name": "run_design",
                "status": "failed",
                "message": "Failed.",
                "error": {
                    "kind": "internal",
                    "code": "tool_internal_error",
                    "summary": "The Tool failed internally.",
                },
            },
            meta=_meta(),
        )
    )

    event = (
        await store.list_session(
            deployment_id="deployment-1",
            session_id="session-1",
        )
    )[0].event
    assert isinstance(event.payload, ToolActivityPayload)
    assert type(event.payload) is ToolActivityPayload
    assert "error" not in event.payload.model_dump()
    await event_log.close()


@pytest.mark.asyncio
async def test_turn_monitor_appends_terminal_event_after_channel_history(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
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
        async def wait_run(self, run_id):
            assert run_id == "run-1"
            return SimpleNamespace(
                status=RunStatus.succeeded,
                result_available=True,
                error=None,
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
    assert history[1].event.kind == SemanticEventKind.TURN_COMPLETED
    assert isinstance(history[1].event.payload, TurnCompletedPayload)
    assert history[1].event.payload.result_available is True
    assert history[1].event.extensions == {
        "aethergraph.integration_id": "agstudio",
        "aethergraph.route_id": "studio-assistant",
    }
    await event_log.close()


@pytest.mark.asyncio
async def test_turn_monitor_projects_cancellation_as_terminal_failure(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)

    class _RunManager:
        async def wait_run(self, run_id):
            return SimpleNamespace(
                status=RunStatus.canceled,
                result_available=False,
                error="Run cancelled by user",
            )

    monitor = SemanticTurnMonitor(
        run_manager=_RunManager(),
        emitter=SemanticEventEmitter(deployment_id="deployment-1", store=store),
    )
    await monitor._observe(
        run_id="run-canceled",
        session_id="session-1",
        route_id="studio-assistant",
        integration_id="agstudio",
    )

    history = await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    event = history[0].event
    assert event.kind == SemanticEventKind.TURN_FAILED
    assert isinstance(event.payload, TurnFailedPayload)
    assert event.payload.code == "run_canceled"
    assert event.payload.message == "Run cancelled by user"
    assert event.payload.retryable is False
    await event_log.close()


@pytest.mark.asyncio
async def test_v2_turn_monitor_uses_engine_outcome_after_infrastructure_success(
    tmp_path,
) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    emitter = SemanticEventEmitter(
        deployment_id="deployment-1",
        store=store,
        semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_V2,
    )
    await emitter.emit_semantic(
        session_id="session-1",
        turn_id="run-1",
        producer="aethergraph.engine",
        kind=SemanticEventKindV2.WARNING_RAISED,
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
        SemanticEventKindV2.WARNING_RAISED,
        SemanticEventKindV2.TURN_OUTCOME,
    ]
    payload = history[1].event.payload
    assert isinstance(payload, TurnOutcomePayload)
    assert payload.outcome == "failed"
    assert payload.engine_turn_id == "engine-turn-7"
    assert "runtime_error" not in payload.model_dump()
    await event_log.close()


@pytest.mark.asyncio
async def test_v2_turn_monitor_does_not_invent_outcome_for_infrastructure_failure(
    tmp_path,
) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)

    class _RunManager:
        async def wait_run(self, run_id, *, return_outputs=False):
            assert return_outputs is True
            return SimpleNamespace(status=RunStatus.failed), None

    monitor = SemanticTurnMonitor(
        run_manager=_RunManager(),
        emitter=SemanticEventEmitter(
            deployment_id="deployment-1",
            store=store,
            semantic_event_protocol_version=SEMANTIC_EVENT_PROTOCOL_V2,
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
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
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

from __future__ import annotations

import pytest

from aethergraph.contracts.integration import (
    InteractionRequestedPayload,
    MessageCompletedPayload,
    SemanticEventKind,
    StructuredOutputPayload,
)
from aethergraph.contracts.services.channel import Button, OutEvent
from aethergraph.services.integration import (
    EventLogSemanticEventStore,
    SemanticDeliveryError,
    SemanticEventChannelAdapter,
    SemanticEventEmitter,
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

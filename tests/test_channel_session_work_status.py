from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession


class _FakeBus:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)


class _FakeContext:
    def __init__(self, *, origin_channel_key: str) -> None:
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.origin_binding = SimpleNamespace(channel_key=origin_channel_key)
        self.services = SimpleNamespace(
            channels=_FakeBus(),
            continuation_store=None,
            memory_facade=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "channel_key",
    ["endpoint:sessions/session-1", "slack:team/T:chan/C", "tg:chat/123"],
)
async def test_work_status_replace_emits_provider_neutral_structured_output(
    channel_key: str,
) -> None:
    ctx = _FakeContext(origin_channel_key=channel_key)
    payload = {
        "workflow_id": "wf-1",
        "title": "Workflow",
        "kind": "workflow",
        "status": "running",
        "updated_at": "",
        "items": [],
    }

    result = await ChannelSession(ctx).work_status().replace(payload)

    assert result == {"operation": "replace", "work_status": payload}
    event = ctx.services.channels.events[0]
    assert event.type == "structured.output"
    assert event.channel == channel_key
    assert event.rich == {"output_name": "workflow.status", "value": result}
    assert event.meta["session_id"] == "session-1"
    assert event.meta["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_work_status_patch_uses_explicit_workflow_id_override() -> None:
    ctx = _FakeContext(origin_channel_key="endpoint:sessions/session-1")
    handle = ChannelSession(ctx).work_status(workflow_id="wf-bound")

    result = await handle.patch(
        workflow_id="wf-explicit",
        status="running",
        summary="In progress",
        active_item_id="stage-1",
        item_updates=[{"id": "stage-1", "status": "running"}],
    )

    assert result == {
        "operation": "patch",
        "workflow_id": "wf-explicit",
        "status": "running",
        "summary": "In progress",
        "active_item_id": "stage-1",
        "item_updates": [{"id": "stage-1", "status": "running"}],
    }


@pytest.mark.asyncio
async def test_work_status_clear_emits_clear_operation() -> None:
    ctx = _FakeContext(origin_channel_key="endpoint:sessions/session-1")

    result = await ChannelSession(ctx).work_status().clear()

    assert result == {"operation": "clear"}
    assert ctx.services.channels.events[0].rich["value"] == result

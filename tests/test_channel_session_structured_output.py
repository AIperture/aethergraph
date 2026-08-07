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
    def __init__(self) -> None:
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.origin_binding = SimpleNamespace(channel_key="endpoint:studio:session/session-1")
        self.services = SimpleNamespace(
            channels=_FakeBus(),
            continuation_store=None,
            memory_facade=None,
        )


@pytest.mark.asyncio
async def test_send_structured_output_uses_origin_and_context_metadata() -> None:
    ctx = _FakeContext()

    await ChannelSession(ctx).send_structured_output(
        output_name="agstudio.workbench.suggestion",
        value={"suggestion_id": "sug-1", "revision": 1},
        upsert_key="suggestion:sug-1",
    )

    event = ctx.services.channels.events[0]
    assert event.type == "structured.output"
    assert event.channel == "endpoint:studio:session/session-1"
    assert event.rich == {
        "output_name": "agstudio.workbench.suggestion",
        "value": {"suggestion_id": "sug-1", "revision": 1},
    }
    assert event.upsert_key == "suggestion:sug-1"
    assert event.meta["run_id"] == "run-1"
    assert event.meta["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_send_structured_output_rejects_empty_name() -> None:
    ctx = _FakeContext()

    with pytest.raises(ValueError, match="non-empty output_name"):
        await ChannelSession(ctx).send_structured_output(
            output_name=" ",
            value={},
        )

    assert ctx.services.channels.events == []


@pytest.mark.asyncio
async def test_send_tool_activity_preserves_structured_identity() -> None:
    ctx = _FakeContext()

    await ChannelSession(ctx).send_tool_activity(
        tool_call_id="call-1",
        tool_name="inspect_project",
        status="completed",
        message="Project inspected.",
    )

    event = ctx.services.channels.events[0]
    assert event.type == "agent.tool.activity"
    assert event.upsert_key == "tool:call-1"
    assert event.rich == {
        "tool_call_id": "call-1",
        "tool_name": "inspect_project",
        "status": "completed",
        "message": "Project inspected.",
    }
    assert event.meta["run_id"] == "run-1"
    assert event.meta["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_send_tool_activity_carries_safe_failure_envelope() -> None:
    ctx = _FakeContext()
    error = {
        "kind": "rejected",
        "code": "stale_project",
        "summary": "Refresh the project before retrying.",
        "retryable": True,
    }

    await ChannelSession(ctx).send_tool_activity(
        tool_call_id="call-2",
        tool_name="apply_patch",
        status="failed",
        message="Refresh the project before retrying.",
        error=error,
    )

    event = ctx.services.channels.events[0]
    assert event.rich["error"] == error
    assert event.upsert_key == "tool:call-2"

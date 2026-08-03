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
async def test_dashboard_replace_emits_provider_neutral_structured_output(
    channel_key: str,
) -> None:
    ctx = _FakeContext(origin_channel_key=channel_key)
    state = {
        "dashboard_id": "dash-1",
        "dashboard_type": "generic.dashboard",
        "workflow_id": "wf-1",
        "revision": 1,
        "status": "running",
        "updated_at": "",
        "data": {},
    }

    result = await ChannelSession(ctx).dashboard_state(dashboard_id="dash-1").replace(state)

    assert result == {"operation": "replace", "dashboard": state}
    event = ctx.services.channels.events[0]
    assert event.type == "structured.output"
    assert event.channel == channel_key
    assert event.rich == {"output_name": "workflow.dashboard", "value": result}
    assert event.meta["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_dashboard_patch_and_clear_carry_bound_dashboard_id() -> None:
    ctx = _FakeContext(origin_channel_key="endpoint:sessions/session-1")
    handle = ChannelSession(ctx).dashboard_state(dashboard_id="dash-bound")

    patch = await handle.patch(
        revision=2,
        status="running",
        ops=[{"op": "replace", "path": "/status", "value": "running"}],
    )
    clear = await handle.clear()

    assert patch == {
        "operation": "patch",
        "patch": {
            "dashboard_id": "dash-bound",
            "revision": 2,
            "status": "running",
            "ops": [{"op": "replace", "path": "/status", "value": "running"}],
        },
    }
    assert clear == {"operation": "clear", "dashboard_id": "dash-bound"}


@pytest.mark.asyncio
async def test_dashboard_replace_rejects_mismatched_identity() -> None:
    ctx = _FakeContext(origin_channel_key="endpoint:sessions/session-1")

    with pytest.raises(ValueError, match="bound dashboard_id"):
        await (
            ChannelSession(ctx)
            .dashboard_state(dashboard_id="dash-1")
            .replace({"dashboard_id": "dash-2"})
        )

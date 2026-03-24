from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession


class _FakeBus:
    def __init__(self) -> None:
        self.published = []

    def get_default_channel_key(self) -> str:
        return "ui:session/test-session"

    def resolve_channel_key(self, key: str) -> str:
        return key

    async def publish(self, event):
        self.published.append(event)


class _FakeMemoryFacade:
    def __init__(self) -> None:
        self.calls = []

    async def record_chat(self, role, text, *, tags=None, data=None, severity=2, signal=None):
        self.calls.append(
            {
                "role": role,
                "text": text,
                "tags": tags,
                "data": data,
                "severity": severity,
                "signal": signal,
            }
        )


class _FakeContext:
    def __init__(self) -> None:
        self.run_id = "run-ctx"
        self.node_id = "node-ctx"
        self.session_id = "session-ctx"
        self.graph_id = "graph-ctx"
        self.agent_id = "agent-ctx"
        self.app_id = "app-ctx"
        self.services = SimpleNamespace(
            channels=_FakeBus(),
            continuation_store=None,
            memory_facade=_FakeMemoryFacade(),
        )


@pytest.mark.asyncio
async def test_send_run_card_emits_rich_component_payload() -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)

    await chan.send_run_card(
        "run-123",
        graph_id="graph-123",
        title="Training run",
        subtitle="live monitor",
        session_id="session-123",
        show_preview=False,
        poll_ms=2500,
        meta={"origin": "test"},
    )

    assert len(ctx.services.channels.published) == 1
    event = ctx.services.channels.published[0]
    assert event.type == "agent.message"
    assert event.text == "This is a live run: run-123"
    assert event.rich["kind"] == "component"
    assert event.rich["payload"]["component_type"] == "ag.ui.run_card.v1"
    assert event.rich["payload"]["props"] == {
        "version": "run_card.v1",
        "run_id": "run-123",
        "graph_id": "graph-123",
        "title": "Training run",
        "subtitle": "live monitor",
        "session_id": "session-123",
        "view": {
            "show_preview": False,
            "show_actions": True,
            "poll_ms": 2500,
        },
        "fallback": {"text": "This is a live run: run-123"},
    }
    assert event.meta["origin"] == "test"
    assert event.meta["run_id"] == "run-ctx"
    assert event.meta["graph_id"] == "graph-ctx"
    assert event.meta["session_id"] == "session-ctx"

    assert len(ctx.services.memory_facade.calls) == 1
    assert ctx.services.memory_facade.calls[0]["text"] == "This is a live run: run-123"


@pytest.mark.asyncio
async def test_send_run_card_respects_memory_log_false() -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)

    await chan.send_run_card("run-456", memory_log=False)

    assert len(ctx.services.channels.published) == 1
    assert ctx.services.memory_facade.calls == []

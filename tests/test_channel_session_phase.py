from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession


class _FakeBus:
    def __init__(self) -> None:
        self.published = []

    async def publish(self, event):
        self.published.append(event)


class _FakeContext:
    def __init__(self) -> None:
        self.run_id = "run-phase"
        self.node_id = "node-phase"
        self.session_id = "session-phase"
        self.graph_id = "graph-phase"
        self.agent_id = "agent-phase"
        self.app_id = "app-phase"
        self.origin_binding = SimpleNamespace(channel_key="endpoint:sessions/test-session")
        self.services = SimpleNamespace(
            channels=_FakeBus(),
            continuation_store=None,
            memory_facade=None,
        )


class _ThinkingLlm:
    async def chat_stream(self, **kwargs):
        await kwargs["on_thinking_delta"]("alpha ")
        await kwargs["on_thinking_delta"]("beta")
        await kwargs["on_delta"]("done")
        return "done", {"input_tokens": 1, "output_tokens": 1}


@pytest.mark.asyncio
async def test_send_phase_emits_explicit_phase_key_payload() -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)

    await chan.send_phase(
        phase="tool",
        status="active",
        phase_key="tool-call-1",
        label="Calling tool",
    )

    assert len(ctx.services.channels.published) == 1
    event = ctx.services.channels.published[0]
    assert event.type == "agent.progress.update"
    assert event.rich["kind"] == "phase"
    assert event.rich["phase_key"] == "tool-call-1"
    assert event.rich["phase_key_source"] == "explicit"
    assert event.rich["phase_event_id"]
    assert event.meta["phase_key"] == "tool-call-1"
    assert event.meta["phase_key_source"] == "explicit"
    assert event.meta["run_id"] == "run-phase"


@pytest.mark.asyncio
async def test_send_phase_omitted_phase_key_uses_event_fallback() -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)

    await chan.send_phase(phase="thinking", status="active")

    event = ctx.services.channels.published[0]
    assert event.rich["phase_key"] == event.rich["phase_event_id"]
    assert event.rich["phase_key_source"] == "event"
    assert event.meta["phase_key"] == event.rich["phase_event_id"]
    assert event.meta["phase_key_source"] == "event"


@pytest.mark.asyncio
async def test_chat_and_stream_thinking_phase_reuses_explicit_phase_key() -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)

    text, usage, thinking = await chan.chat_and_stream(
        llm=_ThinkingLlm(),
        messages=[{"role": "user", "content": "hello"}],
        emit_thinking_phase=True,
        thinking_detail_interval_s=0,
    )

    phase_events = [
        event
        for event in ctx.services.channels.published
        if event.rich and event.rich.get("kind") == "phase"
    ]
    phase_keys = {event.rich["phase_key"] for event in phase_events}

    assert text == "done"
    assert usage == {"input_tokens": 1, "output_tokens": 1}
    assert thinking == "alpha beta"
    assert len(phase_events) >= 3
    assert len(phase_keys) == 1
    assert {event.rich["phase_key_source"] for event in phase_events} == {"explicit"}
    assert [event.rich["status"] for event in phase_events][-1] == "done"

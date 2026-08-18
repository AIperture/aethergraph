from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession


class _Bus:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event) -> None:
        self.events.append(event)


class _Context:
    def __init__(self) -> None:
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.origin_binding = SimpleNamespace(channel_key="ui:test")
        self.services = SimpleNamespace(
            channels=_Bus(),
            continuation_store=None,
            memory_facade=None,
        )


@pytest.mark.asyncio
async def test_send_text_preserves_explicit_idempotency_key() -> None:
    context = _Context()
    session = ChannelSession(context)

    await session.send_text(
        "Working on it.",
        upsert_key="assistant-output:turn-1:0",
        memory_log=False,
    )

    assert len(context.services.channels.events) == 1
    assert context.services.channels.events[0].upsert_key == "assistant-output:turn-1:0"

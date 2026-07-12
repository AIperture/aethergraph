from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession
from aethergraph.services.runner.facade import RunFacade


class _SharedBus:
    def get_default_channel_key(self) -> str:
        return "console:stdin"

    def resolve_channel_key(self, key: str) -> str:
        return key


def _context(channel_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        run_id=f"run-{channel_key.rsplit('/', 1)[-1]}",
        node_id="node-1",
        session_id=channel_key.rsplit("/", 1)[-1],
        graph_id="graph-1",
        agent_id="agent-1",
        app_id=None,
        default_channel_key=channel_key,
        services=SimpleNamespace(
            channels=_SharedBus(),
            continuation_store=None,
            memory_facade=None,
        ),
    )


@pytest.mark.asyncio
async def test_concurrent_contexts_resolve_distinct_run_scoped_channels() -> None:
    first = ChannelSession(_context("ui:session/first"))
    second = ChannelSession(_context("ui:session/second"))

    first_key, second_key = await asyncio.gather(
        asyncio.to_thread(first._resolve_key),
        asyncio.to_thread(second._resolve_key),
    )

    assert first_key == "ui:session/first"
    assert second_key == "ui:session/second"


def test_explicit_channel_overrides_run_scoped_default() -> None:
    channel = ChannelSession(_context("ui:session/default"))

    assert channel._resolve_key("ui:session/explicit") == "ui:session/explicit"


class _CapturingRunManager:
    def __init__(self) -> None:
        self.submitted: dict = {}
        self.waited: dict = {}

    async def submit_run(self, **kwargs):
        self.submitted = kwargs
        return SimpleNamespace(run_id="child-spawned")

    async def run_and_wait(self, graph_id, **kwargs):
        self.waited = {"graph_id": graph_id, **kwargs}
        return SimpleNamespace(run_id="child-waited"), {"ok": True}, False, []


@pytest.mark.asyncio
async def test_runner_facade_propagates_default_channel_to_child_runs() -> None:
    manager = _CapturingRunManager()
    facade = RunFacade(
        run_manager=manager,
        session_id="session-parent",
        default_channel_key="ui:session/session-parent",
    )

    await facade.spawn_run("child-graph", inputs={"mode": "spawn"})
    await facade.run_and_wait("child-graph", inputs={"mode": "wait"})

    expected = {"default_channel_key": "ui:session/session-parent"}
    assert manager.submitted["run_config"] == expected
    assert manager.waited["run_config"] == expected

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pydantic import ValidationError
import pytest

from aethergraph.contracts.integration import OriginBinding
from aethergraph.contracts.services.channel import ChannelRoutingError
from aethergraph.core.runtime import graph_runner
from aethergraph.services.channel.session import ChannelSession
from aethergraph.services.runner.facade import RunFacade


class _SharedBus:
    pass


def _context(channel_key: str) -> SimpleNamespace:
    session_id = channel_key.rsplit("/", 1)[-1]
    return SimpleNamespace(
        run_id=f"run-{session_id}",
        node_id="node-1",
        session_id=session_id,
        graph_id="graph-1",
        agent_id="agent-1",
        app_id=None,
        origin_binding=OriginBinding(
            integration_id="integration.ui",
            route_id="route.ui",
            session_id=session_id,
            channel_key=channel_key,
            external_conversation_id=session_id,
            capability_profile_id="ag-ui/v1",
        ),
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


def test_missing_origin_fails_without_console_or_bus_default() -> None:
    context = _context("ui:session/default")
    context.origin_binding = None

    with pytest.raises(ChannelRoutingError) as exc_info:
        ChannelSession(context)._resolve_key()

    assert exc_info.value.code == "channel.origin_required"
    assert exc_info.value.channel_key is None


@pytest.mark.asyncio
async def test_graph_runner_deserializes_closed_origin_binding(monkeypatch) -> None:
    binding = _context("ui:session/session-1").origin_binding
    monkeypatch.setattr(graph_runner, "_get_container", lambda: SimpleNamespace())

    env, _, _ = await graph_runner._build_env(
        SimpleNamespace(graph_id="graph-1", spec={}),
        {},
        origin_binding=binding.model_dump(mode="json"),
    )

    assert env.origin_binding == binding

    invalid = binding.model_dump(mode="json")
    invalid["provider_payload"] = {"not": "part of the contract"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        await graph_runner._build_env(
            SimpleNamespace(graph_id="graph-1", spec={}),
            {},
            origin_binding=invalid,
        )


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
async def test_runner_facade_propagates_origin_binding_to_child_runs() -> None:
    manager = _CapturingRunManager()
    origin_binding = OriginBinding(
        integration_id="integration.ui",
        route_id="route.ui",
        session_id="session-parent",
        channel_key="ui:session/session-parent",
        external_conversation_id="session-parent",
        capability_profile_id="ag-ui/v1",
    )
    facade = RunFacade(
        run_manager=manager,
        session_id="session-parent",
        origin_binding=origin_binding,
    )

    await facade.spawn_run("child-graph", inputs={"mode": "spawn"})
    await facade.run_and_wait("child-graph", inputs={"mode": "wait"})

    expected = {"origin_binding": origin_binding.model_dump(mode="json")}
    assert manager.submitted["run_config"] == expected
    assert manager.waited["run_config"] == expected

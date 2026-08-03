from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.integration import OriginBinding
from aethergraph.plugins.channel.utils import turn_dispatch


class _Registry:
    def get_meta(self, *, nspace: str, name: str) -> dict:
        assert nspace == "agent"
        assert name == "assistant"
        return {
            "backing": {"type": "graphfn", "name": "assistant_graph"},
            "run_visibility": "inline",
            "run_importance": "ephemeral",
            "app_id": "assistant-app",
        }


class _RunManager:
    def __init__(self) -> None:
        self.submitted: dict = {}

    async def submit_run(self, **kwargs):
        self.submitted = kwargs
        return SimpleNamespace(run_id="run-1")


@pytest.mark.asyncio
async def test_dispatch_closes_exact_origin_into_run_config(monkeypatch) -> None:
    run_manager = _RunManager()
    container = SimpleNamespace(run_manager=run_manager)
    monkeypatch.setattr(turn_dispatch, "scoped_registry", lambda identity: _Registry())

    run_id = await turn_dispatch.dispatch_channel_turn_run(
        container=container,
        identity=RequestIdentity(user_id="user-1", org_id="org-1", mode="local"),
        agent_id="assistant",
        text="Hello",
        attachments=[],
        session_id="session-1",
        origin_channel_key="slack:team/T:chan/C",
        integration_id="slack",
        route_id="route-support",
        external_conversation_id="slack:C#thread:100.1",
        external_thread_id="100.1",
        capability_profile_id="slack-v1",
    )

    assert run_id == "run-1"
    assert run_manager.submitted["session_id"] == "session-1"
    binding = OriginBinding.model_validate(run_manager.submitted["run_config"]["origin_binding"])
    assert binding.channel_key == "slack:team/T:chan/C"
    assert binding.route_id == "route-support"
    assert binding.external_thread_id == "100.1"


@pytest.mark.asyncio
async def test_dispatch_allocates_one_session_for_unbound_conversation(monkeypatch) -> None:
    run_manager = _RunManager()
    container = SimpleNamespace(run_manager=run_manager)
    monkeypatch.setattr(turn_dispatch, "scoped_registry", lambda identity: _Registry())

    await turn_dispatch.dispatch_channel_turn_run(
        container=container,
        identity=None,
        agent_id="assistant",
        text="Hello",
        attachments=[],
        origin_channel_key="tg:chat/99",
        integration_id="telegram",
        route_id="route-support",
        external_conversation_id="tg:99",
        external_thread_id=None,
        capability_profile_id="telegram-v1",
    )

    session_id = run_manager.submitted["session_id"]
    binding = OriginBinding.model_validate(run_manager.submitted["run_config"]["origin_binding"])
    assert session_id.startswith("session-")
    assert binding.session_id == session_id
    assert binding.external_conversation_id == "tg:99"

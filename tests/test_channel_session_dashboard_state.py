from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession


class _FakeBus:
    def __init__(self, default_channel_key: str = "ui:session/test-session") -> None:
        self.default_channel_key = default_channel_key

    def get_default_channel_key(self) -> str:
        return self.default_channel_key

    def resolve_channel_key(self, key: str) -> str:
        if key == "ui:session":
            return "ui:session/test-session"
        if key == "ui:run":
            return "ui:run/test-run"
        return key


class _FakeContext:
    def __init__(self, *, default_channel_key: str = "ui:session/test-session") -> None:
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.services = SimpleNamespace(
            channels=_FakeBus(default_channel_key=default_channel_key),
            continuation_store=None,
            memory_facade=None,
        )


@pytest.mark.asyncio
async def test_dashboard_state_replace_uses_default_ui_session_channel(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)
    captured = {}

    async def fake_replace_session_dashboard_state(*, session_id, dashboard_state, meta):
        captured["session_id"] = session_id
        captured["dashboard_state"] = dashboard_state
        captured["meta"] = meta
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_dashboard_state.replace_session_dashboard_state",
        fake_replace_session_dashboard_state,
    )

    payload = {
        "dashboard_id": "dash-1",
        "dashboard_type": "generic.dashboard",
        "workflow_id": "wf-1",
        "revision": 1,
        "status": "running",
        "updated_at": "",
        "data": {},
    }
    out = await chan.dashboard_state(dashboard_id="dash-1").replace(payload)

    assert out == {"ok": True}
    assert captured["session_id"] == "test-session"
    assert captured["dashboard_state"] == payload
    assert captured["meta"]["channel_key"] == "ui:session/test-session"


@pytest.mark.asyncio
async def test_dashboard_state_patch_uses_bound_dashboard_id(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx, "ui:session")
    captured = {}

    async def fake_patch_session_dashboard_state(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_dashboard_state.patch_session_dashboard_state",
        fake_patch_session_dashboard_state,
    )

    out = await chan.dashboard_state(dashboard_id="dash-bound").patch(
        revision=2,
        status="running",
        ops=[{"op": "replace", "path": "/status", "value": "running"}],
    )

    assert out == {"ok": True}
    assert captured["session_id"] == "test-session"
    assert captured["dashboard_id"] == "dash-bound"
    assert captured["revision"] == 2
    assert captured["status"] == "running"


@pytest.mark.asyncio
async def test_dashboard_state_clear_uses_explicit_ui_session_channel(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx, "ui:session/custom-session")
    captured = {}

    async def fake_clear_session_dashboard_state(*, session_id, dashboard_id, meta=None):
        captured["session_id"] = session_id
        captured["dashboard_id"] = dashboard_id
        captured["meta"] = meta
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_dashboard_state.clear_session_dashboard_state",
        fake_clear_session_dashboard_state,
    )

    out = await chan.dashboard_state(dashboard_id="dash-clear").clear()

    assert out == {"ok": True}
    assert captured["session_id"] == "custom-session"
    assert captured["dashboard_id"] == "dash-clear"
    assert captured["meta"]["channel_key"] == "ui:session/custom-session"

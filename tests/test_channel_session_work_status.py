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
async def test_work_status_replace_uses_default_ui_session_channel(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx)
    captured = {}

    async def fake_replace_session_work_status(*, session_id, work_status, meta):
        captured["session_id"] = session_id
        captured["work_status"] = work_status
        captured["meta"] = meta
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_work_status.replace_session_work_status",
        fake_replace_session_work_status,
    )

    payload = {
        "workflow_id": "wf-1",
        "title": "Workflow",
        "kind": "workflow",
        "status": "running",
        "updated_at": "",
        "items": [],
    }
    out = await chan.work_status().replace(payload)

    assert out == {"ok": True}
    assert captured["session_id"] == "test-session"
    assert captured["work_status"] == payload
    assert captured["meta"]["channel_key"] == "ui:session/test-session"
    assert captured["meta"]["run_id"] == "run-1"
    assert captured["meta"]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_work_status_patch_uses_bound_workflow_id(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx, "ui:session")
    captured = {}

    async def fake_patch_session_work_status(
        *,
        session_id,
        workflow_id=None,
        status=None,
        summary=None,
        active_item_id=None,
        item_updates=None,
        meta=None,
    ):
        captured["session_id"] = session_id
        captured["workflow_id"] = workflow_id
        captured["status"] = status
        captured["summary"] = summary
        captured["active_item_id"] = active_item_id
        captured["item_updates"] = item_updates
        captured["meta"] = meta
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_work_status.patch_session_work_status",
        fake_patch_session_work_status,
    )

    out = await chan.work_status(workflow_id="wf-bound").patch(
        status="running",
        summary="In progress",
        active_item_id="stage-1",
        item_updates=[{"id": "stage-1", "status": "running"}],
    )

    assert out == {"ok": True}
    assert captured["session_id"] == "test-session"
    assert captured["workflow_id"] == "wf-bound"
    assert captured["status"] == "running"
    assert captured["summary"] == "In progress"
    assert captured["active_item_id"] == "stage-1"
    assert captured["item_updates"] == [{"id": "stage-1", "status": "running"}]
    assert captured["meta"]["channel_key"] == "ui:session/test-session"


@pytest.mark.asyncio
async def test_work_status_patch_allows_explicit_workflow_id_override(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx, "ui:session/test-session")
    captured = {}

    async def fake_patch_session_work_status(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_work_status.patch_session_work_status",
        fake_patch_session_work_status,
    )

    await chan.work_status(workflow_id="wf-bound").patch(workflow_id="wf-explicit")

    assert captured["session_id"] == "test-session"
    assert captured["workflow_id"] == "wf-explicit"


@pytest.mark.asyncio
async def test_work_status_clear_uses_explicit_ui_session_channel(monkeypatch) -> None:
    ctx = _FakeContext()
    chan = ChannelSession(ctx, "ui:session/custom-session")
    captured = {}

    async def fake_clear_session_work_status(*, session_id, meta=None):
        captured["session_id"] = session_id
        captured["meta"] = meta
        return {"ok": True}

    monkeypatch.setattr(
        "aethergraph.services.channel.session_work_status.clear_session_work_status",
        fake_clear_session_work_status,
    )

    out = await chan.work_status().clear()

    assert out == {"ok": True}
    assert captured["session_id"] == "custom-session"
    assert captured["meta"]["channel_key"] == "ui:session/custom-session"


@pytest.mark.parametrize(
    ("default_channel_key", "channel_key"),
    [
        ("ui:run/test-run", None),
        ("slack:team/T:chan/C", None),
        ("telegram:chat/123", None),
        ("console:stdin", None),
        ("ui:session/test-session", "ui:run"),
        ("ui:session/test-session", "slack:team/T:chan/C"),
        ("ui:session/test-session", "telegram:chat/123"),
        ("ui:session/test-session", "console:stdin"),
    ],
)
def test_work_status_rejects_non_ui_session_channels(
    default_channel_key: str,
    channel_key: str | None,
) -> None:
    ctx = _FakeContext(default_channel_key=default_channel_key)
    chan = ChannelSession(ctx, channel_key)

    with pytest.raises(RuntimeError, match="requires a ui:session channel"):
        chan.work_status()

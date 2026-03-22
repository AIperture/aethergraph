from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.services.channel.session import ChannelSession


class _FakeBus:
    def get_default_channel_key(self) -> str:
        return "ui:session/test-session"

    def resolve_channel_key(self, key: str) -> str:
        return key


class _FakeContext:
    def __init__(self, *, resume_payload=None):
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = None
        self.app_id = None
        self.resume_payload = resume_payload
        self.services = SimpleNamespace(
            channels=_FakeBus(),
            continuation_store=None,
            memory_facade=None,
        )

    async def create_continuation(self, **kwargs):
        raise AssertionError("create_continuation should not be called for replay resume payloads")

    def prepare_wait_for_resume(self, token: str):
        raise AssertionError(
            "prepare_wait_for_resume should not be called for replay resume payloads"
        )


@pytest.mark.asyncio
async def test_ask_text_consumes_matching_resume_payload_without_waiting():
    ctx = _FakeContext(
        resume_payload={
            "_channel_wait_kind": "user_input",
            "prompt": "What is your name?",
            "text": "Ada",
        }
    )
    chan = ChannelSession(ctx)

    reply = await chan.ask_text("What is your name?")

    assert reply == "Ada"
    assert getattr(ctx, "_channel_resume_payload_consumed", False) is True


@pytest.mark.asyncio
async def test_ask_text_ignores_non_matching_resume_payload():
    ctx = _FakeContext(
        resume_payload={
            "_channel_wait_kind": "approval",
            "prompt": {"title": "Approve?"},
            "choice": "approve",
        }
    )
    chan = ChannelSession(ctx)

    matched = chan._take_matching_resume_payload(
        kind="user_input",
        expected_payload={"prompt": "What is your name?", "_silent": False},
    )

    assert matched is None
    assert getattr(ctx, "_channel_resume_payload_consumed", False) is False


@pytest.mark.asyncio
async def test_ask_approval_infers_choice_from_text_when_resume_payload_has_no_choice():
    ctx = _FakeContext(
        resume_payload={
            "_channel_wait_kind": "approval",
            "prompt": {"title": "Approve?", "buttons": ["Approve", "Cancel"]},
            "text": "Approve",
        }
    )
    chan = ChannelSession(ctx)

    reply = await chan.ask_approval("Approve?", options=["Approve", "Cancel"])

    assert reply["approved"] is True
    assert reply["choice"] == "Approve"
    assert reply["text"] == "Approve"
    assert getattr(ctx, "_channel_resume_payload_consumed", False) is True

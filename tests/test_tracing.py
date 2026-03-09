from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.services.channel.session import ChannelSession
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.runner.facade import RunFacade
from aethergraph.services.tracing import EventLogTracer, summarize_payload


class FakeEventLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def append(self, evt: dict) -> None:
        self.rows.append(evt)

    async def query(self, **kwargs):
        return self.rows

    async def get_many(self, scope_id: str, event_ids: list[str]):
        return [row for row in self.rows if row.get("id") in event_ids]


class _FakeBus:
    def get_default_channel_key(self) -> str:
        return "ui:session/test-session"

    def resolve_channel_key(self, key: str) -> str:
        return key


class _FakeContext:
    def __init__(self, tracer, *, resume_payload=None):
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.resume_payload = resume_payload
        self.services = SimpleNamespace(
            channels=_FakeBus(),
            continuation_store=None,
            memory_facade=None,
            tracer=tracer,
        )

    async def create_continuation(self, **kwargs):
        raise AssertionError("create_continuation should not be called for replay resume payloads")

    def prepare_wait_for_resume(self, token: str):
        raise AssertionError(
            "prepare_wait_for_resume should not be called for replay resume payloads"
        )


class _FakeRunManager:
    async def submit_run(self, **kwargs):
        return SimpleNamespace(run_id=kwargs.get("run_id") or "run-child-1")


@pytest.mark.asyncio
async def test_trace_payload_summary_hashes_and_preview() -> None:
    summary = summarize_payload({"prompt": "x" * 400, "count": 3})
    assert summary["metadata"]["type"] == "dict"
    assert summary["metadata"]["count"] == 2
    assert "sha256" in summary["hashes"]
    assert summary["preview"]["prompt"].endswith("...")


@pytest.mark.asyncio
async def test_channel_ask_text_emits_trace_rows_for_resume_flow() -> None:
    log = FakeEventLog()
    tracer = EventLogTracer(event_log=log)
    ctx = _FakeContext(
        tracer,
        resume_payload={
            "_channel_wait_kind": "user_input",
            "prompt": "What is your name?",
            "text": "Ada",
        },
    )
    chan = ChannelSession(ctx)

    reply = await chan.ask_text("What is your name?")

    assert reply == "Ada"
    phases = [row["payload"]["phase"] for row in log.rows]
    operations = [row["payload"]["operation"] for row in log.rows]
    assert "ask_text" in operations
    assert "_ask_core:user_input" in operations
    assert "start" in phases
    assert "resume" in phases
    assert "end" in phases


@pytest.mark.asyncio
async def test_runner_spawn_run_emits_target_run_id() -> None:
    log = FakeEventLog()
    tracer = EventLogTracer(event_log=log)
    token = current_meter_context.set(
        {"run_id": "parent-run", "trace_id": "tr_parent", "span_id": None}
    )
    try:
        from aethergraph.core.runtime.runtime_services import use_services

        facade = RunFacade(
            run_manager=_FakeRunManager(),
            identity=None,
            session_id="session-1",
            agent_id="agent-1",
            app_id="app-1",
        )
        with use_services(SimpleNamespace(tracer=tracer)):
            run_id = await facade.spawn_run("child-graph", inputs={"x": 1}, run_id="run-child-42")
    finally:
        current_meter_context.reset(token)

    assert run_id == "run-child-42"
    end_rows = [row for row in log.rows if row["payload"]["phase"] == "end"]
    assert end_rows[-1]["payload"]["target_run_id"] == "run-child-42"
    assert end_rows[-1]["payload"]["parent_span_id"] is None


@pytest.mark.asyncio
async def test_llm_chat_emits_trace_with_usage_metrics() -> None:
    log = FakeEventLog()
    tracer = EventLogTracer(event_log=log)
    client = GenericLLMClient(provider="openai", model="gpt-test")

    async def fake_chat_dispatch(messages, **kwargs):
        return "hello back", {"prompt_tokens": 11, "completion_tokens": 7}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    services = SimpleNamespace(tracer=tracer)
    token = current_meter_context.set(
        {
            "run_id": "run-1",
            "graph_id": "graph-1",
            "trace_id": "tr_root",
            "span_id": None,
        }
    )
    try:
        from aethergraph.core.runtime.runtime_services import use_services

        with use_services(services):
            text, usage = await client.chat([{"role": "user", "content": "hello"}])
    finally:
        current_meter_context.reset(token)

    assert text == "hello back"
    assert usage["completion_tokens"] == 7
    end_rows = [
        row
        for row in log.rows
        if row["payload"]["operation"] == "chat" and row["payload"]["phase"] == "end"
    ]
    assert end_rows
    payload = end_rows[-1]["payload"]
    assert payload["metrics"]["prompt_tokens"] == 11
    assert payload["metrics"]["completion_tokens"] == 7

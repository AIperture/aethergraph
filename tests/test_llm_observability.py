from __future__ import annotations

from pathlib import Path
import json

import pytest

from aethergraph.config.config import AppSettings
from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.services.container.default_container import build_default_container
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.observability import (
    JsonlLLMObservationSink,
    LLMObservationRecord,
    ConsoleLLMObservationSink,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_console_sink_compact_view_renders_prompt_and_output(capsys) -> None:
    sink = ConsoleLLMObservationSink(prompt_view="compact", width=60, truncation_chars=80)
    record = LLMObservationRecord.new(
        call_type="chat",
        provider="openai",
        model="gpt-4o-mini",
        dimensions={"run_id": "run-1", "graph_id": "graph-1"},
        messages=[
            {"role": "system", "content": "You are concise and helpful."},
            {"role": "user", "content": "Explain attention in one sentence."},
        ],
        reasoning_effort=None,
        max_output_tokens=64,
        output_format="text",
        json_schema=None,
        schema_name="output",
        strict_schema=True,
        validate_json=True,
        extra_params={},
        trace_payload=None,
    )
    record.raw_text = "Attention weights the most relevant inputs for the current prediction."
    record.usage = {"prompt_tokens": 12, "completion_tokens": 9}
    record.latency_ms = 1546

    await sink.emit(record, capture_mode="full")
    out = capsys.readouterr().out
    assert "LLM CALL  openai/gpt-4o-mini" in out
    assert "[SYSTEM]" in out
    assert "[USER]" in out
    assert "[OUTPUT]" in out
    assert "tokens:  in=12  out=9  total=21" in out


@pytest.mark.asyncio
async def test_llm_observation_non_stream_success_full_capture(tmp_path: Path) -> None:
    sink_path = tmp_path / "llm_calls.jsonl"
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=JsonlLLMObservationSink(sink_path),
        observation_capture_mode="full",
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return "hello back", {"prompt_tokens": 11, "completion_tokens": 7}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    token = current_meter_context.set(
        {"run_id": "run-1", "graph_id": "graph-1", "user_id": "user-1", "org_id": "org-1"}
    )
    try:
        text, usage = await client.chat(
            [{"role": "user", "content": "hello"}],
            max_output_tokens=128,
            trace_payload={"stage": "test"},
        )
    finally:
        current_meter_context.reset(token)

    assert text == "hello back"
    assert usage["prompt_tokens"] == 11
    rows = _read_jsonl(sink_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-test"
    assert row["run_id"] == "run-1"
    assert row["messages"][0]["content"] == "hello"
    assert row["raw_text"] == "hello back"
    assert row["usage"]["completion_tokens"] == 7
    assert row["trace_payload"]["stage"] == "test"
    assert row["latency_ms"] is not None


@pytest.mark.asyncio
async def test_llm_observation_non_stream_failure_emits_record(tmp_path: Path) -> None:
    sink_path = tmp_path / "llm_calls.jsonl"
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=JsonlLLMObservationSink(sink_path),
        observation_capture_mode="full",
    )

    async def fake_chat_dispatch(messages, **kwargs):
        raise RuntimeError("boom")

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="boom"):
        await client.chat([{"role": "user", "content": "hello"}])

    rows = _read_jsonl(sink_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["error_type"] == "RuntimeError"
    assert row["error_message"] == "boom"
    assert row["raw_text"] is None


@pytest.mark.asyncio
async def test_llm_observation_stream_success_full_capture(tmp_path: Path) -> None:
    sink_path = tmp_path / "llm_calls.jsonl"
    client = GenericLLMClient(
        provider="openai",
        model="gpt-stream",
        observation_sink=JsonlLLMObservationSink(sink_path),
        observation_capture_mode="full",
    )

    async def fake_stream(messages, **kwargs):
        return "streamed output", {"input_tokens": 3, "output_tokens": 5}

    client._chat_openai_responses_stream = fake_stream  # type: ignore[method-assign]

    text, usage = await client.chat_stream([{"role": "user", "content": "hi"}])
    assert text == "streamed output"
    assert usage["output_tokens"] == 5

    rows = _read_jsonl(sink_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["call_type"] == "chat_stream"
    assert row["raw_text"] == "streamed output"
    assert row["usage"]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_llm_observation_metadata_mode_redacts_bodies(tmp_path: Path) -> None:
    sink_path = tmp_path / "llm_calls.jsonl"
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=JsonlLLMObservationSink(sink_path),
        observation_capture_mode="metadata",
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return "secret answer", {"prompt_tokens": 2, "completion_tokens": 4}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    await client.chat([{"role": "user", "content": "secret prompt"}], trace_payload={"foo": "bar"})

    rows = _read_jsonl(sink_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["messages"] is None
    assert row["raw_text"] is None
    assert row["trace_payload"] is None
    assert row["messages_preview"]["count"] == 1
    assert row["raw_text_preview"]["length"] == len("secret answer")


@pytest.mark.asyncio
async def test_default_container_llm_observability_writes_jsonl(tmp_path: Path) -> None:
    settings = AppSettings(
        root=str(tmp_path),
        deploy_mode="local",
        llm={
            "enabled": True,
            "default": {"provider": "openai", "model": "gpt-container"},
            "observability": {
                "enabled": True,
                "sink": "file",
                "path": "events/llm/custom_calls.jsonl",
                "capture_mode": "full",
            },
        },
    )
    container = build_default_container(root=str(tmp_path), cfg=settings)
    assert container.llm is not None
    client = container.llm.get()

    async def fake_chat_dispatch(messages, **kwargs):
        return "container output", {"prompt_tokens": 9, "completion_tokens": 1}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    await client.chat([{"role": "user", "content": "from container"}])

    sink_path = tmp_path / "events" / "llm" / "custom_calls.jsonl"
    assert sink_path.exists()
    rows = _read_jsonl(sink_path)
    assert len(rows) == 1
    assert rows[0]["raw_text"] == "container output"

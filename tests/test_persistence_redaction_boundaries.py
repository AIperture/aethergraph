from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aethergraph.contracts.services.runtime_output import RuntimeOutputFrame
from aethergraph.services.inspect.agent_events import emit_agent_event
from aethergraph.services.observability import (
    LLMObservationRecord,
    ObservationPolicy,
    ObservationRecord,
    SQLiteObservationStore,
)
from aethergraph.services.observability.prompt_store import PromptStore
from aethergraph.services.runtime_output import EventLogRuntimeOutputSink

_DATA_URL = "data:image/png;base64,c2Vuc2l0aXZlLWJ5dGVz"
_MARKER = "[redacted data URL: image/png]"


class _EventLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


@pytest.mark.asyncio
async def test_observation_persistence_applies_canonical_redaction(tmp_path: Path) -> None:
    store = SQLiteObservationStore(tmp_path / "observability.db")
    observation_id = await store.append_observation(
        ObservationRecord(
            category="tool",
            name="result",
            summary=f"result {_DATA_URL}",
            attributes={"image": _DATA_URL, "binary": b"payload"},
        )
    )

    persisted = await store.get_observation(observation_id)
    assert persisted["summary"] == f"result {_MARKER}"
    assert persisted["attributes"]["image"] == _MARKER
    assert persisted["attributes"]["binary"]["binary_bytes"] == 7


def test_prompt_persistence_applies_canonical_redaction() -> None:
    record = LLMObservationRecord.new(
        call_type="chat",
        provider="test",
        model="test",
        dimensions={},
        messages=[{"role": "user", "content": _DATA_URL}],
        reasoning_effort=None,
        max_output_tokens=None,
        output_format="text",
        json_schema=None,
        schema_name=None,
        strict_schema=None,
        validate_json=None,
        extra_params={},
        request_args={},
        provider_request_args={"attachment": _DATA_URL},
        compatibility_notes=[],
        trace_payload=None,
    )
    record.raw_text = _DATA_URL

    capture = PromptStore(ObservationPolicy(capture_mode="full")).prepare(record)
    bodies = [fragment.body for fragment in capture.fragments]
    assert all(_DATA_URL not in body for body in bodies)
    assert all(_MARKER in body for body in bodies)


@pytest.mark.asyncio
async def test_agent_event_persistence_applies_canonical_redaction() -> None:
    event_log = _EventLog()
    await emit_agent_event(
        event_type="tool.completed",
        summary=f"completed {_DATA_URL}",
        payload={"image": _DATA_URL, "binary": b"payload"},
        tags=[_DATA_URL],
        event_log=event_log,
    )

    envelope = event_log.rows[0]["payload"]
    assert envelope["summary"] == f"completed {_MARKER}"
    assert envelope["payload"]["image"] == _MARKER
    assert envelope["payload"]["binary"]["binary_bytes"] == 7
    assert envelope["tags"] == [_MARKER]


@pytest.mark.asyncio
async def test_runtime_output_persistence_applies_canonical_redaction() -> None:
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(event_log=event_log)
    sink.emit(
        RuntimeOutputFrame(
            execution_id="execution-1",
            run_id="run-1",
            session_id="session-1",
            graph_id="graph-1",
            node_id="node-1",
            tool_name="tool-1",
            stream="stdout",
            sequence=1,
            text=f"output {_DATA_URL}",
        )
    )
    await sink.flush_execution("execution-1")
    await sink.close()

    assert event_log.rows[0]["payload"]["text"] == f"output {_MARKER}"

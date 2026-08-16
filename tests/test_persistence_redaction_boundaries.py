from __future__ import annotations

import pytest

from aethergraph.observability import (
    LLMObservationRecord,
    ObservationPolicy,
    ObservationRecord,
    emit_agent_event,
)
from aethergraph.observability.prompt_store import PromptStore

_DATA_URL = "data:image/png;base64,c2Vuc2l0aXZlLWJ5dGVz"
_MARKER = "[redacted data URL: image/png]"


class _ObservationSink:
    def __init__(self) -> None:
        self.rows: list[ObservationRecord] = []

    async def append_observation(self, row: ObservationRecord) -> str:
        self.rows.append(row)
        return row.observation_id


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
    sink = _ObservationSink()
    await emit_agent_event(
        event_type="tool.completed",
        summary=f"completed {_DATA_URL}",
        payload={"image": _DATA_URL, "binary": b"payload"},
        tags=[_DATA_URL],
        observation_sink=sink,
    )

    record = sink.rows[0]
    assert record.summary == f"completed {_MARKER}"
    assert record.attributes["payload"]["image"] == _MARKER
    assert record.attributes["payload"]["binary"]["binary_bytes"] == 7
    assert record.attributes["tags"] == [_MARKER]

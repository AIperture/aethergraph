"""Opt-in live OpenAI contract for Tool-surface continuity."""

from __future__ import annotations

import json
import os

import pytest

from aethergraph.services.llm import (
    ToolCallOutput,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolDiscoveryRequest,
    ToolDiscoveryResult,
    ToolPath,
)
from aethergraph.services.llm.generic_client import GenericLLMClient

pytestmark = pytest.mark.skipif(
    os.getenv("AG_RUN_OPENAI_TOOL_SURFACE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set AG_RUN_OPENAI_TOOL_SURFACE_SMOKE=1 and OPENAI_API_KEY to run live smoke",
)

_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "minLength": 1},
        "exact_tool_names": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
    },
    "required": ["goal"],
    "additionalProperties": False,
}


def _tools() -> tuple[ToolDefinition, ToolDefinition]:
    path = ToolPath("contract.probes", "Conversation contract deferred probe namespace.")
    return (
        ToolDefinition(
            "finish",
            "Terminal action. Call after deferred_receipt_probe reports completion.",
            {
                "type": "object",
                "properties": {"receipt": {"type": "string"}},
                "required": ["receipt"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            "deferred_receipt_probe",
            "Run one deferred receipt probe, then call finish. Never call this twice.",
            {
                "type": "object",
                "properties": {"receipt": {"type": "string"}},
                "required": ["receipt"],
                "additionalProperties": False,
            },
            exposure="deferred",
            path=path,
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["native_client", "native_hosted"])
async def test_openai_continuation_retains_finish_after_deferred_tool_result(mode: str) -> None:
    """Require the real provider to see and select the immediate terminal Tool."""

    model = os.getenv("AG_OPENAI_TOOL_SURFACE_SMOKE_MODEL", "gpt-5.6-luna")
    client = GenericLLMClient("openai", model, api_key=os.environ["OPENAI_API_KEY"])
    assert client._tool_discovery_capabilities is not None
    tools = _tools()
    discovery = ToolDiscoveryRequest(
        mode,
        max_results=5,
        search_schema=_SEARCH_SCHEMA if mode == "native_client" else None,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Run this exact diagnostic contract. First search for and load "
                "deferred_receipt_probe. Call deferred_receipt_probe exactly once. "
                "After its result, call finish exactly once with receipt complete. "
                "Never repeat a completed Tool and never answer with ordinary text."
            ),
        },
        {"role": "user", "content": "Run the deferred Tool continuation diagnostic."},
    ]
    initial_request = ToolCallRequest(
        tools=tools,
        choice="required",
        discovery=discovery,
        turn_id=f"live-{mode}",
    )
    try:
        initial, _usage = await client.chat(
            messages,
            tool_request=initial_request,
            reasoning_effort="low",
            max_output_tokens=1_024,
        )
        assert isinstance(initial, ToolCallResponse)
        selected = initial
        if mode == "native_client":
            assert initial.discovery_events
            assert initial.transport_checkpoint is not None
            discovery_event = initial.discovery_events[0]
            assert len(discovery_event.provider_reference_ids) == 1
            selected_request = ToolCallRequest(
                tools=tools,
                choice="required",
                discovery=discovery,
                turn_id=f"live-{mode}",
                active_tool_names=("deferred_receipt_probe",),
                transport_checkpoint=initial.transport_checkpoint,
                discovery_result=ToolDiscoveryResult(
                    discovery_event_id=discovery_event.event_id,
                    provider_reference_id=discovery_event.provider_reference_ids[0],
                    status="completed",
                    tool_names=("deferred_receipt_probe",),
                ),
            )
            selected, _usage = await client.chat(
                messages,
                tool_request=selected_request,
                reasoning_effort="low",
                max_output_tokens=1_024,
            )
            assert isinstance(selected, ToolCallResponse)

        assert selected.calls
        assert [call.name for call in selected.calls] == ["deferred_receipt_probe"]
        assert selected.transport_checkpoint is not None
        deferred_call = selected.calls[0]
        result_request = ToolCallRequest(
            tools=tools,
            choice="required",
            discovery=discovery,
            turn_id=f"live-{mode}",
            active_tool_names=("deferred_receipt_probe",),
            transport_checkpoint=selected.transport_checkpoint,
            tool_outputs=(
                ToolCallOutput(
                    deferred_call.call_id,
                    json.dumps({"receipt": "deferred probe completed"}),
                ),
            ),
        )
        finished, _usage = await client.chat(
            messages,
            tool_request=result_request,
            reasoning_effort="low",
            max_output_tokens=1_024,
        )

        assert isinstance(finished, ToolCallResponse)
        assert [call.name for call in finished.calls] == ["finish"]
    finally:
        await client.aclose()

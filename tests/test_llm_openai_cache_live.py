"""Opt-in live OpenAI cache smoke for native Tool continuation."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from aethergraph.services.llm import (
    PromptCacheRequest,
    ToolCallOutput,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolDiscoveryCapabilities,
    ToolDiscoveryModeCapability,
    ToolDiscoveryRequest,
)
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.usage import normalize_llm_usage

pytestmark = pytest.mark.skipif(
    os.getenv("AG_RUN_OPENAI_CACHE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set AG_RUN_OPENAI_CACHE_SMOKE=1 and OPENAI_API_KEY to run live smoke",
)


@pytest.mark.asyncio
async def test_openai_native_continuation_cache_grows_with_multi_entry_ledger() -> None:
    """Exercise the actual growing-ledger shape across search and Tool output."""

    model = os.getenv("AG_OPENAI_CACHE_SMOKE_MODEL", "gpt-5.6")
    client = GenericLLMClient(
        "openai",
        model,
        api_key=os.environ["OPENAI_API_KEY"],
    )
    client.bind_tool_discovery_capabilities(
        ToolDiscoveryCapabilities(
            provider="openai",
            model=model,
            endpoint_family="responses",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="native_client",
                    replay_requirement="previous_response",
                    max_results=50,
                    protocol_version="responses.tool_search",
                ),
            ),
        )
    )
    read_document = ToolDefinition(
        "read_document",
        "Read the requested document and return its text.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        exposure="deferred",
    )
    family = f"openai.native-ledger-live.{uuid4().hex}"
    stable_header = " ".join(
        [
            "Stable instruction: use client Tool search to load read_document, "
            "call it once for report.md, then summarize its returned contents."
        ]
        * 260
    )
    initial_messages = [
        {"role": "system", "content": stable_header},
        {"role": "user", "content": "ledger request: inspect report.md"},
        {"role": "user", "content": "volatile frame: cycle 0; search before answering"},
    ]
    initial_request = ToolCallRequest(
        tools=(read_document,),
        choice="auto",
        discovery=ToolDiscoveryRequest("native_client", max_results=10),
        turn_id="live-turn-1",
    )

    try:
        first, first_usage = await client.chat(
            initial_messages,
            tool_request=initial_request,
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=512,
        )
        assert isinstance(first, ToolCallResponse)
        assert first.discovery_events
        assert first.transport_checkpoint is not None

        search_messages = [
            *initial_messages[:2],
            {"role": "assistant", "content": "ledger discovery event: document tools requested"},
            {"role": "user", "content": "ledger plan event: read report.md before summarizing"},
            {"role": "user", "content": "ledger observation: read_document activated"},
            {"role": "user", "content": "volatile frame: cycle 1; call read_document now"},
        ]
        loaded_request = ToolCallRequest(
            tools=(read_document,),
            choice="auto",
            discovery=ToolDiscoveryRequest("native_client", max_results=10),
            turn_id="live-turn-1",
            active_tool_names=("read_document",),
            transport_checkpoint=first.transport_checkpoint,
        )
        second, second_usage = await client.chat(
            search_messages,
            tool_request=loaded_request,
            prompt_cache=PromptCacheRequest((0, 1, 2, 3, 4), family),
            reasoning_effort="low",
            max_output_tokens=512,
        )
        assert isinstance(second, ToolCallResponse)
        assert second.calls
        assert second.calls[0].name == "read_document"
        assert second.transport_checkpoint is not None

        result_messages = [
            *search_messages[:5],
            {"role": "assistant", "content": "ledger tool call: read_document(report.md)"},
            {"role": "user", "content": "ledger tool result: quarterly status is green"},
            {"role": "user", "content": "ledger plan event: document inspection complete"},
            {"role": "user", "content": "ledger observation: final summary is now available"},
            {"role": "user", "content": "volatile frame: cycle 2; summarize the result"},
        ]
        completed_request = ToolCallRequest(
            tools=(read_document,),
            choice="auto",
            discovery=ToolDiscoveryRequest("native_client", max_results=10),
            turn_id="live-turn-1",
            active_tool_names=("read_document",),
            transport_checkpoint=second.transport_checkpoint,
            tool_outputs=(
                ToolCallOutput(
                    second.calls[0].call_id,
                    '{"path":"report.md","text":"Quarterly status is green."}',
                ),
            ),
        )
        third, third_usage = await client.chat(
            result_messages,
            tool_request=completed_request,
            prompt_cache=PromptCacheRequest(tuple(range(9)), family),
            reasoning_effort="low",
            max_output_tokens=512,
        )
        assert isinstance(third, ToolCallResponse)

        normalized = [
            normalize_llm_usage(usage) for usage in (first_usage, second_usage, third_usage)
        ]
        assert normalized[2]["cache_read_tokens"] > normalized[1]["cache_read_tokens"], normalized
    finally:
        await client.aclose()

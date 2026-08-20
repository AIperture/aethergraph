"""Opt-in GenericLLMClient parity test for Tool-search schema caching."""

from __future__ import annotations

import os

import pytest

from aethergraph.services.llm import (
    PromptCacheRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolDiscoveryCapabilities,
    ToolDiscoveryModeCapability,
    ToolDiscoveryRequest,
    ToolPath,
)
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.observability import ConsoleLLMObservationSink

from .openai_tool_search_cache_scenario import (
    SEARCH_SCHEMA,
    assert_usage_consistent,
    build_cache_scenario,
    normalize_usage,
)

pytestmark = pytest.mark.skipif(
    os.getenv("AG_RUN_OPENAI_CACHE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set AG_RUN_OPENAI_CACHE_SMOKE=1 and OPENAI_API_KEY to run live smoke",
)


@pytest.mark.asyncio
async def test_generic_client_reports_search_and_activation_cache_usage() -> None:
    """Measure exact client search/load replays through the AG adapter."""

    scenario = build_cache_scenario(os.getenv("AG_OPENAI_CACHE_SMOKE_MODEL", "gpt-5.6-luna"))
    family = f"ag-cache-adapter-{scenario.scenario_id}"
    client = GenericLLMClient(
        "openai",
        scenario.model,
        api_key=os.environ["OPENAI_API_KEY"],
        observation_sink=ConsoleLLMObservationSink(prompt_view="off"),
        observation_capture_mode="manifest",
    )
    client.bind_tool_discovery_capabilities(
        ToolDiscoveryCapabilities(
            provider="openai",
            model=scenario.model,
            endpoint_family="responses",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="native_client",
                    replay_requirement="previous_response",
                    max_results=50,
                    protocol_version="responses.tool_search",
                    selection_owner="application",
                    tool_representation="search_schema_manifest",
                    inventory_timing="search",
                    path_transport="manifest",
                ),
            ),
        )
    )
    path = ToolPath("cache.documents.read", "Read cached document partitions.")
    tools = tuple(
        ToolDefinition(
            value["name"],
            value["description"],
            value["parameters"],
            exposure="deferred",
            path=path,
        )
        for value in scenario.tools
    )
    discovery = ToolDiscoveryRequest("native_client", max_results=50, search_schema=SEARCH_SCHEMA)
    initial_request = ToolCallRequest(
        tools=tools,
        choice="auto",
        discovery=discovery,
        turn_id="cache-smoke",
    )
    try:
        initial, initial_usage = await client.chat(
            list(scenario.messages),
            tool_request=initial_request,
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
        assert isinstance(initial, ToolCallResponse)
        assert initial.transport_checkpoint is not None
        search_replay, search_replay_usage = await client.chat(
            list(scenario.messages),
            tool_request=initial_request,
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
        assert isinstance(search_replay, ToolCallResponse)
        assert search_replay.transport_checkpoint is not None
        loaded_request = ToolCallRequest(
            tools=tools,
            choice="auto",
            discovery=discovery,
            turn_id="cache-smoke",
            active_tool_names=tuple(tool.name for tool in tools),
            transport_checkpoint=initial.transport_checkpoint,
        )
        _, activation_usage = await client.chat(
            list(scenario.messages),
            tool_request=loaded_request,
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
        _, activation_replay_usage = await client.chat(
            list(scenario.messages),
            tool_request=loaded_request,
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
    finally:
        await client.aclose()

    for label, usage in (
        ("client_search", initial_usage),
        ("client_search_replay", search_replay_usage),
        ("client_activation", activation_usage),
        ("client_activation_replay", activation_replay_usage),
    ):
        normalized = normalize_usage(usage)
        assert_usage_consistent(normalized)
        print(
            "OpenAI AG cache diagnostic "
            f"label={label} model={scenario.model} "
            f"input_tokens={normalized.input_tokens} "
            f"output_tokens={normalized.output_tokens} "
            f"cache_read_tokens={normalized.cache_read_tokens} "
            f"cache_write_tokens={normalized.cache_write_tokens}"
        )

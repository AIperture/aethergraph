"""Opt-in live cache regression for ordinary OpenAI Responses Tool continuations."""

from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from aethergraph.services.llm import (
    PromptCacheRequest,
    ToolCallOutput,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from aethergraph.services.llm.generic_client import GenericLLMClient

from .openai_tool_search_cache_scenario import normalize_usage

pytestmark = pytest.mark.skipif(
    os.getenv("AG_RUN_OPENAI_CACHE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set AG_RUN_OPENAI_CACHE_SMOKE=1 and OPENAI_API_KEY to run live smoke",
)


@pytest.mark.asyncio
async def test_openai_ordinary_tool_continuation_grows_latest_cache_boundary() -> None:
    """Require cache growth without provider-native Tool discovery."""

    model = os.getenv("AG_OPENAI_CACHE_SMOKE_MODEL", "gpt-5.6-luna")
    scenario_id = uuid4().hex
    stable_header = " ".join(
        [
            "Stable workflow contract: call the advance Tool exactly once for each "
            "model decision and preserve the append-only workflow ledger."
        ]
        * 360
    )
    stable_result_one = " ".join(
        ["Stable ledger result: the first workflow action completed successfully."] * 180
    )
    stable_result_two = " ".join(
        ["Stable ledger result: the second workflow action completed successfully."] * 180
    )
    tool = ToolDefinition(
        "advance",
        "Advance the workflow by one deterministic step.",
        {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    family = f"ordinary-tool-continuation-{scenario_id}"
    turn_id = f"turn-{scenario_id}"
    client = GenericLLMClient(
        "openai",
        model,
        api_key=os.environ["OPENAI_API_KEY"],
        observation_capture_mode="manifest",
    )

    first_messages = [
        {"role": "system", "content": stable_header},
        {"role": "user", "content": "Advance this workflow through its next step."},
        {"role": "user", "content": "Volatile frame: decision 0."},
    ]
    try:
        first, first_raw_usage = await client.chat(
            first_messages,
            tool_request=ToolCallRequest(
                tools=(tool,),
                choice="required",
                turn_id=turn_id,
            ),
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
        assert isinstance(first, ToolCallResponse)
        assert len(first.calls) == 1
        assert first.transport_checkpoint is not None

        second, second_raw_usage = await client.chat(
            first_messages,
            tool_request=ToolCallRequest(
                tools=(tool,),
                choice="required",
                turn_id=turn_id,
                transport_checkpoint=first.transport_checkpoint,
                tool_outputs=(
                    ToolCallOutput(
                        first.calls[0].call_id,
                        json.dumps({"status": "completed", "detail": stable_result_one}),
                    ),
                ),
            ),
            # React v3 pins the original request root throughout one active
            # provider exchange. The grown Ledger is tracked separately as
            # effective_messages and does not replace this cache request.
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
        assert isinstance(second, ToolCallResponse)
        assert len(second.calls) == 1
        assert second.transport_checkpoint is not None

        third, third_raw_usage = await client.chat(
            first_messages,
            tool_request=ToolCallRequest(
                tools=(tool,),
                choice="required",
                turn_id=turn_id,
                transport_checkpoint=second.transport_checkpoint,
                tool_outputs=(
                    ToolCallOutput(
                        second.calls[0].call_id,
                        json.dumps({"status": "completed", "detail": stable_result_two}),
                    ),
                ),
            ),
            prompt_cache=PromptCacheRequest((0, 1), family),
            reasoning_effort="low",
            max_output_tokens=256,
        )
        assert isinstance(third, ToolCallResponse)
    finally:
        await client.aclose()

    first_usage = normalize_usage(first_raw_usage)
    second_usage = normalize_usage(second_raw_usage)
    third_usage = normalize_usage(third_raw_usage)
    print(
        "ordinary OpenAI Tool cache "
        f"first(read={first_usage.cache_read_tokens},write={first_usage.cache_write_tokens}) "
        f"second(read={second_usage.cache_read_tokens},write={second_usage.cache_write_tokens}) "
        f"third(read={third_usage.cache_read_tokens},write={third_usage.cache_write_tokens})"
    )

    assert first_usage.cache_write_tokens > 0
    assert second_usage.cache_read_tokens > 0
    assert second_usage.cache_write_tokens > 0
    assert third_usage.cache_read_tokens > second_usage.cache_read_tokens

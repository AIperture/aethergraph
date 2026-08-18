"""Opt-in direct Responses API control for Tool-search schema caching."""

from __future__ import annotations

import logging
import os

import httpx
import pytest

from .openai_tool_search_cache_scenario import (
    assert_loaded_branch_replayed,
    build_cache_scenario,
    normalize_usage,
    raw_loaded_body,
    search_tool,
)

pytestmark = pytest.mark.skipif(
    os.getenv("AG_RUN_OPENAI_CACHE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set AG_RUN_OPENAI_CACHE_SMOKE=1 and OPENAI_API_KEY to run live smoke",
)
_LOG = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_raw_openai_replays_loaded_schema_branch() -> None:
    """Use raw HTTP as the provider control for the exact replay assertion."""

    scenario = build_cache_scenario(os.getenv("AG_OPENAI_CACHE_SMOKE_MODEL", "gpt-5.4"))
    cache_key = f"ag-cache-raw-{scenario.scenario_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    initial_body = {
        "model": scenario.model,
        "input": list(scenario.messages),
        "tools": [search_tool()],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "prompt_cache_key": cache_key,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 256,
    }
    async with httpx.AsyncClient(timeout=180) as client:
        initial = await client.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=initial_body,
        )
        initial.raise_for_status()
        initial_json = initial.json()
        search = next(
            item
            for item in initial_json.get("output", [])
            if item.get("type") == "tool_search_call"
        )
        loaded_body = raw_loaded_body(
            scenario,
            response_id=initial_json["id"],
            call_id=search["call_id"],
            cache_key=cache_key,
        )
        load = await client.post(
            "https://api.openai.com/v1/responses", headers=headers, json=loaded_body
        )
        load.raise_for_status()
        replay = await client.post(
            "https://api.openai.com/v1/responses", headers=headers, json=loaded_body
        )
        replay.raise_for_status()

    _LOG.info(
        "OpenAI raw cache smoke completed model=%s initial_response_id=%s load_response_id=%s replay_response_id=%s",
        scenario.model,
        initial_json["id"],
        load.json().get("id"),
        replay.json().get("id"),
    )

    assert_loaded_branch_replayed(
        load=normalize_usage(load.json().get("usage") or {}),
        replay=normalize_usage(replay.json().get("usage") or {}),
        loaded_schema_branch_floor=scenario.loaded_schema_branch_floor,
    )

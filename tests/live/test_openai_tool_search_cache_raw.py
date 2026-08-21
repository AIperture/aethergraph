"""Opt-in direct Responses API control for Tool-search schema caching."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
import pytest

from .openai_tool_search_cache_scenario import (
    assert_completed_response,
    build_cache_scenario,
    raw_client_search_body,
    raw_hosted_search_body,
    raw_loaded_body,
    request_fingerprint,
)

pytestmark = pytest.mark.skipif(
    os.getenv("AG_RUN_OPENAI_CACHE_SMOKE") != "1" or not os.getenv("OPENAI_API_KEY"),
    reason="set AG_RUN_OPENAI_CACHE_SMOKE=1 and OPENAI_API_KEY to run live smoke",
)
_LOG = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_raw_openai_reports_search_and_activation_cache_usage() -> None:
    """Measure exact search/activation replays without requiring cache admission."""

    scenario = build_cache_scenario(os.getenv("AG_OPENAI_CACHE_SMOKE_MODEL", "gpt-5.6-luna"))
    cache_key = f"ag-cache-raw-{scenario.scenario_id}"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": "application/json",
    }
    client_search_body = raw_client_search_body(scenario, cache_key=cache_key)
    client_search_replay_body = raw_client_search_body(scenario, cache_key=cache_key)
    hosted_search_body = raw_hosted_search_body(scenario, cache_key=cache_key)
    hosted_search_replay_body = raw_hosted_search_body(scenario, cache_key=cache_key)
    assert request_fingerprint(client_search_body) == request_fingerprint(client_search_replay_body)
    assert request_fingerprint(hosted_search_body) == request_fingerprint(hosted_search_replay_body)

    async with httpx.AsyncClient(timeout=180) as client:

        async def send(label: str, body: dict[str, Any]) -> dict[str, Any]:
            response = await client.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=body,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                _LOG.error(
                    "OpenAI cache diagnostic request failed label=%s model=%s "
                    "request_fingerprint=%s status=%s response=%s",
                    label,
                    scenario.model,
                    request_fingerprint(body)[:16],
                    response.status_code,
                    response.text[:2_000],
                )
                raise
            payload = response.json()
            usage = assert_completed_response(payload)
            _LOG.info(
                "OpenAI cache diagnostic label=%s model=%s request_fingerprint=%s "
                "response_id=%s input_tokens=%s output_tokens=%s "
                "cache_read_tokens=%s cache_write_tokens=%s",
                label,
                scenario.model,
                request_fingerprint(body)[:16],
                payload["id"],
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            )
            return payload

        client_search = await send("client_search", client_search_body)
        client_search_replay = await send("client_search_replay", client_search_replay_body)
        search = next(
            item
            for item in client_search.get("output", [])
            if item.get("type") == "tool_search_call" and item.get("execution") == "client"
        )
        loaded_body = raw_loaded_body(
            scenario,
            response_id=client_search["id"],
            call_id=search["call_id"],
            cache_key=cache_key,
        )
        loaded_replay_body = raw_loaded_body(
            scenario,
            response_id=client_search["id"],
            call_id=search["call_id"],
            cache_key=cache_key,
        )
        assert request_fingerprint(loaded_body) == request_fingerprint(loaded_replay_body)
        await send("client_activation", loaded_body)
        await send("client_activation_replay", loaded_replay_body)
        await send("hosted_search", hosted_search_body)
        await send("hosted_search_replay", hosted_search_replay_body)

    assert client_search_replay["id"] != client_search["id"]

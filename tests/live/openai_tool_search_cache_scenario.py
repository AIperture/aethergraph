"""Shared OpenAI deferred-schema cache scenario and assertions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import uuid4


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {"type": "string", "minLength": 1},
        "paths": {"type": "array", "items": {"type": "string"}, "default": []},
        "exact_tool_names": {
            "type": "array",
            "items": {"type": "string"},
            "default": [],
        },
        "effects": {"type": "array", "items": {"type": "string"}, "default": []},
    },
    "required": ["goal"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CacheScenario:
    scenario_id: str
    model: str
    messages: tuple[dict[str, str], ...]
    tools: tuple[dict, ...]
    loaded_schema_branch_floor: int


@dataclass(frozen=True)
class CacheUsage:
    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


def build_cache_scenario(model: str) -> CacheScenario:
    """Build one stable long-prefix scenario with exactly 50 deferred Tools."""

    tools = tuple(
        {
            "type": "function",
            "name": f"read_document_{index:02d}",
            "description": f"Read document partition {index:02d} by path and return text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative path."},
                    "section": {
                        "type": "string",
                        "description": "Optional named document section.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            "strict": True,
            "defer_loading": True,
        }
        for index in range(50)
    )
    stable_header = " ".join(
        [
            "Stable cache instruction: use client Tool search, load every document "
            "reader returned by the application, and call read_document_00 for report.md."
        ]
        * 320
    )
    messages = (
        {"role": "system", "content": stable_header},
        {"role": "user", "content": "Inspect report.md with the document reader."},
    )
    stable_chars = len(json.dumps(messages, sort_keys=True))
    schema_chars = len(json.dumps(tools, sort_keys=True))
    # Conservative character-to-token floor. It is intentionally below common
    # tokenizer ratios while still requiring the replay to include the loaded branch.
    loaded_floor = max(4_096, (stable_chars + schema_chars) // 7)
    return CacheScenario(
        scenario_id=uuid4().hex,
        model=model,
        messages=messages,
        tools=tools,
        loaded_schema_branch_floor=loaded_floor,
    )


def normalize_usage(value: dict) -> CacheUsage:
    """Normalize raw or AetherGraph OpenAI usage without retaining provider bodies."""

    details = dict(
        value.get("input_tokens_details")
        or value.get("prompt_tokens_details")
        or {}
    )
    return CacheUsage(
        input_tokens=int(value.get("input_tokens") or value.get("prompt_tokens") or 0),
        cache_read_tokens=int(
            value.get("cache_read_tokens")
            or details.get("cached_tokens")
            or 0
        ),
        cache_write_tokens=int(
            value.get("cache_write_tokens")
            or details.get("cache_write_tokens")
            or 0
        ),
    )


def assert_loaded_branch_replayed(
    *,
    load: CacheUsage,
    replay: CacheUsage,
    loaded_schema_branch_floor: int,
    tolerance: int = 256,
) -> None:
    """Prove exact replay reads the longer schema branch, not only the base prefix."""

    assert replay.cache_read_tokens >= load.cache_write_tokens - tolerance, (
        load,
        replay,
    )
    assert replay.cache_read_tokens >= loaded_schema_branch_floor, (
        loaded_schema_branch_floor,
        load,
        replay,
    )
    assert replay.cache_write_tokens <= tolerance, (load, replay)


def search_tool() -> dict:
    """Return the exact client-executed Responses Tool-search declaration."""

    return {
        "type": "tool_search",
        "execution": "client",
        "description": "Authorized path: cache.documents.read",
        "parameters": SEARCH_SCHEMA,
    }


def raw_loaded_body(
    scenario: CacheScenario,
    *,
    response_id: str,
    call_id: str,
    cache_key: str,
) -> dict:
    """Build the body reused byte-for-byte for load and exact replay."""

    return {
        "model": scenario.model,
        "previous_response_id": response_id,
        "input": [
            {
                "type": "tool_search_output",
                "execution": "client",
                "call_id": call_id,
                "status": "completed",
                "tools": list(scenario.tools),
            }
        ],
        "prompt_cache_key": cache_key,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 256,
    }


__all__ = [
    "CacheScenario",
    "CacheUsage",
    "SEARCH_SCHEMA",
    "assert_loaded_branch_replayed",
    "build_cache_scenario",
    "normalize_usage",
    "raw_loaded_body",
    "search_tool",
]

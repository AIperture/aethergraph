"""Shared OpenAI deferred-schema cache scenario and assertions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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


@dataclass(frozen=True)
class CacheUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


def build_cache_scenario(model: str) -> CacheScenario:
    """Build one stable cache diagnostic with exactly 50 deferred Tools."""

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
        * 120
    )
    messages = (
        {"role": "system", "content": stable_header},
        {"role": "user", "content": "Inspect report.md with the document reader."},
    )
    return CacheScenario(
        scenario_id=uuid4().hex,
        model=model,
        messages=messages,
        tools=tools,
    )


def normalize_usage(value: dict) -> CacheUsage:
    """Normalize raw or AetherGraph OpenAI usage without retaining provider bodies."""

    details = dict(value.get("input_tokens_details") or value.get("prompt_tokens_details") or {})
    return CacheUsage(
        input_tokens=int(value.get("input_tokens") or value.get("prompt_tokens") or 0),
        output_tokens=int(value.get("output_tokens") or value.get("completion_tokens") or 0),
        cache_read_tokens=int(value.get("cache_read_tokens") or details.get("cached_tokens") or 0),
        cache_write_tokens=int(
            value.get("cache_write_tokens") or details.get("cache_write_tokens") or 0
        ),
    )


def assert_usage_consistent(usage: CacheUsage) -> None:
    """Validate provider usage without requiring a cache admission outcome."""

    assert usage.input_tokens >= 0, usage
    assert usage.output_tokens >= 0, usage
    assert 0 <= usage.cache_read_tokens <= usage.input_tokens, usage
    assert 0 <= usage.cache_write_tokens <= usage.input_tokens, usage


def assert_completed_response(value: dict) -> CacheUsage:
    """Require one complete, non-truncated Responses result with valid usage."""

    assert isinstance(value.get("id"), str) and value["id"], value
    assert value.get("status") == "completed", value
    assert value.get("error") is None, value
    assert value.get("incomplete_details") is None, value
    usage = normalize_usage(value.get("usage") or {})
    assert_usage_consistent(usage)
    return usage


def request_fingerprint(value: dict) -> str:
    """Return a stable credential-free identity for one provider request body."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def search_tool() -> dict:
    """Return the exact client-executed Responses Tool-search declaration."""

    return {
        "type": "tool_search",
        "execution": "client",
        "description": "Authorized path: cache.documents.read",
        "parameters": SEARCH_SCHEMA,
    }


def raw_client_search_body(scenario: CacheScenario, *, cache_key: str) -> dict:
    """Build the byte-stable client-executed search request used for replay."""

    return {
        "model": scenario.model,
        "input": list(scenario.messages),
        "tools": [search_tool()],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "prompt_cache_key": cache_key,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 256,
    }


def raw_hosted_search_body(scenario: CacheScenario, *, cache_key: str) -> dict:
    """Build the current OpenAI hosted-search request used for exact replay."""

    return {
        "model": scenario.model,
        "input": list(scenario.messages),
        "tools": [*scenario.tools, {"type": "tool_search"}],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "prompt_cache_key": cache_key,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 256,
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
    "assert_completed_response",
    "assert_usage_consistent",
    "build_cache_scenario",
    "normalize_usage",
    "raw_client_search_body",
    "raw_hosted_search_body",
    "raw_loaded_body",
    "request_fingerprint",
    "search_tool",
]

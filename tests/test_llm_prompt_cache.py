from __future__ import annotations

import asyncio
from dataclasses import fields
import json
from typing import Any

import pytest

from aethergraph.services.llm import (
    PromptCacheRequest,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    prompt_cache,
)
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.prompt_cache import prepare_prompt_cache
from aethergraph.services.llm.types import LLMUnsupportedFeatureError
from aethergraph.storage.contracts import StorageScope


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.last_json: dict[str, Any] | None = None

    async def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Any = None,
    ) -> _FakeResponse:
        self.last_json = json
        return _FakeResponse(self.payload)


def test_prompt_cache_request_rejects_ambiguous_boundaries() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        PromptCacheRequest((2, 1), "agent.v1")

    with pytest.raises(ValueError, match="sorted and unique"):
        PromptCacheRequest((0, 0), "agent.v1")


def test_prompt_cache_partition_uses_only_stable_canonical_scope() -> None:
    canonical_fields = {item.name for item in fields(StorageScope)}

    assert set(prompt_cache._CACHE_SCOPE_KEYS) <= canonical_fields
    assert "app_id" not in prompt_cache._CACHE_SCOPE_KEYS
    assert {"run_id", "session_id", "node_id"}.isdisjoint(prompt_cache._CACHE_SCOPE_KEYS)


def test_prepare_openai_explicit_cache_is_deterministic_and_detached() -> None:
    messages = [
        {"role": "system", "content": "header"},
        {"role": "user", "content": "volatile"},
    ]
    request = PromptCacheRequest((0,), "agent.v1")

    first = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.6",
        scope_dimensions={
            "tenant_id": "tenant-1",
            "project_id": "project-1",
            "org_id": "local",
            "graph_id": "graph-1",
            "agent_id": "agent-1",
            "app_id": "deprecated-app-1",
            "span_id": "ignored-1",
        },
    )
    second = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.6",
        scope_dimensions={
            "tenant_id": "tenant-1",
            "project_id": "project-1",
            "org_id": "local",
            "graph_id": "graph-1",
            "agent_id": "agent-1",
            "app_id": "deprecated-app-2",
            "span_id": "ignored-2",
        },
    )
    other_graph = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.6",
        scope_dimensions={
            "tenant_id": "tenant-1",
            "project_id": "project-1",
            "org_id": "local",
            "graph_id": "graph-2",
            "agent_id": "agent-1",
        },
    )

    assert messages[0]["content"] == "header"
    assert first.provider_request_fields == second.provider_request_fields
    assert first.provider_request_fields != other_graph.provider_request_fields
    assert len(first.provider_request_fields["prompt_cache_key"]) == 64
    assert first.observation == second.observation
    assert first.observation == {
        "strategy": "stable_prefix",
        "requested_boundary_count": 1,
        "effective_boundary_count": 1,
        "effective_mode": "explicit",
        "capability_source": "openai_explicit_model_family",
        "key_fingerprint": first.observation["key_fingerprint"],
        "tool_contract_fingerprint": "",
        "tool_catalog_fingerprint": "",
        "tool_surface_fingerprint": "",
        "tool_discovery_mode": "",
        "max_new_writes_per_request": 4,
    }
    assert first.messages[0]["content"] == [
        {
            "type": "input_text",
            "text": "header",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    assert first.messages[1] == {"role": "user", "content": "volatile"}


def test_openai_responses_tool_exchange_keeps_implicit_latest_breakpoint() -> None:
    prepared = prepare_prompt_cache(
        PromptCacheRequest((0,), "agent.ledger.v1"),
        [
            {"role": "system", "content": "stable header"},
            {"role": "user", "content": "volatile frame"},
        ],
        provider="openai",
        model="gpt-5.6-luna",
        endpoint_id="openai_responses",
        tool_request=ToolCallRequest(
            tools=(
                ToolDefinition(
                    "advance",
                    "Advance the workflow.",
                    {"type": "object", "properties": {}},
                ),
            ),
            choice="required",
            turn_id="turn-1",
        ),
    )

    assert prepared.provider_request_fields.keys() == {"prompt_cache_key"}
    assert prepared.observation["implicit_latest_breakpoint"] is True
    assert prepared.observation["tool_discovery_mode"] == ""


def test_prompt_cache_resolution_honors_preselected_endpoint() -> None:
    prepared = prepare_prompt_cache(
        PromptCacheRequest((0,), "agent.v1"),
        [{"role": "system", "content": "stable"}],
        provider="openai",
        model="gpt-5.6",
        endpoint_id="openai_chat_completions",
    )

    assert prepared.observation["effective_mode"] == "unavailable"
    assert prepared.provider_request_fields == {}


def test_required_prompt_cache_rejects_incompatible_preselected_endpoint() -> None:
    with pytest.raises(LLMUnsupportedFeatureError, match="cataloged cache capability"):
        prepare_prompt_cache(
            PromptCacheRequest((0,), "agent.v1"),
            [{"role": "system", "content": "stable"}],
            provider="openai",
            model="gpt-5.6",
            endpoint_id="openai_chat_completions",
            policy="required",
        )


def test_prepare_openai_assistant_boundary_uses_output_text() -> None:
    prepared = prepare_prompt_cache(
        PromptCacheRequest((0,), "agent.ledger.v1"),
        [{"role": "assistant", "content": "Prior Tool selection."}],
        provider="openai",
        model="gpt-5.6",
    )

    assert prepared.messages[0]["content"] == [
        {
            "type": "output_text",
            "text": "Prior Tool selection.",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]


def test_prepare_openai_preserves_more_than_four_persistent_boundaries() -> None:
    messages = [{"role": "user", "content": f"stable segment {index}"} for index in range(6)]

    prepared = prepare_prompt_cache(
        PromptCacheRequest(tuple(range(6)), "agent.ledger.v1"),
        messages,
        provider="openai",
        model="gpt-5.6",
    )

    marked = [
        index
        for index, message in enumerate(prepared.messages)
        if message["content"][-1].get("prompt_cache_breakpoint") == {"mode": "explicit"}
    ]
    assert marked == list(range(6))
    assert prepared.observation["requested_boundary_count"] == 6
    assert prepared.observation["effective_boundary_count"] == 6
    assert prepared.observation["max_new_writes_per_request"] == 4


def test_prepare_openai_append_keeps_prior_breakpoints() -> None:
    initial_messages = [
        {"role": "user", "content": f"stable segment {index}"} for index in range(5)
    ]
    initial = prepare_prompt_cache(
        PromptCacheRequest(tuple(range(5)), "agent.ledger.v1"),
        initial_messages,
        provider="openai",
        model="gpt-5.6",
    )
    appended = prepare_prompt_cache(
        PromptCacheRequest(tuple(range(6)), "agent.ledger.v1"),
        [*initial_messages, {"role": "user", "content": "new stable segment"}],
        provider="openai",
        model="gpt-5.6",
    )

    assert appended.messages[:5] == initial.messages
    assert appended.messages[5]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_prepare_anthropic_limits_boundaries_and_preserves_latest() -> None:
    messages = [{"role": "user", "content": str(index)} for index in range(6)]

    prepared = prepare_prompt_cache(
        PromptCacheRequest(tuple(range(6)), "ledger.v1"),
        messages,
        provider="anthropic",
        model="claude-sonnet-4-5",
    )

    marked = [
        index
        for index, message in enumerate(prepared.messages)
        if message.get("cache_control") == {"type": "ephemeral"}
    ]
    assert marked == [0, 3, 4, 5]
    assert prepared.observation["effective_boundary_count"] == 4


def test_prepare_implicit_and_unavailable_modes_add_no_provider_fields() -> None:
    request = PromptCacheRequest((0,), "agent.v1")
    messages = [{"role": "system", "content": "header"}]

    openai = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.4",
    )
    gemini = prepare_prompt_cache(
        request,
        messages,
        provider="google",
        model="gemini-2.5-pro",
    )
    unknown = prepare_prompt_cache(
        request,
        messages,
        provider="custom",
        model="custom",
    )

    assert openai.observation["effective_mode"] == "implicit"
    assert gemini.observation["effective_mode"] == "implicit"
    assert unknown.observation["effective_mode"] == "unavailable"
    assert openai.provider_request_fields == {}
    assert gemini.provider_request_fields == {}
    assert unknown.provider_request_fields == {}


def test_disabled_policy_emits_no_cache_directive_or_key() -> None:
    messages = [{"role": "system", "content": "stable"}]

    prepared = prepare_prompt_cache(
        PromptCacheRequest((0,), "agent.v1"),
        messages,
        provider="openai",
        model="gpt-5.6",
        policy="disabled",
    )

    assert prepared.messages == tuple(messages)
    assert prepared.provider_request_fields == {}
    assert prepared.observation["effective_mode"] == "disabled"
    assert prepared.observation["effective_boundary_count"] == 0
    assert prepared.observation["key_fingerprint"] == ""


def test_required_policy_accepts_deepseek_implicit_cache_without_wire_fields() -> None:
    prepared = prepare_prompt_cache(
        PromptCacheRequest((0,), "agent.v1"),
        [{"role": "system", "content": "stable"}],
        provider="deepseek",
        model="deepseek-v4-pro",
        policy="required",
    )

    assert prepared.observation["effective_mode"] == "implicit"
    assert prepared.observation["capability_source"] == ("deepseek_automatic_prefix_cache")
    assert prepared.provider_request_fields == {}


def test_required_policy_rejects_unknown_cache_capability() -> None:
    with pytest.raises(LLMUnsupportedFeatureError, match="prompt_cache"):
        prepare_prompt_cache(
            PromptCacheRequest((0,), "agent.v1"),
            [{"role": "system", "content": "stable"}],
            provider="custom",
            model="custom-chat",
            policy="required",
        )


@pytest.mark.asyncio
async def test_required_policy_rejects_missing_stable_prefix_request() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5.6",
        api_key="test",
        prompt_cache_policy="required",
    )

    with pytest.raises(LLMUnsupportedFeatureError, match="explicit stable-prefix"):
        await client.chat([{"role": "user", "content": "volatile"}])


def test_prepare_prompt_cache_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="outside the message list"):
        prepare_prompt_cache(
            PromptCacheRequest((1,), "agent.v1"),
            [{"role": "system", "content": "header"}],
            provider="openai",
            model="gpt-5.6",
        )


def test_tool_contract_changes_rotate_cache_identity() -> None:
    request = PromptCacheRequest((0,), "agent.v1")
    messages = [{"role": "system", "content": "stable"}]
    first_tool_request = ToolCallRequest(
        tools=(ToolDefinition("read", "Read.", {"type": "object"}),)
    )
    second_tool_request = ToolCallRequest(
        tools=(ToolDefinition("write", "Write.", {"type": "object"}),)
    )

    first = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.6",
        tool_request=first_tool_request,
    )
    second = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.6",
        tool_request=second_tool_request,
    )

    assert first.provider_request_fields != second.provider_request_fields
    assert first.observation["tool_contract_fingerprint"]
    assert (
        first.observation["tool_contract_fingerprint"]
        != second.observation["tool_contract_fingerprint"]
    )
    assert first.observation["tool_catalog_fingerprint"]
    assert first.observation["tool_surface_fingerprint"]


@pytest.mark.asyncio
async def test_openai_chat_sends_explicit_cache_fields_and_markers() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {},
    }
    client = GenericLLMClient(provider="openai", model="gpt-5.6", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "volatile"},
        ],
        prompt_cache=PromptCacheRequest((0,), "agent.v1"),
    )

    assert text == "ok"
    assert fake_http.last_json is not None
    assert fake_http.last_json["prompt_cache_key"].startswith("agpc_")
    assert len(fake_http.last_json["prompt_cache_key"]) == 64
    assert fake_http.last_json["prompt_cache_options"] == {"mode": "explicit"}
    assert fake_http.last_json["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert "prompt_cache_breakpoint" not in str(fake_http.last_json["input"][1])


@pytest.mark.asyncio
async def test_openai_responses_tool_exchange_omits_explicit_only_cache_mode() -> None:
    payload = {
        "id": "resp-tool-1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call-advance-1",
                "name": "advance",
                "arguments": "{}",
            }
        ],
        "usage": {},
    }
    client = GenericLLMClient(provider="openai", model="gpt-5.6-luna", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, _usage = await client.chat(
        [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "advance"},
        ],
        tool_request=ToolCallRequest(
            tools=(
                ToolDefinition(
                    "advance",
                    "Advance the workflow.",
                    {"type": "object", "properties": {}},
                ),
            ),
            choice="required",
            turn_id="turn-1",
        ),
        prompt_cache=PromptCacheRequest((0,), "agent.ledger.v1"),
    )

    assert isinstance(response, ToolCallResponse)
    assert fake_http.last_json is not None
    assert fake_http.last_json["prompt_cache_key"].startswith("agpc_")
    assert "prompt_cache_options" not in fake_http.last_json
    assert fake_http.last_json["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


@pytest.mark.asyncio
async def test_openai_disabled_policy_keeps_cache_fields_off_wire() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {},
    }
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5.6",
        api_key="test",
        prompt_cache_policy="disabled",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [{"role": "system", "content": "stable"}],
        prompt_cache=PromptCacheRequest((0,), "agent.v1"),
    )

    assert text == "ok"
    assert fake_http.last_json is not None
    assert "prompt_cache_key" not in fake_http.last_json
    assert "prompt_cache_options" not in fake_http.last_json
    assert "prompt_cache_breakpoint" not in str(fake_http.last_json)


@pytest.mark.asyncio
async def test_openai_chat_preserves_assistant_history_as_output_text() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
        "usage": {},
    }
    client = GenericLLMClient(provider="openai", model="gpt-5.6", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "request"},
            {"role": "assistant", "content": "prior selection"},
            {"role": "user", "content": "prior result"},
        ],
        prompt_cache=PromptCacheRequest((0, 2), "agent.ledger.v1"),
    )

    assert text == "ok"
    assert fake_http.last_json is not None
    assistant = fake_http.last_json["input"][2]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {
            "type": "output_text",
            "text": "prior selection",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_chat_translates_cache_boundary_to_content_block() -> None:
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {},
    }
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-sonnet-4-5",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [
            {"role": "system", "content": "stable"},
            {"role": "user", "content": "volatile"},
        ],
        prompt_cache=PromptCacheRequest((0,), "agent.v1"),
    )

    assert text == "ok"
    assert fake_http.last_json is not None
    assert fake_http.last_json["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in str(fake_http.last_json["messages"])


@pytest.mark.asyncio
async def test_gemini_chat_preserves_implicit_cache_usage() -> None:
    payload = {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 5,
            "cachedContentTokenCount": 80,
        },
    }
    client = GenericLLMClient(provider="google", model="gemini-2.5-pro", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, usage = await client.chat(
        [{"role": "user", "content": "stable"}],
        prompt_cache=PromptCacheRequest((0,), "agent.v1"),
    )

    assert text == "ok"
    assert usage == {
        "input_tokens": 100,
        "output_tokens": 5,
        "cache_read_tokens": 80,
    }
    assert fake_http.last_json is not None
    assert "prompt_cache" not in str(fake_http.last_json)

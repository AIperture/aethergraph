from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aethergraph.services.llm import PromptCacheRequest
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.prompt_cache import prepare_prompt_cache


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
        scope_dimensions={"org_id": "local", "span_id": "ignored-1"},
    )
    second = prepare_prompt_cache(
        request,
        messages,
        provider="openai",
        model="gpt-5.6",
        scope_dimensions={"org_id": "local", "span_id": "ignored-2"},
    )

    assert messages[0]["content"] == "header"
    assert first.provider_request_fields == second.provider_request_fields
    assert first.observation == second.observation
    assert first.observation == {
        "strategy": "stable_prefix",
        "requested_boundary_count": 1,
        "effective_boundary_count": 1,
        "effective_mode": "explicit",
        "capability_source": "openai_explicit_model_family",
        "key_fingerprint": first.observation["key_fingerprint"],
    }
    assert first.messages[0]["content"] == [
        {
            "type": "input_text",
            "text": "header",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    assert first.messages[1] == {"role": "user", "content": "volatile"}


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


def test_prepare_prompt_cache_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="outside the message list"):
        prepare_prompt_cache(
            PromptCacheRequest((1,), "agent.v1"),
            [{"role": "system", "content": "header"}],
            provider="openai",
            model="gpt-5.6",
        )


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
    assert fake_http.last_json["prompt_cache_options"] == {"mode": "explicit"}
    assert fake_http.last_json["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert "prompt_cache_breakpoint" not in str(fake_http.last_json["input"][1])


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

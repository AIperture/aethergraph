from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from aethergraph.api.v1.schemas.settings import LLMProfilePayload
from aethergraph.api.v1.settings import _collect_llm_env
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.types import LLMUnsupportedFeatureError


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.last_json: dict[str, Any] | None = None

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any], timeout=None):
        self.last_json = json
        return _FakeResponse(self.payload)


@pytest.mark.asyncio
async def test_chat_json_alias_warns_and_returns_canonical_json(caplog) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")

    async def fake_chat_dispatch(messages, **kwargs):
        return '{"b":2,"a":1}', {"prompt_tokens": 1, "completion_tokens": 1}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with caplog.at_level(logging.WARNING):
        text, usage = await client.chat(
            [{"role": "user", "content": "hello"}],
            output_format="json",
            validate_json=True,
        )

    assert json.loads(text) == {"a": 1, "b": 2}
    assert usage["completion_tokens"] == 1
    assert "deprecated" in caplog.text


@pytest.mark.asyncio
async def test_chat_uses_profile_compatibility_policy_when_fail_flag_omitted() -> None:
    strict_client = GenericLLMClient(
        provider="openai", model="gpt-test", compatibility_policy="strict"
    )
    compat_client = GenericLLMClient(
        provider="openai", model="gpt-test", compatibility_policy="compat"
    )

    strict_seen: dict[str, Any] = {}
    compat_seen: dict[str, Any] = {}

    async def fake_dispatch_strict(messages, **kwargs):
        strict_seen.update(kwargs)
        return "ok", {}

    async def fake_dispatch_compat(messages, **kwargs):
        compat_seen.update(kwargs)
        return "ok", {}

    strict_client._chat_dispatch = fake_dispatch_strict  # type: ignore[method-assign]
    compat_client._chat_dispatch = fake_dispatch_compat  # type: ignore[method-assign]

    await strict_client.chat([{"role": "user", "content": "x"}])
    await compat_client.chat([{"role": "user", "content": "x"}])

    assert strict_seen["fail_on_unsupported"] is True
    assert compat_seen["fail_on_unsupported"] is False


@pytest.mark.asyncio
async def test_chat_stream_rejects_structured_output_modes() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")

    with pytest.raises(LLMUnsupportedFeatureError):
        await client.chat_stream(
            [{"role": "user", "content": "hello"}],
            output_format="json_object",
        )


@pytest.mark.asyncio
async def test_deepseek_non_streaming_uses_openai_compatible_body() -> None:
    payload = {
        "choices": [{"message": {"content": '{"answer":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    client = GenericLLMClient(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="ds-key",
        compatibility_policy="compat",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, usage = await client._chat_openai_like_chat_completions(  # type: ignore[misc]
        [{"role": "user", "content": "hello"}],
        model="deepseek-v4-pro",
        reasoning_effort="xhigh",
        max_output_tokens=256,
        output_format="json_object",
        json_schema=None,
        fail_on_unsupported=False,
    )

    assert json.loads(text) == {"answer": "ok"}
    assert usage["completion_tokens"] == 4
    assert fake_http.last_json is not None
    assert fake_http.last_json["response_format"] == {"type": "json_object"}
    assert fake_http.last_json["max_tokens"] == 256
    assert fake_http.last_json["reasoning_effort"] == "max"
    assert fake_http.last_json["thinking"] == {"type": "enabled"}


def test_collect_llm_env_includes_compatibility_policy() -> None:
    env = _collect_llm_env(
        {
            "DEEPSEEK": LLMProfilePayload(
                provider="deepseek",
                model="deepseek-v4-pro",
                reasoning_effort="high",
                thinking_mode="auto",
                compatibility_policy="compat",
            )
        }
    )

    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__REASONING_EFFORT"] == "high"
    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__THINKING_MODE"] == "auto"
    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__COMPATIBILITY_POLICY"] == "compat"


@pytest.mark.asyncio
async def test_chat_uses_profile_reasoning_effort_when_call_omits_it() -> None:
    client = GenericLLMClient(
        provider="deepseek",
        model="deepseek-v4-pro",
        api_key="ds-key",
        reasoning_effort="xhigh",
        thinking_mode="auto",
    )
    payload = {
        "choices": [{"message": {"content": '{"ok":true}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [{"role": "user", "content": "hello"}],
        output_format="json_object",
    )

    assert json.loads(text) == {"ok": True}
    assert fake_http.last_json is not None
    assert fake_http.last_json["reasoning_effort"] == "max"
    assert fake_http.last_json["thinking"] == {"type": "enabled"}

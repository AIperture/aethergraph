from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aethergraph.api.v1.schemas.settings import LLMProfilePayload
from aethergraph.api.v1.settings import _collect_llm_env
from aethergraph.config.llm import LLMProfile
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.service import LLMService
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
async def test_chat_json_alias_warns_and_returns_canonical_json() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")

    async def fake_chat_dispatch(messages, **kwargs):
        return '{"b":2,"a":1}', {"prompt_tokens": 1, "completion_tokens": 1}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    warnings: list[str] = []
    original_warning = client._logger.warning
    client._logger.warning = lambda msg: warnings.append(str(msg))  # type: ignore[assignment]
    try:
        text, usage = await client.chat(
            [{"role": "user", "content": "hello"}],
            output_format="json",
            validate_json=True,
        )
    finally:
        client._logger.warning = original_warning  # type: ignore[assignment]

    assert json.loads(text) == {"a": 1, "b": 2}
    assert usage["completion_tokens"] == 1
    assert any("deprecated" in message for message in warnings)


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


@pytest.mark.asyncio
async def test_lmstudio_json_object_uses_text_response_format_with_local_validation() -> None:
    payload = {
        "choices": [{"message": {"content": '{"answer":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    client = GenericLLMClient(
        provider="lmstudio",
        model="local-model",
        compatibility_policy="compat",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, usage = await client.chat(
        [{"role": "user", "content": "hello"}],
        output_format="json_object",
        validate_json=True,
    )

    assert json.loads(text) == {"answer": "ok"}
    assert usage["completion_tokens"] == 4
    assert fake_http.last_json is not None
    assert fake_http.last_json["response_format"] == {"type": "text"}


@pytest.mark.asyncio
async def test_lmstudio_json_object_strict_mode_rejects_text_fallback() -> None:
    client = GenericLLMClient(
        provider="lmstudio",
        model="local-model",
        compatibility_policy="strict",
    )

    with pytest.raises(RuntimeError, match="json_object"):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            output_format="json_object",
        )


@pytest.mark.asyncio
async def test_lmstudio_json_schema_uses_native_response_format() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    payload = {
        "choices": [{"message": {"content": '{"answer":"ok"}'}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    client = GenericLLMClient(provider="lmstudio", model="local-model")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [{"role": "user", "content": "hello"}],
        output_format="json_schema",
        json_schema=schema,
        schema_name="answer_schema",
        strict_schema=False,
    )

    assert json.loads(text) == {"answer": "ok"}
    assert fake_http.last_json is not None
    assert fake_http.last_json["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "schema": schema,
            "strict": False,
        },
    }


@pytest.mark.asyncio
async def test_anthropic_tools_are_not_silently_dropped_in_compat_mode() -> None:
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="anthropic-key",
        compatibility_policy="compat",
    )

    with pytest.raises(LLMUnsupportedFeatureError, match="tools"):
        await client._chat_anthropic_messages(  # type: ignore[misc]
            [{"role": "user", "content": "hello"}],
            model="claude-test",
            output_format="text",
            json_schema=None,
            fail_on_unsupported=False,
            tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
        )


@pytest.mark.asyncio
async def test_gemini_tools_are_not_passed_through_in_compat_mode() -> None:
    client = GenericLLMClient(
        provider="google",
        model="gemini-test",
        api_key="google-key",
        compatibility_policy="compat",
    )

    with pytest.raises(LLMUnsupportedFeatureError, match="tools"):
        await client._chat_gemini_generate_content(  # type: ignore[misc]
            [{"role": "user", "content": "hello"}],
            model="gemini-test",
            output_format="text",
            json_schema=None,
            fail_on_unsupported=False,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )


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


def test_collect_llm_env_includes_explicit_vision_fields() -> None:
    env = _collect_llm_env(
        {
            "local_vision": LLMProfilePayload(
                provider="lmstudio",
                model="local-vlm",
                vision_enabled=True,
                vision_max_images=3,
                vision_max_image_bytes=2048,
                vision_resize_enabled=True,
                vision_resize_max_dimension=1024,
                vision_resize_max_pixels=800_000,
                vision_resize_jpeg_quality=82,
                vision_resize_min_jpeg_quality=68,
                vision_accepted_mime_prefixes=["image/"],
                vision_accepted_mime_types=["image/png"],
            )
        }
    )

    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_ENABLED"] == "true"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_MAX_IMAGES"] == "3"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_MAX_IMAGE_BYTES"] == "2048"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_RESIZE_ENABLED"] == "true"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_RESIZE_MAX_DIMENSION"] == "1024"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_RESIZE_MAX_PIXELS"] == "800000"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_RESIZE_JPEG_QUALITY"] == "82"
    assert env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_RESIZE_MIN_JPEG_QUALITY"] == "68"
    assert (
        env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_ACCEPTED_MIME_PREFIXES"]
        == '["image/"]'
    )
    assert (
        env["AETHERGRAPH_LLM__PROFILES__LOCAL_VISION__VISION_ACCEPTED_MIME_TYPES"]
        == '["image/png"]'
    )


def test_llm_service_exposes_explicit_profile_metadata() -> None:
    profile = LLMProfile(
        provider="lmstudio",
        model="local-vlm",
        vision_enabled=True,
        vision_max_images=2,
        vision_max_image_bytes=4096,
        vision_resize_max_dimension=900,
        vision_resize_max_pixels=700_000,
        vision_resize_jpeg_quality=80,
        vision_resize_min_jpeg_quality=65,
    )
    service = LLMService(
        clients={"default": object()},  # type: ignore[arg-type]
        profiles={"default": profile},
    )

    exposed = service.profile("default")

    assert exposed is profile
    assert exposed.vision_enabled is True
    assert exposed.vision_max_images == 2
    assert exposed.vision_max_image_bytes == 4096
    assert exposed.vision_resize_enabled is True
    assert exposed.vision_resize_max_dimension == 900
    assert exposed.vision_resize_max_pixels == 700_000
    assert exposed.vision_resize_jpeg_quality == 80
    assert exposed.vision_resize_min_jpeg_quality == 65
    assert service.profile("missing") is None


def test_llm_service_configure_profile_updates_vision_metadata() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")
    service = LLMService(
        clients={"default": client},
        profiles={"default": LLMProfile(provider="openai", model="gpt-test")},
    )

    service.configure_profile(
        profile="default",
        vision_enabled=True,
        vision_max_images=1,
        vision_max_image_bytes=1024,
        vision_resize_enabled=False,
        vision_resize_max_dimension=768,
        vision_resize_max_pixels=400_000,
        vision_resize_jpeg_quality=78,
        vision_resize_min_jpeg_quality=62,
        vision_accepted_mime_types=["image/png"],
    )

    profile = service.profile("default")
    assert profile is not None
    assert profile.vision_enabled is True
    assert profile.vision_max_images == 1
    assert profile.vision_max_image_bytes == 1024
    assert profile.vision_resize_enabled is False
    assert profile.vision_resize_max_dimension == 768
    assert profile.vision_resize_max_pixels == 400_000
    assert profile.vision_resize_jpeg_quality == 78
    assert profile.vision_resize_min_jpeg_quality == 62
    assert profile.vision_accepted_mime_types == ["image/png"]


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

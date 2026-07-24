from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aethergraph.api.v1.schemas.settings import LLMProfilePayload
from aethergraph.config.llm import LLMProfile
from aethergraph.config.llm_env import encode_llm_profile_env
from aethergraph.services.llm import StructuredOutputRequest
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.service import LLMService
from aethergraph.services.llm.structured_output import prepare_structured_output
from aethergraph.services.llm.types import (
    LLMStructuredOutputCapabilityError,
    LLMUnsupportedFeatureError,
)


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


class _ObservationSink:
    def __init__(self) -> None:
        self.records = []

    async def emit(self, record, *, capture_mode) -> None:
        self.records.append(record)


def test_structured_output_request_detaches_caller_schema() -> None:
    schema = {"type": "object", "properties": {}}

    request = StructuredOutputRequest(name=" Answer ", schema=schema)
    schema["properties"]["late"] = {"type": "string"}

    assert request.name == "Answer"
    assert request.schema == {"type": "object", "properties": {}}


def test_openai_closed_schema_selects_native_strict_output() -> None:
    prepared = prepare_structured_output(
        StructuredOutputRequest(
            "Answer",
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        ),
        provider="openai",
        model="gpt-5-mini",
    )

    assert prepared.mode == "native_strict"
    assert prepared.provider_strict
    assert prepared.provider_schema == prepared.canonical_schema


def test_openai_free_form_schema_preserves_semantics_in_native_schema_mode() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inferred": {"type": "object", "additionalProperties": True},
        },
        "required": ["inferred"],
        "additionalProperties": False,
    }

    prepared = prepare_structured_output(
        StructuredOutputRequest("MetalensExtract", schema),
        provider="openai",
        model="gpt-5-mini",
    )

    assert prepared.mode == "native_schema"
    assert not prepared.provider_strict
    assert prepared.provider_schema == schema
    assert prepared.provider_schema["properties"]["inferred"]["additionalProperties"] is True
    assert any(item.code == "strict_object_not_closed" for item in prepared.diagnostics)


@pytest.mark.asyncio
async def test_openai_free_form_schema_is_sent_with_strict_disabled() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inferred": {"type": "object", "additionalProperties": True},
        },
        "required": ["inferred"],
        "additionalProperties": False,
    }
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"inferred":{"x":1}}'}],
            }
        ],
        "usage": {},
    }
    client = GenericLLMClient(provider="openai", model="gpt-5-mini", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat(
        [{"role": "user", "content": "extract"}],
        structured_output=StructuredOutputRequest("MetalensExtract", schema),
    )

    assert json.loads(text) == {"inferred": {"x": 1}}
    assert fake_http.last_json is not None
    response_schema = fake_http.last_json["text"]["format"]
    assert response_schema["type"] == "json_schema"
    assert response_schema["strict"] is False
    assert response_schema["schema"]["properties"]["inferred"]["additionalProperties"] is True


@pytest.mark.asyncio
async def test_openrouter_uses_the_prepared_native_schema_request() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    payload = {
        "choices": [{"message": {"content": '{"answer":"ok"}'}}],
        "usage": {},
    }
    sink = _ObservationSink()
    client = GenericLLMClient(
        provider="openrouter",
        model="openai/gpt-5-mini",
        api_key="test",
        observation_sink=sink,
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    await client.chat(
        [{"role": "user", "content": "answer"}],
        structured_output=StructuredOutputRequest("Answer", schema),
    )

    assert fake_http.last_json is not None
    actual = fake_http.last_json["response_format"]
    assert actual["type"] == "json_schema"
    assert actual["json_schema"]["strict"] is True
    assert sink.records[0].provider_request_args["response_format"] == actual


def test_native_required_rejects_deepseek_before_transport() -> None:
    with pytest.raises(LLMStructuredOutputCapabilityError, match="native_required"):
        prepare_structured_output(
            StructuredOutputRequest("Answer", {"type": "object"}),
            provider="deepseek",
            model="deepseek-v4-pro",
            policy="native_required",
        )


@pytest.mark.asyncio
async def test_deepseek_uses_json_object_guidance_and_canonical_validation() -> None:
    client = GenericLLMClient(provider="deepseek", model="deepseek-v4-pro")
    seen: dict[str, Any] = {}

    async def fake_chat_dispatch(messages, **kwargs):
        seen["messages"] = messages
        seen.update(kwargs)
        return '{"answer":7}', {}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(Exception, match="string"):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            structured_output=StructuredOutputRequest(
                "Answer",
                {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            ),
        )

    assert seen["output_format"] == "json_object"
    assert seen["json_schema"] is None
    assert "JSON Schema" in seen["messages"][0]["content"]


@pytest.mark.asyncio
async def test_profile_native_required_fails_before_provider_dispatch() -> None:
    client = GenericLLMClient(
        provider="deepseek",
        model="deepseek-v4-pro",
        structured_output_policy="native_required",
    )
    dispatched = False

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched = True
        return "{}", {}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(LLMStructuredOutputCapabilityError):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            structured_output=StructuredOutputRequest("Answer", {"type": "object"}),
        )

    assert not dispatched


@pytest.mark.asyncio
async def test_new_and_deprecated_structured_forms_normalize_identically() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    seen: list[dict[str, Any]] = []

    async def fake_chat_dispatch(messages, **kwargs):
        seen.append(dict(kwargs))
        return '{"answer":"ok"}', {}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    await client.chat(
        [{"role": "user", "content": "hello"}],
        structured_output=StructuredOutputRequest("Answer", schema),
    )
    with pytest.warns(DeprecationWarning, match="removed in 0.2.0"):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            output_format="json_schema",
            json_schema=schema,
            schema_name="Answer",
            strict_schema=True,
            validate_json=True,
        )

    normalized_keys = (
        "output_format",
        "json_schema",
        "schema_name",
        "strict_schema",
        "validate_json",
        "fail_on_unsupported",
    )
    assert {key: seen[0][key] for key in normalized_keys} == {
        key: seen[1][key] for key in normalized_keys
    }


@pytest.mark.asyncio
async def test_structured_output_rejects_mixed_deprecated_parameters() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")
    dispatched = False

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched = True
        return "{}", {}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="cannot be combined"):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            structured_output=StructuredOutputRequest("Answer", {"type": "object"}),
            json_schema={"type": "object"},
        )

    assert not dispatched


@pytest.mark.asyncio
async def test_deprecated_structured_parameters_are_observable() -> None:
    sink = _ObservationSink()
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        observation_sink=sink,
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return '{"answer":"ok"}', {}

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.warns(DeprecationWarning):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            output_format="json_schema",
            json_schema={"type": "object"},
            schema_name="Answer",
        )

    record = sink.records[0]
    assert record.request_args["deprecated_parameters"] == [
        "json_schema",
        "schema_name",
    ]
    assert any("0.2.0" in note for note in record.compatibility_notes)


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
async def test_anthropic_without_cache_control_keeps_classic_system_string() -> None:
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 3, "output_tokens": 4},
    }
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="anthropic-key",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, usage = await client._chat_anthropic_messages(  # type: ignore[misc]
        [
            {"role": "system", "content": "Stable rules."},
            {"role": "user", "content": "hello"},
        ],
        model="claude-test",
        output_format="text",
        json_schema=None,
        fail_on_unsupported=False,
    )

    assert text == "ok"
    assert usage["input_tokens"] == 3
    assert fake_http.last_json is not None
    assert fake_http.last_json["system"] == "Stable rules."
    assert "cache_control" not in fake_http.last_json
    assert "cache_control" not in fake_http.last_json["messages"][0]["content"][0]


@pytest.mark.asyncio
async def test_anthropic_cache_control_passes_through_system_and_messages() -> None:
    payload = {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 4,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 50,
        },
    }
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="anthropic-key",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, usage = await client._chat_anthropic_messages(  # type: ignore[misc]
        [
            {
                "role": "system",
                "content": "Frozen header.",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Frozen ledger.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": "Volatile tail.",
                "cache_control": {"type": "ephemeral"},
            },
        ],
        model="claude-test",
        output_format="text",
        json_schema=None,
        fail_on_unsupported=False,
        cache_control={"type": "ephemeral"},
    )

    assert text == "ok"
    assert usage["cache_read_input_tokens"] == 50
    assert fake_http.last_json is not None
    assert fake_http.last_json["cache_control"] == {"type": "ephemeral"}
    assert fake_http.last_json["system"] == [
        {
            "type": "text",
            "text": "Frozen header.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages = fake_http.last_json["messages"]
    assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_anthropic_cache_control_rejects_more_than_four_breakpoints() -> None:
    payload = {"content": [{"type": "text", "text": "ok"}], "usage": {}}
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="anthropic-key",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    messages = [
        {"role": "user", "content": f"block {index}", "cache_control": {"type": "ephemeral"}}
        for index in range(5)
    ]

    with pytest.raises(ValueError, match="at most 4"):
        await client._chat_anthropic_messages(  # type: ignore[misc]
            messages,
            model="claude-test",
            output_format="text",
            json_schema=None,
            fail_on_unsupported=False,
        )

    assert fake_http.last_json is None


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


def test_encode_llm_profile_env_includes_compatibility_policy() -> None:
    env = encode_llm_profile_env(
        "DEEPSEEK",
        LLMProfilePayload(
            provider="deepseek",
            model="deepseek-v4-pro",
            reasoning_effort="high",
            thinking_mode="auto",
            compatibility_policy="compat",
        ),
    )

    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__REASONING_EFFORT"] == "high"
    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__THINKING_MODE"] == "auto"
    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__COMPATIBILITY_POLICY"] == "compat"


def test_encode_llm_profile_env_includes_structured_output_policy() -> None:
    env = encode_llm_profile_env(
        "DEEPSEEK",
        LLMProfilePayload(
            provider="deepseek",
            model="deepseek-v4-pro",
            structured_output_policy="native_required",
        ),
    )

    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__STRUCTURED_OUTPUT_POLICY"] == "native_required"


def test_encode_llm_profile_env_includes_explicit_vision_fields() -> None:
    env = encode_llm_profile_env(
        "local_vision",
        LLMProfilePayload(
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
        ),
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

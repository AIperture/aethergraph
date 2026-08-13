from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aethergraph.api.v1 import settings as settings_api
from aethergraph.api.v1.schemas.settings import LLMProfilePayload
from aethergraph.config.llm import LLMProfile
from aethergraph.config.llm_env import encode_llm_profile_env
from aethergraph.services.llm import (
    LLMToolCallCapabilityError,
    LLMToolCallResponseError,
    ModelRequest,
    PromptCacheRequest,
    StructuredOutputRequest,
    ToolCallOutput,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    message_from_text,
)
from aethergraph.services.llm.adapters import OpenAICompatibleChatAdapter
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.provider_transport import (
    LLMProviderRequestError,
    ProviderCallResult,
)
from aethergraph.services.llm.service import LLMService
from aethergraph.services.llm.structured_output import (
    prepare_structured_output,
    resolve_structured_output_capabilities,
)
from aethergraph.services.llm.types import (
    LLMStructuredOutputCapabilityError,
    LLMStructuredOutputParseError,
    LLMStructuredOutputRefusalError,
    LLMStructuredOutputTruncationError,
    LLMStructuredOutputValidationError,
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
        self.last_url: str | None = None

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any], timeout=None):
        self.last_url = url
        self.last_json = json
        return _FakeResponse(self.payload)


class _FakeStreamResponse:
    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.is_error = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _FakeStreamingHttpClient:
    def __init__(self, lines: list[str]):
        self.lines = list(lines)
        self.last_json: dict[str, Any] | None = None
        self.last_url: str | None = None

    def stream(self, method: str, url: str, headers: dict[str, str], json: dict[str, Any]):
        self.last_url = url
        self.last_json = json
        return _FakeStreamResponse(self.lines)

    async def post(self, *args: Any, **kwargs: Any):
        raise AssertionError("native streaming adapter issued a non-streaming request")


class _ObservationSink:
    def __init__(self) -> None:
        self.records = []

    async def emit(self, record, *, capture_mode) -> None:
        self.records.append(record)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "expected_key"),
    [
        ("openai", "openai-key"),
        ("azure", "azure-key"),
        ("anthropic", "anthropic-key"),
        ("google", "google-key"),
        ("deepseek", "deepseek-key"),
        ("openrouter", "openrouter-key"),
        ("openai_compatible", "compatible-key"),
    ],
)
async def test_generic_client_uses_only_exact_provider_environment_credential(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    expected_key: str,
) -> None:
    values = {
        "OPENAI_API_KEY": "openai-key",
        "AZURE_OPENAI_KEY": "azure-key",
        "ANTHROPIC_API_KEY": "anthropic-key",
        "GOOGLE_API_KEY": "google-key",
        "DEEPSEEK_API_KEY": "deepseek-key",
        "OPENROUTER_API_KEY": "openrouter-key",
        "OPENAI_COMPATIBLE_API_KEY": "compatible-key",
        "AZURE_OPENAI_ENDPOINT": "https://azure.example",
        "OPENAI_COMPATIBLE_BASE_URL": "http://localhost:9000/v1",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    client = GenericLLMClient(provider=provider, model="test-model")
    try:
        assert client.api_key == expected_key
    finally:
        await client.aclose()


def _native_tool_request(*, max_calls: int = 2) -> ToolCallRequest:
    return ToolCallRequest(
        tools=(
            ToolDefinition(
                name="lookup",
                description="Look up one record.",
                input_schema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
            ToolDefinition(
                name="finish",
                description="Finish the task.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
        max_calls=max_calls,
    )


@pytest.mark.asyncio
async def test_generate_wraps_direct_assistant_output_with_typed_usage() -> None:
    payload = {
        "id": "response_1",
        "status": "completed",
        "output": [
            {
                "id": "message_1",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello."}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response = await client.generate(ModelRequest(messages=(message_from_text("user", "Hello"),)))

    assert response.text == "Hello."
    assert response.calls == ()
    assert response.assistant_outputs[0].output_id.startswith("assistant_output_")
    assert response.usage.availability == "complete"
    assert response.usage.total_input_tokens == 4
    assert response.usage.output_tokens == 2
    assert fake_http.last_json is not None
    assert fake_http.last_json["input"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_generate_does_not_dispatch_through_public_chat_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "id": "response_1",
        "status": "completed",
        "output": [
            {
                "id": "message_1",
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello."}],
            }
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    async def reject_public_chat(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("generate must not dispatch through the public chat facade")

    monkeypatch.setattr(client, "chat", reject_public_chat)

    response = await client.generate(ModelRequest(messages=(message_from_text("user", "Hello"),)))

    assert response.text == "Hello."


@pytest.mark.asyncio
async def test_generate_projects_canonical_tools_to_existing_provider_path() -> None:
    payload = {
        "id": "response_1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "function_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"key":"A"}',
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    legacy = _native_tool_request(max_calls=1)

    response = await client.generate(
        ModelRequest(
            messages=(message_from_text("user", "Look up A"),),
            tools=legacy.tools,
            tool_choice="required",
        )
    )

    assert response.calls[0].call_id == "call_1"
    assert response.calls[0].arguments == {"key": "A"}
    assert response.usage.availability == "complete"
    assert fake_http.last_json is not None
    assert fake_http.last_json["tool_choice"] == "required"


def test_native_tool_definition_rejects_provider_unsafe_name() -> None:
    with pytest.raises(ValueError, match="letters, numbers, underscores, or hyphens"):
        ToolDefinition(
            name="workspace.read_text",
            description="Read one file.",
            input_schema={"type": "object", "properties": {}},
        )


@pytest.mark.asyncio
async def test_openai_native_tool_items_preserve_multiple_call_boundaries() -> None:
    payload = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"key":"A"}',
                "status": "completed",
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "finish",
                "arguments": "{}",
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    client = GenericLLMClient(provider="openai", model="gpt-5.6-test", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, usage = await client.chat(
        [
            {"role": "system", "content": "stable instructions"},
            {"role": "user", "content": "look up and finish"},
            {"role": "assistant", "content": "prior Tool selection"},
            {"role": "user", "content": "prior Tool result"},
        ],
        tool_request=_native_tool_request(max_calls=2),
        prompt_cache=PromptCacheRequest((0, 2), "agent.native-tools.v1"),
    )

    assert isinstance(response, ToolCallResponse)
    assert [call.call_id for call in response.calls] == ["call_1", "call_2"]
    assert [call.name for call in response.calls] == ["lookup", "finish"]
    assert response.calls[0].arguments == {"key": "A"}
    assert response.usage.availability == "complete"
    assert response.usage.total_input_tokens == 10
    assert response.usage.output_tokens == 5
    assert usage["input_tokens"] == 10
    assert fake_http.last_json is not None
    assert fake_http.last_json["parallel_tool_calls"] is True
    assert fake_http.last_json["tool_choice"] == "required"
    assert fake_http.last_json["input"][2]["content"] == [
        {
            "type": "output_text",
            "text": "prior Tool selection",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    assert [tool["name"] for tool in fake_http.last_json["tools"]] == [
        "lookup",
        "finish",
    ]


@pytest.mark.asyncio
async def test_anthropic_native_tool_blocks_preserve_multiple_call_boundaries() -> None:
    payload = {
        "id": "msg_1",
        "stop_reason": "tool_use",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {"key": "A"},
            },
            {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "finish",
                "input": {},
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, _usage = await client.chat(
        [{"role": "user", "content": "look up and finish"}],
        tool_request=_native_tool_request(max_calls=2),
    )

    assert isinstance(response, ToolCallResponse)
    assert [call.call_id for call in response.calls] == ["toolu_1", "toolu_2"]
    assert [call.name for call in response.calls] == ["lookup", "finish"]
    assert fake_http.last_json is not None
    assert fake_http.last_json["tool_choice"] == {
        "type": "any",
        "disable_parallel_tool_use": False,
    }
    assert [tool["name"] for tool in fake_http.last_json["tools"]] == [
        "lookup",
        "finish",
    ]


@pytest.mark.asyncio
async def test_anthropic_does_not_silently_weaken_required_tool_choice() -> None:
    client = GenericLLMClient(
        provider="anthropic",
        model="claude-test",
        api_key="test",
    )
    fake_http = _FakeHttpClient({})
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    with pytest.raises(
        LLMToolCallCapabilityError,
        match="required_tool_choice_with_thinking",
    ):
        await client.chat(
            [{"role": "user", "content": "look up"}],
            tool_request=_native_tool_request(max_calls=1),
            thinking_mode="on",
        )

    assert fake_http.last_json is None


@pytest.mark.asyncio
async def test_gemini_native_function_parts_preserve_multiple_call_boundaries() -> None:
    payload = {
        "candidates": [
            {
                "index": 0,
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "id": "gemini_1",
                                "name": "lookup",
                                "args": {"key": "A"},
                            },
                            "thoughtSignature": "opaque",
                        },
                        {
                            "functionCall": {
                                "id": "gemini_2",
                                "name": "finish",
                                "args": {},
                            }
                        },
                    ]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5},
    }
    client = GenericLLMClient(provider="google", model="gemini-test", api_key="test")
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, _usage = await client.chat(
        [{"role": "user", "content": "look up and finish"}],
        tool_request=_native_tool_request(max_calls=2),
    )

    assert isinstance(response, ToolCallResponse)
    assert [call.call_id for call in response.calls] == ["gemini_1", "gemini_2"]
    assert response.calls[0].provider_metadata["thought_signature"] == "opaque"
    assert fake_http.last_json is not None
    function_config = fake_http.last_json["toolConfig"]["functionCallingConfig"]
    assert function_config["mode"] == "ANY"
    assert function_config["allowedFunctionNames"] == ["lookup", "finish"]


@pytest.mark.asyncio
async def test_gemini_profile_thinking_mode_is_projected_once() -> None:
    client = GenericLLMClient(
        provider="google",
        model="gemini-2.5-flash",
        api_key="google-key",
        thinking_mode="off",
    )
    fake_http = _FakeHttpClient(
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "done"}]},
                }
            ]
        }
    )
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    text, _usage = await client.chat([{"role": "user", "content": "Hello"}])

    assert text == "done"
    assert fake_http.last_json is not None
    assert fake_http.last_json["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


def test_structured_output_request_detaches_caller_schema() -> None:
    schema = {"type": "object", "properties": {}}

    request = StructuredOutputRequest(name=" Answer ", schema=schema)
    schema["properties"]["late"] = {"type": "string"}

    assert request.name == "Answer"
    assert request.schema == {"type": "object", "properties": {}}
    assert request.validation_owner == "aethergraph"


def test_structured_output_request_rejects_unknown_validation_owner() -> None:
    with pytest.raises(ValueError, match="validation_owner"):
        StructuredOutputRequest(
            name="Answer",
            schema={"type": "object"},
            validation_owner="engine",  # type: ignore[arg-type]
        )


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
    assert len(prepared.canonical_schema_fingerprint) == 64
    assert prepared.provider_schema_fingerprint == prepared.canonical_schema_fingerprint


def test_structured_output_resolution_honors_preselected_endpoint() -> None:
    capabilities = resolve_structured_output_capabilities(
        "openai",
        "gpt-5-mini",
        endpoint_id="openai_chat_completions",
    )

    assert capabilities.native_schema is True
    assert capabilities.native_strict_schema is True


def test_structured_output_resolution_rejects_cross_provider_endpoint() -> None:
    with pytest.raises(ValueError, match="not registered"):
        resolve_structured_output_capabilities(
            "anthropic",
            "claude-sonnet-4-5",
            endpoint_id="openai_responses",
        )


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
    request_args = sink.records[0].request_args
    assert request_args["structured_output_policy"] == "best_available"
    assert request_args["structured_output_effective_mode"] == "native_strict"
    assert request_args["structured_output_capability_source"].startswith("ag_static/")
    assert len(request_args["structured_output_canonical_schema_fingerprint"]) == 64
    assert (
        request_args["structured_output_provider_schema_fingerprint"]
        == request_args["structured_output_canonical_schema_fingerprint"]
    )
    assert request_args["structured_output_validation_outcome"] == "passed"
    assert request_args["structured_output_response_state"] == "completed"


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
        return ProviderCallResult(('{"answer":7}', {}))

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
async def test_local_schema_failure_retains_provider_response_usage_and_exact_issue() -> None:
    class _Metering:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        async def record_llm(self, **record: Any) -> None:
            self.records.append(record)

    sink = _ObservationSink()
    metering = _Metering()
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        observation_sink=sink,
        metering=metering,
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return ProviderCallResult(('{"answer":7}', {"prompt_tokens": 11, "completion_tokens": 3}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(LLMStructuredOutputValidationError) as exc_info:
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

    error = exc_info.value
    assert error.code == "schema_invalid"
    assert error.path == "$.answer"
    assert error.validator == "type"
    assert error.invalid_value == "7"
    assert error.expected == ("string",)
    assert len(error.canonical_schema_fingerprint) == 64

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.raw_text == '{"answer":7}'
    assert record.usage == {"prompt_tokens": 11, "completion_tokens": 3}
    assert record.request_args["structured_output_response_state"] == "schema_invalid"
    assert record.request_args["structured_output_validation_outcome"] == "failed"
    assert record.request_args["structured_output_error"]["path"] == "$.answer"
    assert len(metering.records) == 1
    assert metering.records[0]["prompt_tokens"] == 11
    assert metering.records[0]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_caller_owned_validation_returns_unvalidated_provider_response() -> None:
    sink = _ObservationSink()
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        observation_sink=sink,
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return ProviderCallResult(('{"calls":[1,2]}', {"prompt_tokens": 5, "completion_tokens": 3}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]
    text, usage = await client.chat(
        [{"role": "user", "content": "select"}],
        structured_output=StructuredOutputRequest(
            "Action",
            {
                "type": "object",
                "properties": {
                    "calls": {
                        "type": "array",
                        "maxItems": 1,
                    }
                },
                "required": ["calls"],
                "additionalProperties": False,
            },
            validation_owner="caller",
        ),
    )

    assert text == '{"calls":[1,2]}'
    assert usage == {"prompt_tokens": 5, "completion_tokens": 3}
    request_args = sink.records[0].request_args
    assert request_args["structured_output_validation_owner"] == "caller"
    assert request_args["structured_output_validation_outcome"] == "delegated"
    assert request_args["structured_output_response_state"] == "returned_unvalidated"


@pytest.mark.asyncio
async def test_local_json_parse_failure_retains_raw_response_without_copying_it_to_error() -> None:
    sink = _ObservationSink()
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        observation_sink=sink,
    )

    async def fake_chat_dispatch(messages, **kwargs):
        return ProviderCallResult(('{"answer":', {"prompt_tokens": 5, "completion_tokens": 1}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(LLMStructuredOutputParseError) as exc_info:
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

    error = exc_info.value
    assert error.code == "invalid_json"
    assert error.response_state == "invalid_json"
    assert '{"answer":' not in str(error)
    assert sink.records[0].raw_text == '{"answer":'
    assert sink.records[0].usage == {"prompt_tokens": 5, "completion_tokens": 1}


@pytest.mark.asyncio
async def test_profile_native_required_fails_before_provider_dispatch() -> None:
    sink = _ObservationSink()
    client = GenericLLMClient(
        provider="deepseek",
        model="deepseek-v4-pro",
        structured_output_policy="native_required",
        observation_sink=sink,
    )
    dispatched = False

    async def fake_chat_dispatch(messages, **kwargs):
        nonlocal dispatched
        dispatched = True
        return ProviderCallResult(("{}", {}))

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(LLMStructuredOutputCapabilityError):
        await client.chat(
            [{"role": "user", "content": "hello"}],
            structured_output=StructuredOutputRequest("Answer", {"type": "object"}),
        )

    assert not dispatched
    assert len(sink.records) == 1
    request_args = sink.records[0].request_args
    assert request_args["structured_output_effective_mode"] == "unavailable"
    assert request_args["structured_output_validation_outcome"] == "not_run"
    assert request_args["structured_output_response_state"] == ("capability_rejected")
    assert request_args["structured_output_capability_source"].startswith("ag_static/")
    assert len(request_args["structured_output_canonical_schema_fingerprint"]) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "validation_outcome", "response_state"),
    [
        (
            LLMProviderRequestError(
                provider="openai",
                model="gpt-5-mini",
                operation="chat",
                code="provider_request_rejected",
                message="rejected",
                retryable=False,
            ),
            "not_run",
            "provider_request_rejected",
        ),
        (
            LLMStructuredOutputRefusalError("refused"),
            "not_run",
            "refused",
        ),
        (
            LLMStructuredOutputTruncationError("truncated"),
            "not_run",
            "truncated",
        ),
        (
            LLMStructuredOutputValidationError(
                code="schema_invalid",
                summary="invalid",
                path="$.answer",
                validator="type",
                response_state="schema_invalid",
            ),
            "failed",
            "schema_invalid",
        ),
    ],
)
async def test_structured_output_failures_record_generic_state(
    error: Exception,
    validation_outcome: str,
    response_state: str,
) -> None:
    sink = _ObservationSink()
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        observation_sink=sink,
    )

    async def fake_chat_dispatch(messages, **kwargs):
        raise error

    client._chat_dispatch = fake_chat_dispatch  # type: ignore[method-assign]

    with pytest.raises(type(error)):
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

    assert len(sink.records) == 1
    request_args = sink.records[0].request_args
    assert request_args["structured_output_validation_outcome"] == validation_outcome
    assert request_args["structured_output_response_state"] == response_state


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
        return ProviderCallResult(('{"answer":"ok"}', {}))

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
        return ProviderCallResult(("{}", {}))

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
        return ProviderCallResult(('{"answer":"ok"}', {}))

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
        return ProviderCallResult(('{"b":2,"a":1}', {"prompt_tokens": 1, "completion_tokens": 1}))

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
        return ProviderCallResult(("ok", {}))

    async def fake_dispatch_compat(messages, **kwargs):
        compat_seen.update(kwargs)
        return ProviderCallResult(("ok", {}))

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
async def test_explicit_openai_chat_completions_stream_never_switches_to_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        endpoint_id="openai_chat_completions",
    )
    calls: list[str] = []

    async def fake_chat_completions_stream(host, messages, **kwargs):
        calls.append("chat.completions")
        return ProviderCallResult(("streamed", {}))

    async def fail_responses_stream(messages, **kwargs):
        raise AssertionError("pinned Chat Completions endpoint switched to Responses")

    monkeypatch.setattr(
        OpenAICompatibleChatAdapter,
        "stream",
        fake_chat_completions_stream,
    )
    client._chat_openai_responses_stream = fail_responses_stream  # type: ignore[method-assign]

    text, usage = await client.chat_stream([{"role": "user", "content": "hello"}])

    assert text == "streamed"
    assert usage == {}
    assert calls == ["chat.completions"]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint_id", [None, "azure_chat_completions"])
async def test_azure_chat_completions_stream_uses_native_sse_and_terminal_usage(
    endpoint_id: str | None,
) -> None:
    client = GenericLLMClient(
        provider="azure",
        model="deployment-a",
        endpoint_id=endpoint_id,
        base_url="https://example.openai.azure.com",
        azure_deployment="deployment-a",
        api_key="test",
    )
    fake_http = _FakeStreamingHttpClient(
        [
            'data: {"choices":[{"delta":{"content":"Hel"}}],"usage":null}',
            'data: {"choices":[{"delta":{"content":"lo"}}],"usage":null}',
            (
                'data: {"choices":[],"usage":'
                '{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}'
            ),
            "data: [DONE]",
        ]
    )
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    deltas: list[str] = []
    usage_updates: list[dict[str, int]] = []

    async def on_delta(delta: str) -> None:
        deltas.append(delta)

    async def on_usage_update(usage: dict[str, int]) -> None:
        usage_updates.append(usage)

    text, usage = await client.chat_stream(
        [{"role": "user", "content": "Hello"}],
        on_delta=on_delta,
        on_usage_update=on_usage_update,
    )

    assert text == "Hello"
    assert deltas == ["Hel", "lo"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2
    assert usage_updates == [{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}]
    assert fake_http.last_url is not None
    assert "/chat/completions?" in fake_http.last_url
    assert fake_http.last_json is not None
    assert fake_http.last_json["stream"] is True
    assert fake_http.last_json["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_gemini_stream_uses_native_sse_with_thoughts_and_usage() -> None:
    client = GenericLLMClient(provider="google", model="gemini-test", api_key="test")
    fake_http = _FakeStreamingHttpClient(
        [
            ('data: {"candidates":[{"content":{"parts":[{"text":"Plan","thought":true}]}}]}'),
            'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}',
            (
                'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}],'
                '"usageMetadata":{"promptTokenCount":4,"candidatesTokenCount":2,'
                '"thoughtsTokenCount":3}}'
            ),
        ]
    )
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    deltas: list[str] = []
    thoughts: list[str] = []
    usage_updates: list[dict[str, int]] = []

    async def on_delta(delta: str) -> None:
        deltas.append(delta)

    async def on_thinking_delta(delta: str) -> None:
        thoughts.append(delta)

    async def on_usage_update(usage: dict[str, int]) -> None:
        usage_updates.append(usage)

    text, usage = await client.chat_stream(
        [{"role": "user", "content": "Hello"}],
        reasoning_summary="auto",
        on_delta=on_delta,
        on_thinking_delta=on_thinking_delta,
        on_usage_update=on_usage_update,
    )

    assert text == "Hello"
    assert deltas == ["Hel", "lo"]
    assert thoughts == ["Plan"]
    assert usage == {"input_tokens": 4, "output_tokens": 2, "reasoning_tokens": 3}
    assert usage_updates == [usage]
    assert fake_http.last_url is not None
    assert ":streamGenerateContent?alt=sse&key=" in fake_http.last_url
    assert fake_http.last_json is not None
    assert fake_http.last_json["generationConfig"]["thinkingConfig"]["includeThoughts"] is True


@pytest.mark.asyncio
async def test_azure_responses_stream_fails_before_transport() -> None:
    client = GenericLLMClient(
        provider="azure",
        model="deployment-a",
        endpoint_id="azure_responses",
        base_url="https://example.openai.azure.com",
        azure_deployment="deployment-a",
        api_key="test",
    )
    fake_http = _FakeStreamingHttpClient([])
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    with pytest.raises(LLMUnsupportedFeatureError, match="no streaming implementation"):
        async for _event in client.generate_stream(
            ModelRequest(messages=(message_from_text("user", "Hello"),))
        ):
            pass

    assert fake_http.last_url is None


@pytest.mark.asyncio
async def test_legacy_chat_stream_rejects_unimplemented_pinned_adapter_before_transport() -> None:
    client = GenericLLMClient(
        provider="azure",
        model="deployment-a",
        endpoint_id="azure_responses",
        base_url="https://example.openai.azure.com",
        azure_deployment="deployment-a",
        api_key="test",
    )
    fake_http = _FakeStreamingHttpClient([])
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    with pytest.raises(LLMUnsupportedFeatureError, match="no streaming implementation"):
        await client.chat_stream([{"role": "user", "content": "Hello"}])

    assert fake_http.last_url is None


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

    result = await OpenAICompatibleChatAdapter.invoke(
        client,
        [{"role": "user", "content": "hello"}],
        model="deepseek-v4-pro",
        reasoning_effort="xhigh",
        max_output_tokens=256,
        output_format="json_object",
        json_schema=None,
        fail_on_unsupported=False,
    )
    text, usage = result.value

    assert json.loads(text) == {"answer": "ok"}
    assert usage["completion_tokens"] == 4
    assert fake_http.last_json is not None
    assert fake_http.last_json["response_format"] == {"type": "json_object"}
    assert fake_http.last_json["max_tokens"] == 256
    assert fake_http.last_json["reasoning_effort"] == "max"
    assert fake_http.last_json["thinking"] == {"type": "enabled"}


def test_openai_compatible_chat_is_not_inherited_by_generic_client() -> None:
    assert not hasattr(GenericLLMClient, "_chat_openai_like_chat_completions")
    assert not hasattr(GenericLLMClient, "_chat_openai_like_chat_completions_stream")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("deepseek", "deepseek-chat"),
        ("openrouter", "openai/gpt-test"),
        ("lmstudio", "local-model"),
        ("ollama", "local-model"),
        ("openai_compatible", "custom-model"),
        ("azure", "deployment-a"),
    ],
)
async def test_openai_compatible_generate_normalizes_native_tool_calls(
    provider: str,
    model: str,
) -> None:
    payload = {
        "id": "chatcmpl_1",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "I will look that up.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"key":"A"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "finish", "arguments": "{}"},
                        },
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 3},
    }
    client_kwargs: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "api_key": "test",
    }
    if provider == "openai_compatible":
        client_kwargs["base_url"] = "http://localhost:9000/v1"
    elif provider == "azure":
        client_kwargs.update(
            endpoint_id="azure_chat_completions",
            base_url="https://example.openai.azure.com",
            azure_deployment=model,
        )
    client = GenericLLMClient(**client_kwargs)
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    tool_request = _native_tool_request(max_calls=2)

    response = await client.generate(
        ModelRequest(
            messages=(message_from_text("user", "Look up A and finish"),),
            tools=tool_request.tools,
            tool_choice="auto",
            max_tool_calls=2,
        )
    )

    assert response.text == "I will look that up."
    assert [call.call_id for call in response.calls] == ["call_1", "call_2"]
    assert response.calls[0].arguments == {"key": "A"}
    assert response.usage.total_input_tokens == 8
    assert fake_http.last_json is not None
    assert fake_http.last_json["tool_choice"] == "auto"
    assert fake_http.last_json["parallel_tool_calls"] is True
    assert [item["function"]["name"] for item in fake_http.last_json["tools"]] == [
        "lookup",
        "finish",
    ]


@pytest.mark.asyncio
async def test_openai_compatible_tool_request_accepts_direct_assistant_response() -> None:
    payload = {
        "id": "chatcmpl_direct",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "No Tool is needed."},
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    client = GenericLLMClient(
        provider="openrouter",
        model="openai/gpt-test",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    tool_request = _native_tool_request(max_calls=1)

    response = await client.generate(
        ModelRequest(
            messages=(message_from_text("user", "Say hello or use a Tool"),),
            tools=tool_request.tools,
            tool_choice="auto",
        )
    )

    assert response.text == "No Tool is needed."
    assert response.calls == ()
    assert response.finish_reason == "stop"
    assert response.usage.output_tokens == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("deepseek", "deepseek-chat"),
        ("openrouter", "openai/gpt-test"),
        ("lmstudio", "local-model"),
        ("ollama", "local-model"),
        ("openai_compatible", "custom-model"),
    ],
)
async def test_openai_compatible_tool_results_continue_with_integrity_bound_replay(
    provider: str,
    model: str,
) -> None:
    initial_payload = {
        "id": "chatcmpl_initial",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"key":"A"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2},
    }
    final_payload = {
        "id": "chatcmpl_final",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A is complete."},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    client = GenericLLMClient(
        provider=provider,
        model=model,
        api_key="test",
        base_url=("http://localhost:9000/v1" if provider == "openai_compatible" else None),
    )
    fake_http = _FakeHttpClient(initial_payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    tools = _native_tool_request(max_calls=1).tools
    messages = (message_from_text("user", "Look up A"),)

    initial = await client.generate(
        ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            turn_id="turn-1",
        )
    )
    assert initial.continuation is not None
    assert initial.continuation.provider == provider
    assert initial.continuation.revision == 1

    fake_http.payload = final_payload
    final = await client.generate(
        ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            turn_id="turn-1",
            continuation=initial.continuation,
            tool_outputs=(ToolCallOutput("call_1", '{"value":"A"}'),),
        )
    )

    assert final.text == "A is complete."
    assert final.calls == ()
    assert final.continuation is None
    assert fake_http.last_json is not None
    assert [message["role"] for message in fake_http.last_json["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert fake_http.last_json["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"value":"A"}',
    }


@pytest.mark.asyncio
async def test_openai_compatible_continuation_requires_exact_pending_outputs() -> None:
    payload = {
        "id": "chatcmpl_initial",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"A"}'},
                        }
                    ],
                },
            }
        ],
    }
    client = GenericLLMClient(
        provider="openrouter",
        model="openai/gpt-test",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    tools = _native_tool_request(max_calls=1).tools
    messages = (message_from_text("user", "Look up A"),)
    initial = await client.generate(
        ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            turn_id="turn-1",
        )
    )
    assert initial.continuation is not None
    first_payload = fake_http.last_json

    with pytest.raises(LLMToolCallResponseError, match="exactly every pending Tool output"):
        await client.generate(
            ModelRequest(
                messages=messages,
                tools=tools,
                tool_choice="auto",
                turn_id="turn-1",
                continuation=initial.continuation,
            )
        )

    assert fake_http.last_json is first_payload

    with pytest.raises(ValueError, match="checkpoint prompt changed"):
        await client.generate(
            ModelRequest(
                messages=(message_from_text("user", "Look up B"),),
                tools=tools,
                tool_choice="auto",
                turn_id="turn-1",
                continuation=initial.continuation,
                tool_outputs=(ToolCallOutput("call_1", '{"value":"A"}'),),
            )
        )

    assert fake_http.last_json is first_payload


@pytest.mark.asyncio
async def test_openai_compatible_continuation_advances_multi_round_replay() -> None:
    first_payload = {
        "id": "chatcmpl_first",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"A"}'},
                        }
                    ],
                },
            }
        ],
    }
    second_payload = {
        "id": "chatcmpl_second",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Checking B.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"B"}'},
                        }
                    ],
                },
            }
        ],
    }
    client = GenericLLMClient(
        provider="openrouter",
        model="openai/gpt-test",
        api_key="test",
    )
    fake_http = _FakeHttpClient(first_payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    tools = _native_tool_request(max_calls=1).tools
    messages = (message_from_text("user", "Look up A, then B"),)

    first = await client.generate(
        ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            turn_id="turn-1",
        )
    )
    assert first.continuation is not None
    fake_http.payload = second_payload

    second = await client.generate(
        ModelRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            turn_id="turn-1",
            continuation=first.continuation,
            tool_outputs=(ToolCallOutput("call_1", '{"value":"A"}'),),
        )
    )

    assert second.continuation is not None
    assert second.continuation.revision == 2
    assert second.calls[0].call_id == "call_2"
    assert fake_http.last_json is not None
    assert [message["role"] for message in fake_http.last_json["messages"]] == [
        "user",
        "assistant",
        "tool",
    ]
    replay = second.continuation.opaque_payload["replay_messages"]
    assert [message["role"] for message in replay] == ["assistant", "tool", "assistant"]
    assert replay[-1]["tool_calls"][0]["id"] == "call_2"


@pytest.mark.asyncio
async def test_openai_compatible_tool_response_rejects_unknown_tool() -> None:
    payload = {
        "id": "chatcmpl_unknown",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "delete_everything", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }
    client = GenericLLMClient(
        provider="openrouter",
        model="openai/gpt-test",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    with pytest.raises(LLMToolCallResponseError, match="unknown Tool"):
        await client.chat(
            [{"role": "user", "content": "Use a Tool"}],
            tool_request=_native_tool_request(max_calls=1),
        )


@pytest.mark.asyncio
async def test_explicit_openai_chat_completions_endpoint_never_switches_to_responses() -> None:
    payload = {
        "id": "chatcmpl_explicit",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"A"}'},
                        }
                    ],
                },
            }
        ],
    }
    client = GenericLLMClient(
        provider="openai",
        model="gpt-test",
        endpoint_id="openai_chat_completions",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, _usage = await client.chat(
        [{"role": "user", "content": "Look up A"}],
        tool_request=_native_tool_request(max_calls=1),
    )

    assert isinstance(response, ToolCallResponse)
    assert response.calls[0].name == "lookup"
    assert fake_http.last_url == "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
async def test_endpointless_azure_preserves_direct_and_tool_adapter_selection() -> None:
    client = GenericLLMClient(
        provider="azure",
        model="deployment-a",
        base_url="https://example.openai.azure.com",
        azure_deployment="deployment-a",
        api_key="test",
    )
    calls: list[str] = []

    async def fake_chat_completions(messages, **kwargs):
        calls.append("azure_chat_completions")
        return ProviderCallResult(("direct", {}))

    async def fake_responses(messages, **kwargs):
        calls.append("azure_responses")
        return ProviderCallResult((ToolCallResponse(items=()), {}))

    client._chat_azure_chat_completions = fake_chat_completions  # type: ignore[method-assign]
    client._chat_azure_responses = fake_responses  # type: ignore[method-assign]

    direct, _usage = await client.chat([{"role": "user", "content": "Hello"}])
    tools, _usage = await client.chat(
        [{"role": "user", "content": "Look up A"}],
        tool_request=_native_tool_request(max_calls=1),
    )

    assert direct == "direct"
    assert isinstance(tools, ToolCallResponse)
    assert calls == ["azure_chat_completions", "azure_responses"]


@pytest.mark.asyncio
async def test_explicit_azure_chat_completions_normalizes_native_tool_calls() -> None:
    payload = {
        "id": "chatcmpl_azure",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [
                        {
                            "id": "call_azure_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"A"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    client = GenericLLMClient(
        provider="azure",
        model="deployment-a",
        endpoint_id="azure_chat_completions",
        base_url="https://example.openai.azure.com",
        azure_deployment="deployment-a",
        api_key="test",
    )
    fake_http = _FakeHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, _usage = await client.chat(
        [{"role": "user", "content": "Look up A"}],
        tool_request=_native_tool_request(max_calls=1),
        max_output_tokens=128,
    )

    assert isinstance(response, ToolCallResponse)
    assert response.text == "Checking."
    assert response.calls[0].call_id == "call_azure_1"
    assert fake_http.last_url is not None
    assert "/chat/completions?" in fake_http.last_url
    assert fake_http.last_json is not None
    assert fake_http.last_json["max_tokens"] == 128


@pytest.mark.asyncio
async def test_explicit_azure_responses_rejects_direct_chat_without_switching() -> None:
    client = GenericLLMClient(
        provider="azure",
        model="deployment-a",
        endpoint_id="azure_responses",
        base_url="https://example.openai.azure.com",
        azure_deployment="deployment-a",
        api_key="test",
    )
    fake_http = _FakeHttpClient({})
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    with pytest.raises(LLMUnsupportedFeatureError, match="pinned Azure Responses"):
        await client.chat([{"role": "user", "content": "Hello"}])

    assert fake_http.last_url is None


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

    result = await client._chat_anthropic_messages(  # type: ignore[misc]
        [
            {"role": "system", "content": "Stable rules."},
            {"role": "user", "content": "hello"},
        ],
        model="claude-test",
        output_format="text",
        json_schema=None,
        fail_on_unsupported=False,
    )
    text, usage = result.value

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

    result = await client._chat_anthropic_messages(  # type: ignore[misc]
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
    text, usage = result.value

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


def test_encode_llm_profile_env_includes_prompt_cache_policy() -> None:
    env = encode_llm_profile_env(
        "DEEPSEEK",
        LLMProfilePayload(
            provider="deepseek",
            model="deepseek-v4-pro",
            prompt_cache_policy="required",
        ),
    )

    assert env["AETHERGRAPH_LLM__PROFILES__DEEPSEEK__PROMPT_CACHE_POLICY"] == "required"


def test_encode_llm_profile_env_includes_explicit_endpoint() -> None:
    env = encode_llm_profile_env(
        "AZURE_TOOLS",
        LLMProfile(
            provider="azure",
            model="deployment-a",
            endpoint_id="azure_responses",
        ),
    )

    assert env["AETHERGRAPH_LLM__PROFILES__AZURE_TOOLS__ENDPOINT_ID"] == "azure_responses"


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


def test_llm_service_configure_profile_updates_runtime_metadata() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test")
    original_http_client = client._client
    original_retry = client._provider_retry
    service = LLMService(
        clients={"default": client},
        profiles={"default": LLMProfile(provider="openai", model="gpt-test")},
    )

    service.configure_profile(
        profile="default",
        structured_output_policy="native_required",
        prompt_cache_policy="disabled",
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
    assert profile.structured_output_policy == "native_required"
    assert client.structured_output_policy == "native_required"
    assert profile.prompt_cache_policy == "disabled"
    assert client.prompt_cache_policy == "disabled"
    assert client._client is original_http_client
    assert client._provider_retry is original_retry
    assert profile.vision_enabled is True
    assert profile.vision_max_images == 1
    assert profile.vision_max_image_bytes == 1024
    assert profile.vision_resize_enabled is False
    assert profile.vision_resize_max_dimension == 768
    assert profile.vision_resize_max_pixels == 400_000
    assert profile.vision_resize_jpeg_quality == 78
    assert profile.vision_resize_min_jpeg_quality == 62
    assert profile.vision_accepted_mime_types == ["image/png"]


def test_settings_profile_view_includes_structured_output_policy() -> None:
    view = settings_api._llm_profile_view(
        LLMProfile(
            provider="openai",
            model="gpt-5-mini",
            structured_output_policy="native_required",
        )
    )

    assert view.structured_output_policy == "native_required"


def test_settings_profile_view_includes_prompt_cache_policy() -> None:
    view = settings_api._llm_profile_view(
        LLMProfile(
            provider="openai",
            model="gpt-5-mini",
            prompt_cache_policy="required",
        )
    )

    assert view.prompt_cache_policy == "required"


def test_settings_profile_view_includes_explicit_endpoint() -> None:
    view = settings_api._llm_profile_view(
        LLMProfile(
            provider="openai",
            model="gpt-5-mini",
            endpoint_id="openai_responses",
        )
    )

    assert view.endpoint_id == "openai_responses"


def test_settings_hot_reload_rebinds_endpoint_without_replacing_client_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        endpoint_id="openai_responses",
    )
    service = LLMService(
        clients={"default": client},
        profiles={
            "default": LLMProfile(
                provider="openai",
                model="gpt-5-mini",
                endpoint_id="openai_responses",
            )
        },
    )
    original_http_client = client._client
    original_retry = client._provider_retry
    original_rate_gate = original_retry.rate_gate
    client._tool_transport_checkpoints["stale"] = object()  # type: ignore[assignment]
    client._latest_tool_checkpoint_refs[("run", "turn", "tool")] = "stale"
    monkeypatch.setattr(
        settings_api,
        "current_services",
        lambda: type("Services", (), {"llm": service})(),
    )

    settings_api._hot_reload_llm(
        {"default": LLMProfilePayload(endpoint_id="openai_chat_completions")}
    )

    assert service.get("default") is client
    assert client.endpoint_id == "openai_chat_completions"
    assert (
        client._resolve_chat_adapter(has_tool_request=False).protocol_family == "chat.completions"
    )
    assert client._provider_retry is not original_retry
    assert client._provider_retry.rate_gate is original_rate_gate
    assert original_http_client in client._retired_http_clients
    assert client._tool_transport_checkpoints == {}
    assert client._latest_tool_checkpoint_refs == {}
    profile = service.profile("default")
    assert profile is not None
    assert profile.endpoint_id == "openai_chat_completions"


def test_invalid_endpoint_hot_reload_leaves_runtime_and_profile_unchanged() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        endpoint_id="openai_responses",
    )
    original_profile = LLMProfile(
        provider="openai",
        model="gpt-5-mini",
        endpoint_id="openai_responses",
    )
    service = LLMService(
        clients={"default": client},
        profiles={"default": original_profile},
    )
    original_http_client = client._client
    original_retry = client._provider_retry

    with pytest.raises(ValueError, match="not registered"):
        service.configure_profile(
            profile="default",
            endpoint_id="azure_responses",
        )

    assert client.endpoint_id == "openai_responses"
    assert client._client is original_http_client
    assert client._provider_retry is original_retry
    assert service.profile("default") is original_profile


@pytest.mark.asyncio
async def test_connection_reconfiguration_closes_active_and_retired_transports() -> None:
    client = GenericLLMClient(
        provider="openai",
        model="gpt-5-mini",
        endpoint_id="openai_responses",
    )
    retired = client._client
    client.reconfigure_connection(
        provider="openai",
        model="gpt-5-mini",
        endpoint_id="openai_chat_completions",
        base_url=None,
        api_key=None,
        azure_deployment=None,
        timeout=60.0,
    )
    active = client._client

    await client.aclose()

    assert retired is not None and retired.is_closed
    assert active is not None and active.is_closed


def test_settings_hot_reload_applies_structured_output_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-5-mini")
    service = LLMService(
        clients={"default": client},
        profiles={
            "default": LLMProfile(
                provider="openai",
                model="gpt-5-mini",
            )
        },
    )
    monkeypatch.setattr(
        settings_api,
        "current_services",
        lambda: type("Services", (), {"llm": service})(),
    )

    settings_api._hot_reload_llm(
        {"default": LLMProfilePayload(structured_output_policy="native_required")}
    )

    assert client.structured_output_policy == "native_required"
    profile = service.profile("default")
    assert profile is not None
    assert profile.structured_output_policy == "native_required"


def test_settings_hot_reload_applies_prompt_cache_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-5-mini")
    service = LLMService(
        clients={"default": client},
        profiles={"default": LLMProfile(provider="openai", model="gpt-5-mini")},
    )
    monkeypatch.setattr(
        settings_api,
        "current_services",
        lambda: type("Services", (), {"llm": service})(),
    )

    settings_api._hot_reload_llm({"default": LLMProfilePayload(prompt_cache_policy="required")})

    assert client.prompt_cache_policy == "required"
    profile = service.profile("default")
    assert profile is not None
    assert profile.prompt_cache_policy == "required"


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

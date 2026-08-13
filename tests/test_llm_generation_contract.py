from __future__ import annotations

import asyncio

import pytest

from aethergraph.services.llm import (
    AssistantOutput,
    ChatMessage,
    GenerationOptions,
    ImagePart,
    LLMRequestCompatibilityError,
    ModelContinuation,
    ModelReasoningDelta,
    ModelRequest,
    ModelResponse,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsage,
    ModelUsageUpdate,
    StructuredOutputRequest,
    TextPart,
    ToolCall,
    ToolCallOutput,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolDiscoveryEvent,
    ToolDiscoveryRequest,
    get_endpoint_adapter,
    message_from_text,
    validate_model_request,
)
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.request_preparation import prepare_model_request
from aethergraph.services.llm.tool_calling import tool_call_request_fingerprint
from aethergraph.services.llm.types import LLMUnsupportedFeatureError


def _continuation() -> ModelContinuation:
    return ModelContinuation(
        checkpoint_id="checkpoint_1",
        revision=1,
        provider="openai",
        model="gpt-test",
        contract_version="responses/v1",
        turn_id="turn_1",
        integrity_digest="0" * 64,
        opaque_payload={"response_id": "response_1"},
    )


def _tool() -> ToolDefinition:
    return ToolDefinition(
        name="lookup",
        description="Look up one value.",
        input_schema={"type": "object", "properties": {}},
        exposure="deferred",
    )


def test_model_request_supports_direct_completion_without_tools() -> None:
    request = ModelRequest(messages=(message_from_text("user", "Hello"),))

    assert request.tool_choice == "none"
    assert request.tools == ()
    assert request.native_tool_search is None


def test_model_request_preserves_native_discovery_continuation_state() -> None:
    continuation = _continuation()
    request = ModelRequest(
        messages=(ChatMessage("user", (TextPart("Continue"),)),),
        tools=(_tool(),),
        tool_choice="auto",
        max_tool_calls=2,
        native_tool_search=ToolDiscoveryRequest("native_client"),
        active_tool_names=("lookup",),
        turn_id="turn_1",
        continuation=continuation,
        tool_outputs=(ToolCallOutput("call_1", "done"),),
        generation=GenerationOptions(max_output_tokens=128),
    )

    assert request.continuation is continuation
    assert request.tool_outputs[0].call_id == "call_1"
    assert request.max_tool_calls == 2


def test_model_request_rejects_engine_projected_as_native_search() -> None:
    with pytest.raises(ValueError, match="native_hosted or native_client"):
        ModelRequest(
            messages=(message_from_text("user", "Search"),),
            tools=(_tool(),),
            tool_choice="auto",
            turn_id="turn_1",
            native_tool_search=ToolDiscoveryRequest("engine_projected"),
        )


def test_model_request_accepts_structured_response_contract() -> None:
    response_format = StructuredOutputRequest(
        name="Answer",
        schema={"type": "object", "properties": {}},
    )

    request = ModelRequest(
        messages=(message_from_text("user", "Answer"),),
        response_format=response_format,
    )

    assert request.response_format is response_format


def test_canonical_request_preparation_preserves_multimodal_and_tool_messages() -> None:
    request = ModelRequest(
        messages=(
            ChatMessage(
                "user",
                (
                    TextPart("Inspect"),
                    ImagePart(url="https://example.test/image.png"),
                    ImagePart(data=b"image", mime_type="image/png"),
                ),
                name="operator",
            ),
            ChatMessage("tool", (TextPart("done"),), tool_call_id="call_1"),
        )
    )

    messages, tool_request = prepare_model_request(request)

    assert tool_request is None
    assert messages[0]["name"] == "operator"
    assert [part["type"] for part in messages[0]["content"]] == [
        "text",
        "image_url",
        "image",
    ]
    assert messages[0]["content"][2]["source"]["data"] == "aW1hZ2U="
    assert messages[1]["tool_call_id"] == "call_1"


def test_model_request_requires_continuation_for_tool_outputs() -> None:
    with pytest.raises(ValueError, match="require a continuation"):
        ModelRequest(
            messages=(message_from_text("user", "Continue"),),
            tools=(_tool(),),
            tool_choice="auto",
            tool_outputs=(ToolCallOutput("call_1", "done"),),
        )


def test_whole_request_validation_reports_required_adapter_capabilities() -> None:
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(_tool(),),
        tool_choice="auto",
    )

    report = validate_model_request(request)

    assert report.valid is True
    assert report.required_adapter_capabilities == ("native_tools",)
    assert report.required_model_capabilities == ("native_tool_calling",)
    assert report.diagnostics == ()


def test_whole_request_validation_rejects_structured_output_with_native_tools() -> None:
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(_tool(),),
        tool_choice="auto",
        response_format=StructuredOutputRequest(
            name="Answer",
            schema={"type": "object", "properties": {}},
        ),
    )

    report = validate_model_request(request)

    assert report.valid is False
    assert report.diagnostics[0].code == "structured_output_with_native_tools"


def test_whole_request_validation_rejects_raw_output_with_native_tools() -> None:
    tool = ToolDefinition(
        name="lookup",
        description="Look up one value.",
        input_schema={"type": "object", "properties": {}},
    )
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(tool,),
        tool_choice="auto",
        response_format="raw",
    )

    report = validate_model_request(request)

    assert not report.valid
    assert report.diagnostics[0].code == "raw_response_with_native_tools"


def test_whole_request_validation_rejects_continuation_without_tool_catalog() -> None:
    request = ModelRequest(
        messages=(message_from_text("user", "Continue"),),
        turn_id="turn_1",
        continuation=_continuation(),
        tool_outputs=(ToolCallOutput("call_1", "done"),),
    )

    report = validate_model_request(request)

    assert report.valid is False
    assert report.diagnostics[0].code == "tool_continuation_without_tools"


def test_whole_request_validation_clamps_to_preselected_adapter() -> None:
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(_tool(),),
        tool_choice="auto",
    )

    report = validate_model_request(
        request,
        adapter=get_endpoint_adapter("dummy_chat"),
    )

    assert report.valid is False
    assert report.diagnostics[0].code == "adapter_capability_unimplemented"


@pytest.mark.asyncio
async def test_generate_rejects_invalid_whole_request_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(_tool(),),
        tool_choice="auto",
        response_format="json_object",
    )
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")

    async def reject_runtime(*args, **kwargs):
        raise AssertionError("invalid request reached provider runtime")

    monkeypatch.setattr(client, "_invoke_generation_runtime", reject_runtime)

    with pytest.raises(LLMRequestCompatibilityError) as exc_info:
        await client.generate(request)

    assert exc_info.value.report.diagnostics[0].code == ("structured_output_with_native_tools")


@pytest.mark.asyncio
async def test_canonical_request_versions_fingerprint_without_rotating_legacy_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = ToolCallRequest(tools=(_tool(),), choice="auto")
    canonical_request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(_tool(),),
        tool_choice="auto",
        call_name="engine.select_action",
    )
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    captured = {}

    async def capture_runtime(messages, **kwargs):
        captured.update(kwargs)
        return ModelResponse(items=(AssistantOutput("output_1", "done"),))

    monkeypatch.setattr(client, "_invoke_generation_runtime", capture_runtime)
    await client.generate(canonical_request)
    projected = captured["tool_request"]

    assert tool_call_request_fingerprint(legacy) == (
        "b1944bede27a704b605692fc79bf14ac612f92ed1783540975c1e39aec9da1ee"
    )
    assert projected.fingerprint_version == "model_request/v1"
    assert captured["call_name"] == "engine.select_action"
    assert tool_call_request_fingerprint(projected) != tool_call_request_fingerprint(legacy)


@pytest.mark.asyncio
async def test_chat_facade_projects_canonical_runtime_response_and_raw_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    response = ModelResponse(
        items=(AssistantOutput("output_1", "done"),),
        usage=ModelUsage.from_provider_usage(
            {"input_tokens": 3, "output_tokens": 1, "provider_extension": 7}
        ),
    )

    async def typed_runtime(messages, **kwargs):
        return response

    monkeypatch.setattr(client, "_invoke_generation_runtime", typed_runtime)

    text, usage = await client.chat([{"role": "user", "content": "Hello"}])

    assert text == "done"
    assert usage == {"input_tokens": 3, "output_tokens": 1, "provider_extension": 7}


def test_canonical_estimate_counts_tool_schema_and_output_reservation() -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    direct = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        generation=GenerationOptions(max_output_tokens=64),
    )
    with_tool = ModelRequest(
        messages=direct.messages,
        tools=(_tool(),),
        tool_choice="auto",
        generation=direct.generation,
    )

    direct_estimate = client.estimate(direct)
    tool_estimate = client.estimate(with_tool)

    assert direct_estimate.reserved_output_tokens == 64
    assert tool_estimate.estimated_input_tokens > direct_estimate.estimated_input_tokens


@pytest.mark.asyncio
async def test_generate_stream_emits_typed_deltas_and_terminal_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")

    async def fake_stream(messages, **kwargs):
        await kwargs["on_thinking_delta"]("Plan")
        await kwargs["on_usage_update"]({"input_tokens": 3})
        await kwargs["on_usage_update"]({"input_tokens": 3})
        await kwargs["on_delta"]("Hel")
        await kwargs["on_delta"]("lo")
        await kwargs["on_usage_update"]({"input_tokens": 3, "output_tokens": 2})
        return "Hello", {"input_tokens": 3, "output_tokens": 2}

    monkeypatch.setattr(client, "_invoke_stream_runtime", fake_stream)

    events = [
        event
        async for event in client.generate_stream(
            ModelRequest(messages=(message_from_text("user", "Hello"),))
        )
    ]

    assert isinstance(events[0], ModelReasoningDelta)
    assert events[0].index == 0
    usage_updates = [event for event in events if isinstance(event, ModelUsageUpdate)]
    assert [event.index for event in usage_updates] == [0, 1]
    assert usage_updates[0].usage.availability == "partial"
    assert usage_updates[1].usage.availability == "complete"
    assert [event.delta for event in events if isinstance(event, ModelTextDelta)] == [
        "Hel",
        "lo",
    ]
    assert isinstance(events[-1], ModelStreamCompleted)
    assert events[-1].response.text == "Hello"
    assert events[-1].response.usage.total_input_tokens == 3
    assert events[-1].response.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_generate_stream_rejects_tools_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")

    async def reject_runtime(*args, **kwargs):
        raise AssertionError("unsupported stream reached provider runtime")

    monkeypatch.setattr(client, "_invoke_stream_runtime", reject_runtime)
    request = ModelRequest(
        messages=(message_from_text("user", "Look up"),),
        tools=(_tool(),),
        tool_choice="auto",
    )

    with pytest.raises(LLMUnsupportedFeatureError, match="streaming native Tools"):
        async for _event in client.generate_stream(request):
            pass


@pytest.mark.asyncio
async def test_generate_stream_close_cancels_active_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericLLMClient(provider="openai", model="gpt-test", api_key="test")
    cancelled = asyncio.Event()

    async def waiting_stream(messages, **kwargs):
        try:
            await kwargs["on_delta"]("ready")
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(client, "_invoke_stream_runtime", waiting_stream)
    stream = client.generate_stream(ModelRequest(messages=(message_from_text("user", "Hi"),)))

    first = await anext(stream)
    assert isinstance(first, ModelTextDelta)
    await stream.aclose()

    assert cancelled.is_set()


def test_model_usage_distinguishes_unavailable_from_reported_zero() -> None:
    unavailable = ModelUsage.unavailable()
    reported_zero = ModelUsage.from_provider_usage({"prompt_tokens": 0, "completion_tokens": 0})

    assert unavailable.availability == "unavailable"
    assert unavailable.total_input_tokens is None
    assert reported_zero.availability == "complete"
    assert reported_zero.total_input_tokens == 0
    assert reported_zero.output_tokens == 0


def test_model_usage_normalizes_cache_inclusive_total_and_reasoning() -> None:
    usage = ModelUsage.from_provider_usage(
        {
            "input_tokens": 12,
            "output_tokens": 5,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 40,
            "output_tokens_details": {"reasoning_tokens": 3},
        }
    )

    assert usage.availability == "complete"
    assert usage.total_input_tokens == 82
    assert usage.uncached_input_tokens == 12
    assert usage.cache_read_tokens == 40
    assert usage.cache_write_tokens == 30
    assert usage.reasoning_tokens == 3


def test_model_usage_preserves_partial_receipt() -> None:
    usage = ModelUsage.from_provider_usage({"output_tokens": 4})

    assert usage.availability == "partial"
    assert usage.total_input_tokens is None
    assert usage.output_tokens == 4


def test_model_response_preserves_one_cross_category_order() -> None:
    discovery = ToolDiscoveryEvent(
        event_id="search_1",
        mode="native_client",
        source="provider_client",
        arguments={"query": "lookup"},
        tool_refs=("lookup",),
    )
    response = ModelResponse(
        items=(
            AssistantOutput("output_1", "Checking. "),
            discovery,
            ToolCall("call_1", "lookup", {"key": "A"}),
            AssistantOutput("output_2", "Done."),
        ),
        finish_reason="tool_calls",
        transport_checkpoint=_continuation(),
        usage=ModelUsage.from_provider_usage({"prompt_tokens": 10, "completion_tokens": 2}),
    )

    assert ModelResponse is ToolCallResponse
    assert response.text == "Checking. Done."
    assert response.discovery_events == (discovery,)
    assert tuple(call.call_id for call in response.calls) == ("call_1",)
    assert response.continuation is response.transport_checkpoint
    assert response.usage.availability == "complete"


def test_model_response_accepts_direct_assistant_completion() -> None:
    response = ModelResponse(
        items=(AssistantOutput("output_1", "A direct answer."),),
        finish_reason="stop",
    )

    assert response.text == "A direct answer."
    assert response.calls == ()
    assert response.usage.availability == "unavailable"


def test_model_usage_rejects_unavailable_numeric_counts() -> None:
    with pytest.raises(ValueError, match="unavailable usage"):
        ModelUsage(availability="unavailable", total_input_tokens=0)

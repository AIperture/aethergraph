from __future__ import annotations

import pytest

from aethergraph.services.llm import (
    AssistantOutput,
    ModelContinuation,
    ModelResponse,
    ModelUsage,
    ToolCall,
    ToolCallResponse,
    ToolDiscoveryEvent,
)


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

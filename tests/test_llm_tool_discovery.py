from __future__ import annotations

import json

import pytest

from aethergraph.services.llm import (
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolDiscoveryCapabilities,
    ToolDiscoveryError,
    ToolDiscoveryEvent,
    ToolDiscoveryRequest,
    ToolNamespace,
    ToolTransportCheckpoint,
)
from aethergraph.services.llm.tool_calling import tool_call_request_fingerprint


def _checkpoint() -> ToolTransportCheckpoint:
    return ToolTransportCheckpoint(
        checkpoint_id="checkpoint_1",
        provider="openai",
        model="example-model",
        contract_version="responses/v1",
        turn_id="turn_1",
        discovery_event_id="search_1",
        integrity_digest="0" * 64,
        expires_at="end_of_turn",
        opaque_payload={"output": [{"type": "tool_search_call"}]},
    )


def test_tool_call_contract_carries_deferred_discovery_and_opaque_checkpoint() -> None:
    request = ToolCallRequest(
        tools=(
            ToolDefinition(
                name="read_document",
                description="Read one document.",
                input_schema={"type": "object", "properties": {}},
                exposure="deferred",
                namespace=ToolNamespace("docs", "Document operations."),
            ),
        ),
        discovery=ToolDiscoveryRequest("native_client", max_results=7),
        transport_checkpoint=_checkpoint(),
    )

    assert request.tools[0].exposure == "deferred"
    assert request.discovery is not None
    assert request.transport_checkpoint is not None
    assert len(tool_call_request_fingerprint(request)) == 64


def test_response_observation_normalizes_events_without_exposing_checkpoint() -> None:
    response = ToolCallResponse(
        calls=(),
        discovery_events=(
            ToolDiscoveryEvent(
                event_id="search_1",
                mode="native_hosted",
                source="provider_hosted",
                arguments={"paths": ["docs"]},
                tool_refs=("docs.read",),
                provider_reference_ids=("provider_ref_1",),
            ),
        ),
        transport_checkpoint=_checkpoint(),
    )

    observation = json.loads(response.observation_text())

    assert observation["discovery_events"][0]["tool_refs"] == ["docs.read"]
    assert "transport_checkpoint" not in observation
    assert "opaque_payload" not in response.observation_text()


def test_failed_discovery_event_requires_a_structured_error() -> None:
    with pytest.raises(ValueError, match="require an error"):
        ToolDiscoveryEvent(
            event_id="search_1",
            mode="engine_projected",
            source="engine",
            arguments={"query": "read"},
            status="failed",
        )

    event = ToolDiscoveryEvent(
        event_id="search_1",
        mode="engine_projected",
        source="engine",
        arguments={"query": "read"},
        status="failed",
        error=ToolDiscoveryError(
            code="search_unavailable",
            summary="Search is temporarily unavailable.",
            retryable=True,
        ),
    )
    assert event.error is not None and event.error.retryable


def test_capability_binding_is_model_specific_and_has_no_mode_fallback() -> None:
    capabilities = ToolDiscoveryCapabilities(
        provider="google",
        model="example-model",
        endpoint_family="generateContent",
        supported_modes=("engine_projected",),
        max_results=8,
    )

    assert capabilities.supports(ToolDiscoveryRequest("engine_projected", 8))
    assert not capabilities.supports(ToolDiscoveryRequest("native_hosted", 8))
    assert not capabilities.supports(ToolDiscoveryRequest("engine_projected", 9))


def test_checkpoint_requires_private_payload_or_durable_reference() -> None:
    with pytest.raises(ValueError, match="requires opaque_payload or durable_ref"):
        ToolTransportCheckpoint(
            checkpoint_id="checkpoint_1",
            provider="openai",
            model="example-model",
            contract_version="responses/v1",
            turn_id="turn_1",
            discovery_event_id="search_1",
            integrity_digest="0" * 64,
            expires_at="end_of_turn",
        )

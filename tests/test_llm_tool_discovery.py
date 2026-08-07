from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from aethergraph.services.llm import (
    LLMToolCallCapabilityError,
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
    ToolDiscoveryCapabilities,
    ToolDiscoveryError,
    ToolDiscoveryEvent,
    ToolDiscoveryMode,
    ToolDiscoveryModeCapability,
    ToolDiscoveryRequest,
    ToolNamespace,
    ToolTransportCheckpoint,
    resolve_tool_discovery_capabilities,
)
from aethergraph.services.llm._openai_mixin import _openai_checkpoint
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.tool_calling import tool_call_request_fingerprint


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _CountingHttpClient:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {}
        self.calls = 0
        self.last_json: dict[str, Any] | None = None

    async def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.calls += 1
        self.last_json = json
        return _FakeResponse(self.payload)


def _discovery_request(
    mode: ToolDiscoveryMode = "engine_projected",
    *,
    max_results: int = 5,
) -> ToolCallRequest:
    return ToolCallRequest(
        tools=(
            ToolDefinition(
                name="finish",
                description="Finish the task.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
        discovery=ToolDiscoveryRequest(mode, max_results),
        turn_id="turn_1",
    )


def _openai_capabilities(
    *modes: ToolDiscoveryModeCapability,
) -> ToolDiscoveryCapabilities:
    return ToolDiscoveryCapabilities(
        provider="openai",
        model="example-model",
        endpoint_family="responses",
        supported_modes=modes,
    )


def _checkpoint(
    *,
    revision: int = 1,
    turn_id: str = "turn_1",
    durable_ref: str | None = None,
) -> ToolTransportCheckpoint:
    return ToolTransportCheckpoint(
        checkpoint_id=f"checkpoint_{revision}",
        revision=revision,
        provider="openai",
        model="example-model",
        contract_version="responses/v1",
        turn_id=turn_id,
        integrity_digest="0" * 64,
        opaque_payload={"output": [{"type": "tool_search_call"}]},
        durable_ref=durable_ref,
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
        turn_id="turn_1",
        transport_checkpoint=_checkpoint(),
    )

    assert request.tools[0].exposure == "deferred"
    assert request.discovery is not None
    assert request.turn_id == "turn_1"
    assert request.transport_checkpoint is not None
    assert len(tool_call_request_fingerprint(request)) == 64


def test_discovery_turn_identity_is_required_but_does_not_rotate_cache_contract() -> None:
    tool = ToolDefinition("finish", "Finish.", {"type": "object"})

    with pytest.raises(ValueError, match="semantic turn_id"):
        ToolCallRequest(
            tools=(tool,),
            discovery=ToolDiscoveryRequest("native_client"),
        )

    first = ToolCallRequest(
        tools=(tool,),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
    )
    second = ToolCallRequest(
        tools=(tool,),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_2",
    )

    assert tool_call_request_fingerprint(first) == tool_call_request_fingerprint(second)


def test_transport_checkpoint_must_match_request_turn() -> None:
    with pytest.raises(ValueError, match="must match"):
        ToolCallRequest(
            tools=(ToolDefinition("finish", "Finish.", {"type": "object"}),),
            discovery=ToolDiscoveryRequest("native_client"),
            turn_id="turn_2",
            transport_checkpoint=_checkpoint(turn_id="turn_1"),
        )


def test_native_client_deferred_activation_preserves_cache_contract() -> None:
    immediate = ToolDefinition("finish", "Finish.", {"type": "object"})
    deferred = ToolDefinition(
        "read_document",
        "Read.",
        {"type": "object"},
        exposure="deferred",
    )
    initial = ToolCallRequest(
        tools=(immediate,),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
    )
    activated = ToolCallRequest(
        tools=(immediate, deferred),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
    )

    assert tool_call_request_fingerprint(initial) == tool_call_request_fingerprint(activated)


def test_builtin_discovery_capabilities_are_exact_and_implemented_only() -> None:
    openai = resolve_tool_discovery_capabilities("openai", "gpt-5.6", "responses")

    assert openai is not None
    assert [mode.mode for mode in openai.supported_modes] == ["native_client"]
    assert resolve_tool_discovery_capabilities("openai", "gpt-5.5", "responses") is None
    assert resolve_tool_discovery_capabilities("azure", "gpt-5.5", "chat.completions") is None


def test_response_observation_normalizes_events_without_exposing_checkpoint() -> None:
    event = ToolDiscoveryEvent(
        event_id="search_1",
        mode="native_hosted",
        source="provider_hosted",
        arguments={"paths": ["docs"]},
        tool_refs=("docs.read",),
        provider_reference_ids=("provider_ref_1",),
    )
    response = ToolCallResponse(
        items=(event, ToolCall("call_1", "read_document", {"path": "a.md"})),
        transport_checkpoint=_checkpoint(),
    )

    observation = json.loads(response.observation_text())

    assert [item["kind"] for item in observation["items"]] == [
        "tool_discovery",
        "tool_call",
    ]
    assert observation["items"][0]["tool_refs"] == ["docs.read"]
    assert response.discovery_events == (event,)
    assert response.calls[0].call_id == "call_1"
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
        supported_modes=(
            ToolDiscoveryModeCapability(
                mode="engine_projected",
                max_results=8,
            ),
        ),
    )

    assert capabilities.supports(ToolDiscoveryRequest("engine_projected", 8))
    assert not capabilities.supports(ToolDiscoveryRequest("native_hosted", 8))
    assert not capabilities.supports(ToolDiscoveryRequest("engine_projected", 9))


def test_capability_limits_are_owned_by_each_mode() -> None:
    capabilities = ToolDiscoveryCapabilities(
        provider="openai",
        model="example-model",
        endpoint_family="responses",
        supported_modes=(
            ToolDiscoveryModeCapability(
                mode="native_hosted",
                replay_requirement="previous_response",
                result_limit_behavior="provider_fixed",
                max_results=5,
                protocol_version="tool-search/v1",
            ),
            ToolDiscoveryModeCapability(
                mode="engine_projected",
                max_results=10,
            ),
        ),
    )

    assert not capabilities.supports(ToolDiscoveryRequest("native_hosted", 4))
    assert capabilities.supports(ToolDiscoveryRequest("native_hosted", 5))
    assert capabilities.supports(ToolDiscoveryRequest("native_hosted", 8))
    assert capabilities.supports(ToolDiscoveryRequest("engine_projected", 10))
    assert not capabilities.supports(ToolDiscoveryRequest("engine_projected", 11))


def test_capability_rejects_duplicate_mode_records() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        ToolDiscoveryCapabilities(
            provider="openai",
            model="example-model",
            endpoint_family="responses",
            supported_modes=(
                ToolDiscoveryModeCapability(mode="engine_projected"),
                ToolDiscoveryModeCapability(mode="engine_projected", max_results=8),
            ),
        )


def test_capability_binding_rejects_a_different_endpoint_family() -> None:
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")

    with pytest.raises(LLMToolCallCapabilityError, match="does not match active binding"):
        client.bind_tool_discovery_capabilities(
            ToolDiscoveryCapabilities(
                provider="openai",
                model="example-model",
                endpoint_family="chat.completions",
                supported_modes=(ToolDiscoveryModeCapability(mode="engine_projected"),),
            )
        )


@pytest.mark.asyncio
async def test_unbound_discovery_request_fails_before_provider_traffic() -> None:
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")
    fake_http = _CountingHttpClient()
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    with pytest.raises(LLMToolCallCapabilityError, match="No exact discovery capability"):
        await client.chat(
            [{"role": "user", "content": "finish"}],
            tool_request=_discovery_request(),
        )

    assert fake_http.calls == 0


@pytest.mark.asyncio
async def test_openai_native_client_search_round_trips_private_checkpoint() -> None:
    client = GenericLLMClient(
        "openai",
        "gpt-5.6",
        api_key="test",
        base_url="https://api.openai.test/v1",
    )
    client.bind_tool_discovery_capabilities(
        ToolDiscoveryCapabilities(
            provider="openai",
            model="gpt-5.6",
            endpoint_family="responses",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="native_client",
                    replay_requirement="previous_response",
                    max_results=50,
                    protocol_version="responses.tool_search",
                ),
            ),
        )
    )
    fake_http = _CountingHttpClient(
        {
            "id": "resp_search_1",
            "status": "completed",
            "output": [
                {
                    "type": "tool_search_call",
                    "execution": "client",
                    "call_id": "search_call_1",
                    "status": "completed",
                    "arguments": {"goal": "open a document"},
                }
            ],
        }
    )
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    immediate = ToolDefinition("finish", "Finish.", {"type": "object"})
    initial_request = ToolCallRequest(
        tools=(immediate,),
        discovery=ToolDiscoveryRequest("native_client", max_results=5),
        turn_id="turn_1",
    )

    first, _usage = await client.chat(
        [{"role": "user", "content": "open the document"}],
        tool_request=initial_request,
    )

    assert isinstance(first, ToolCallResponse)
    assert first.discovery_events[0].query == "open a document"
    assert first.discovery_events[0].provider_reference_ids == ("search_call_1",)
    assert first.transport_checkpoint is not None
    assert first.transport_checkpoint.revision == 1
    assert fake_http.last_json is not None
    assert fake_http.last_json["tools"][-1]["type"] == "tool_search"
    assert fake_http.last_json["tools"][-1]["execution"] == "client"

    deferred = ToolDefinition(
        "read_document",
        "Read one document.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        exposure="deferred",
        namespace=ToolNamespace("docs", "Document operations."),
    )
    fake_http.payload = {
        "id": "resp_call_1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_document",
                "arguments": '{"path":"a.md"}',
            }
        ],
    }
    continuation_request = ToolCallRequest(
        tools=(immediate, deferred),
        discovery=ToolDiscoveryRequest("native_client", max_results=5),
        turn_id="turn_1",
        transport_checkpoint=first.transport_checkpoint,
    )

    second, _usage = await client.chat(
        [{"role": "user", "content": "open the document"}],
        tool_request=continuation_request,
    )

    assert isinstance(second, ToolCallResponse)
    assert second.calls[0].name == "read_document"
    assert second.transport_checkpoint is not None
    assert second.transport_checkpoint.revision == 2
    assert fake_http.last_json is not None
    assert fake_http.last_json["previous_response_id"] == "resp_search_1"
    assert "tools" not in fake_http.last_json
    search_output = fake_http.last_json["input"][0]
    assert search_output["type"] == "tool_search_output"
    assert search_output["call_id"] == "search_call_1"
    assert [tool["name"] for tool in search_output["tools"]] == ["read_document"]


@pytest.mark.asyncio
async def test_openai_consumed_checkpoint_does_not_replay_search_output() -> None:
    client = GenericLLMClient(
        "openai",
        "gpt-5.6",
        api_key="test",
        base_url="https://api.openai.test/v1",
    )
    client.bind_tool_discovery_capabilities(
        ToolDiscoveryCapabilities(
            provider="openai",
            model="gpt-5.6",
            endpoint_family="responses",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="native_client",
                    replay_requirement="previous_response",
                    max_results=50,
                    protocol_version="responses.tool_search",
                ),
            ),
        )
    )
    immediate = ToolDefinition("finish", "Finish.", {"type": "object"})
    deferred = ToolDefinition(
        "read_document",
        "Read.",
        {"type": "object"},
        exposure="deferred",
    )
    pending_request = ToolCallRequest(
        tools=(immediate,),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
    )
    pending = _openai_checkpoint(
        request=pending_request,
        model="gpt-5.6",
        response_id="resp_search_1",
        state="pending_search",
        call_id="search_call_1",
    )
    loaded_request = ToolCallRequest(
        tools=(immediate, deferred),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
        transport_checkpoint=pending,
    )
    consumed = _openai_checkpoint(
        request=loaded_request,
        model="gpt-5.6",
        response_id="resp_call_1",
        state="consumed",
    )
    fake_http = _CountingHttpClient(
        {
            "id": "resp_finish_1",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_finish_1",
                    "name": "finish",
                    "arguments": "{}",
                }
            ],
        }
    )
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    request = ToolCallRequest(
        tools=(immediate, deferred),
        discovery=ToolDiscoveryRequest("native_client"),
        turn_id="turn_1",
        transport_checkpoint=consumed,
    )

    response, _usage = await client.chat(
        [{"role": "user", "content": "continue"}],
        tool_request=request,
    )

    assert isinstance(response, ToolCallResponse)
    assert fake_http.last_json is not None
    assert "previous_response_id" not in fake_http.last_json
    deferred_body = next(
        tool for tool in fake_http.last_json["tools"] if tool.get("name") == "read_document"
    )
    assert "defer_loading" not in deferred_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_call_request", "model_override", "expected_detail"),
    [
        (
            _discovery_request("native_client"),
            None,
            "selected discovery mode is not declared",
        ),
        (
            _discovery_request("engine_projected", max_results=9),
            None,
            "max_results=9 cannot be honored",
        ),
        (
            _discovery_request("engine_projected"),
            "different-model",
            "does not match active binding",
        ),
    ],
)
async def test_discovery_rejection_occurs_before_provider_traffic(
    tool_call_request: ToolCallRequest,
    model_override: str | None,
    expected_detail: str,
) -> None:
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")
    client.bind_tool_discovery_capabilities(
        _openai_capabilities(
            ToolDiscoveryModeCapability(mode="engine_projected", max_results=8),
        )
    )
    fake_http = _CountingHttpClient()
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    kwargs = {"model": model_override} if model_override is not None else {}

    with pytest.raises(LLMToolCallCapabilityError, match=expected_detail):
        await client.chat(
            [{"role": "user", "content": "finish"}],
            tool_request=tool_call_request,
            **kwargs,
        )

    assert fake_http.calls == 0


@pytest.mark.asyncio
async def test_exact_supported_discovery_binding_reaches_provider_once() -> None:
    payload = {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "finish",
                "arguments": "{}",
                "status": "completed",
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")
    client.bind_tool_discovery_capabilities(
        _openai_capabilities(
            ToolDiscoveryModeCapability(mode="engine_projected", max_results=8),
        )
    )
    fake_http = _CountingHttpClient(payload)
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    response, usage = await client.chat(
        [{"role": "user", "content": "finish"}],
        tool_request=_discovery_request(max_results=8),
    )

    assert fake_http.calls == 1
    assert isinstance(response, ToolCallResponse)
    assert response.calls[0].name == "finish"
    assert usage["input_tokens"] == 3


def test_checkpoint_requires_private_payload_or_durable_reference() -> None:
    with pytest.raises(ValueError, match="requires opaque_payload or durable_ref"):
        ToolTransportCheckpoint(
            checkpoint_id="checkpoint_1",
            revision=1,
            provider="openai",
            model="example-model",
            contract_version="responses/v1",
            turn_id="turn_1",
            integrity_digest="0" * 64,
        )


def test_checkpoint_requires_positive_revision_and_semantic_turn_expiry() -> None:
    with pytest.raises(ValueError, match="revision"):
        ToolTransportCheckpoint(
            checkpoint_id="checkpoint_1",
            revision=0,
            provider="openai",
            model="example-model",
            contract_version="responses/v1",
            turn_id="turn_1",
            integrity_digest="0" * 64,
            opaque_payload={"output": []},
        )

    with pytest.raises(ValueError, match="end_of_turn"):
        ToolTransportCheckpoint(
            checkpoint_id="checkpoint_1",
            revision=1,
            provider="openai",
            model="example-model",
            contract_version="responses/v1",
            turn_id="turn_1",
            integrity_digest="0" * 64,
            expires_at="end_of_session",  # type: ignore[arg-type]
            opaque_payload={"output": []},
        )


def test_client_checkpoint_pin_replaces_only_with_a_newer_same_turn_revision() -> None:
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")
    first = _checkpoint(revision=1)
    second = _checkpoint(revision=2)

    first_ref = client.pin_tool_transport_checkpoint(first)
    second_ref = client.pin_tool_transport_checkpoint(second)

    assert first_ref != second_ref
    assert client.resolve_tool_transport_checkpoint(second_ref, turn_id="turn_1") == second
    with pytest.raises(KeyError, match="not pinned"):
        client.resolve_tool_transport_checkpoint(first_ref, turn_id="turn_1")
    with pytest.raises(ValueError, match="advance monotonically"):
        client.pin_tool_transport_checkpoint(first)


def test_client_checkpoint_resolve_is_same_turn_and_release_is_idempotent() -> None:
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")
    checkpoint = _checkpoint(durable_ref="artifact:checkpoint_1")
    reference = client.pin_tool_transport_checkpoint(checkpoint)

    assert reference == "artifact:checkpoint_1"
    with pytest.raises(ValueError, match="different turn"):
        client.resolve_tool_transport_checkpoint(reference, turn_id="turn_2")

    client.release_tool_transport_checkpoint(reference)
    client.release_tool_transport_checkpoint(reference)
    with pytest.raises(KeyError, match="not pinned"):
        client.resolve_tool_transport_checkpoint(reference, turn_id="turn_1")


@pytest.mark.asyncio
async def test_checkpoint_request_binding_rejects_before_provider_traffic() -> None:
    client = GenericLLMClient(provider="openai", model="example-model", api_key="test")
    client.bind_tool_discovery_capabilities(
        _openai_capabilities(
            ToolDiscoveryModeCapability(mode="engine_projected", max_results=8),
        )
    )
    fake_http = _CountingHttpClient()
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()
    wrong_turn_model = ToolTransportCheckpoint(
        checkpoint_id="checkpoint_1",
        revision=1,
        provider="openai",
        model="different-model",
        contract_version="responses/v1",
        turn_id="turn_1",
        integrity_digest="0" * 64,
        opaque_payload={"output": []},
    )
    request = _discovery_request(max_results=8)
    request = ToolCallRequest(
        tools=request.tools,
        discovery=request.discovery,
        turn_id="turn_1",
        transport_checkpoint=wrong_turn_model,
    )

    with pytest.raises(ValueError, match="binding does not match"):
        await client.chat(
            [{"role": "user", "content": "finish"}],
            tool_request=request,
        )

    assert fake_http.calls == 0

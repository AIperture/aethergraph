"""Physical Azure OpenAI Chat endpoint adapters."""

from __future__ import annotations

import json
from typing import Any

from aethergraph.services.llm.adapters.openai_compatible import (
    _openai_like_continuation_messages,
    _openai_like_tool_call_response,
    _openai_like_tool_definitions,
    _stream_openai_like_chat_completions,
)
from aethergraph.services.llm.adapters.openai_responses import (
    _openai_checkpoint_payload,
    _openai_continuation_request_tools,
    _openai_function_tool,
    _openai_request_tools,
    _openai_tool_call_response,
)
from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.tool_calling import (
    LLMToolCallResponseError,
    ToolCallRequest,
    ToolCallResponse,
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
)
from aethergraph.services.llm.utils import (
    _ensure_system_json_directive,
    _normalize_openai_responses_input,
)


def _first_text(choices):
    """Extract text and usage from OpenAI-style choices list."""
    if not choices:
        return "", {}
    c = choices[0]
    text = (c.get("message", {}) or {}).get("content") or c.get("text") or ""
    usage = {}
    return text, usage


class AzureChatAdapter:
    """Physical adapters for exact Azure OpenAI Chat endpoints."""

    @staticmethod
    async def invoke_chat_completions(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tool_request: ToolCallRequest | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Invoke one Azure Chat Completions request.

        Intro:
            The adapter uses the explicitly pinned Azure Chat Completions route
            for direct, structured, Tool-call, and Tool-result continuation calls.

        Examples:
            Send a direct request:
                ```python
                result = await AzureChatAdapter.invoke_chat_completions(
                    client,
                    messages,
                    model="deployment-a",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=False,
                )
                ```

            Continue a Tool request:
                ```python
                result = await AzureChatAdapter.invoke_chat_completions(
                    client,
                    messages,
                    model="deployment-a",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=False,
                    tool_request=continued_request,
                )
                ```

        Args:
            host: Bound generic client owning the Azure transport.
            messages: Provider-projected stable conversation messages.
            model: Exact Azure deployment identity.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text, JSON, or raw response mode.
            json_schema: Optional canonical JSON schema.
            fail_on_unsupported: Whether unsupported native formatting fails.
            tools: Optional legacy provider Tool declarations.
            tool_choice: Optional legacy provider Tool-selection policy.
            tool_request: Optional canonical native Tool request and continuation.
            **kw: Additional bounded Azure request options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Normalized response and raw Azure usage with transport metadata.

        Notes:
            The selected endpoint never switches to Azure Responses according to
            Tool presence. Shared retry and accounting remain caller-owned.
        """

        await host._ensure_client()
        assert host._client is not None

        if not (host.base_url and host.azure_deployment):
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
            )

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        replay_messages: tuple[dict[str, Any], ...] = ()
        if tool_request is not None:
            msg_for_provider, replay_messages = _openai_like_continuation_messages(
                messages,
                tool_request=tool_request,
                provider="azure",
                model=model,
            )
        payload: dict[str, Any] = {
            "messages": msg_for_provider,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        structured_output_fields = kw.pop("structured_output_fields", None)

        if structured_output_fields:
            payload.update(structured_output_fields)
        elif output_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            if fail_on_unsupported:
                raise RuntimeError(
                    "Azure native json_schema not guaranteed; set fail_on_unsupported=False for best-effort"
                )
            payload["messages"] = _ensure_system_json_directive(messages, schema=json_schema)

        if tool_request is not None:
            payload["tools"] = _openai_like_tool_definitions(tool_request)
            payload["tool_choice"] = tool_request.choice
            payload["parallel_tool_calls"] = tool_request.max_calls > 1
        elif tools is not None:
            payload["tools"] = tools
        if tool_request is None and tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async def _call():
            r = await host._client.post(
                f"{host.base_url}/openai/deployments/{host.azure_deployment}/chat/completions?api-version=2024-08-01-preview",
                headers={"api-key": host.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            metadata = checked_response_metadata("azure", model, "chat", r)

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            if tool_request is not None:
                response = _openai_like_tool_call_response(
                    data,
                    tool_request=tool_request,
                    provider="azure",
                    model=model,
                    stable_messages=messages,
                    replay_messages=replay_messages,
                )
                return ProviderCallResult((response, usage), metadata)

            txt, _ = _first_text(data.get("choices", []))
            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    @staticmethod
    async def stream_chat_completions(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        on_delta: Any = None,
        on_usage_update: Any = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        """Stream one pinned Azure Chat Completions request.

        Intro:
            Uses Azure's deployment-scoped Chat Completions SSE route and requests
            the terminal usage chunk without switching to Azure Responses.

        Examples:
            Stream default text:
                ```python
                result = await AzureChatAdapter.stream_chat_completions(
                    client,
                    messages,
                    model="deployment-a",
                )
                ```

            Forward text deltas:
                ```python
                result = await AzureChatAdapter.stream_chat_completions(
                    client,
                    messages,
                    model="deployment-a",
                    max_output_tokens=256,
                    on_delta=on_delta,
                    on_usage_update=on_usage_update,
                )
                ```

        Args:
            host: Bound generic client owning the Azure transport.
            messages: Provider-projected stable conversation messages.
            model: Exact Azure deployment identity.
            reasoning_effort: Optional reasoning-depth override retained for the
                shared signature; the pinned legacy Azure route does not project it.
            max_output_tokens: Optional maximum generated tokens.
            on_delta: Optional async assistant-text callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded Azure sampling options.

        Returns:
            ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated text,
                terminal provider usage when received, and transport metadata.

        Notes:
            Azure may omit the final usage chunk when a stream is interrupted.
            Canonical terminal usage then remains explicitly unavailable.
        """

        await host._ensure_client()
        assert host._client is not None
        if not (host.base_url and host.azure_deployment):
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
            )
        if str(model or "").strip() != str(host.azure_deployment).strip():
            raise ValueError("Azure Chat Completions model must match the configured deployment")

        body: dict[str, Any] = {
            "messages": messages,
            "temperature": kw.get("temperature", 0.5),
            "top_p": kw.get("top_p", 1.0),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        url = (
            f"{host.base_url}/openai/deployments/{host.azure_deployment}"
            "/chat/completions?api-version=2024-08-01-preview"
        )
        return await _stream_openai_like_chat_completions(
            http_client=host._client,
            provider="azure",
            model=model,
            url=url,
            headers={"api-key": host.api_key, "Content-Type": "application/json"},
            body=body,
            on_delta=on_delta,
            on_usage_update=on_usage_update,
        )

    @staticmethod
    async def invoke_responses(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        tool_request: ToolCallRequest,
        prompt_cache_fields: dict[str, Any] | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[ToolCallResponse, dict[str, int]]]:
        """Run one Azure OpenAI Responses native Tool decision.

        The transport uses Azure's `/openai/v1/responses` route and API-key
        authentication while sharing the Responses Tool-search item protocol.

        Examples:
            Start client Tool search:
                ```python
                result = await AzureChatAdapter.invoke_responses(
                    client,
                    messages,
                    model="gpt-5.5",
                    reasoning_effort=None,
                    max_output_tokens=512,
                    tool_request=request,
                )
                ```

            Continue a pending search:
                ```python
                result = await AzureChatAdapter.invoke_responses(
                    client,
                    messages,
                    model="gpt-5.5",
                    reasoning_effort="medium",
                    max_output_tokens=512,
                    tool_request=continued_request,
                )
                ```

        Args:
            host: Bound generic client owning the Azure transport.
            messages: Provider-neutral conversation messages.
            model: Exact Azure deployment binding.
            reasoning_effort: Optional Responses reasoning effort.
            max_output_tokens: Optional maximum response tokens.
            tool_request: Exact native Tool and discovery request.
            prompt_cache_fields: Optional prepared Responses cache fields.
            **kw: Additional bounded transport options.

        Returns:
            ProviderCallResult[tuple[ToolCallResponse, dict[str, int]]]: Normalized
                ordered response and raw provider usage.

        Notes:
            Chat Completions remains the non-Tool Azure path and is never treated
            as discovery-compatible.
        """

        await host._ensure_client()
        assert host._client is not None
        if not host.base_url:
            raise RuntimeError("Azure OpenAI Responses requires AZURE_OPENAI_ENDPOINT")
        deployment = str(host.azure_deployment or model or "").strip()
        if not deployment or deployment != str(model or "").strip():
            raise ValueError(
                "Azure discovery requires the exact model and deployment binding to match"
            )
        body: dict[str, Any] = {
            "model": deployment,
            "input": _normalize_openai_responses_input(messages),
        }
        if prompt_cache_fields:
            body.update(prompt_cache_fields)
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        checkpoint_payload: dict[str, Any] | None = None
        if tool_request.transport_checkpoint is not None:
            checkpoint_payload = _openai_checkpoint_payload(
                tool_request.transport_checkpoint,
                provider="azure",
            )
            if checkpoint_payload["state"] == "pending_search":
                discovery_result = tool_request.discovery_result
                pending_search_call_id = str(
                    checkpoint_payload.get("call_id") or ""
                )
                if (
                    discovery_result is not None
                    and discovery_result.provider_reference_id
                    != pending_search_call_id
                ):
                    raise LLMToolCallResponseError(
                        code="discovery_result_reference_mismatch",
                        message="Azure discovery result does not match the pending search.",
                    )
                if discovery_result is not None and discovery_result.status == "failed":
                    raise LLMToolCallResponseError(
                        code="discovery_failure_output_unsupported",
                        message=(
                            "Azure Responses has no verified client Tool-search "
                            "failure continuation shape."
                        ),
                    )
                prior_active_names = {
                    str(name) for name in list(checkpoint_payload.get("active_tool_names") or [])
                }
                newly_active_names = set(tool_request.active_tool_names) - prior_active_names
                if discovery_result is not None:
                    newly_active_names = set(discovery_result.tool_names)
                loaded_tools = [
                    tool
                    for tool in tool_request.tools
                    if tool.exposure == "deferred" and tool.name in newly_active_names
                ]
                if not loaded_tools:
                    raise LLMToolCallResponseError(
                        code="discovery_result_missing",
                        message="Azure client Tool search has no newly activated result.",
                    )
                assert tool_request.discovery is not None
                if len(loaded_tools) > tool_request.discovery.max_results:
                    raise LLMToolCallResponseError(
                        code="discovery_result_limit_exceeded",
                        message="Azure client Tool-search results exceed the request bound.",
                    )
                previous_output = checkpoint_payload.get("response_output")
                if not isinstance(previous_output, list) or not previous_output:
                    raise ValueError("Azure Tool checkpoint output history is invalid")
                body["input"] = [
                    *[dict(item) for item in previous_output if isinstance(item, dict)],
                    {
                        "type": "tool_search_output",
                        "execution": "client",
                        "call_id": str(checkpoint_payload.get("call_id") or ""),
                        "status": "completed",
                        "tools": [
                            _openai_function_tool(tool, defer_loading=True) for tool in loaded_tools
                        ],
                    },
                ]
            elif checkpoint_payload["state"] == "pending_tool_outputs":
                previous_output = checkpoint_payload.get("response_output")
                if not isinstance(previous_output, list) or not previous_output:
                    raise ValueError("Azure Tool checkpoint output history is invalid")
                pending_call_ids = tuple(
                    str(call_id)
                    for call_id in list(checkpoint_payload.get("pending_call_ids") or [])
                )
                outputs_by_id = {item.call_id: item.output for item in tool_request.tool_outputs}
                missing = [call_id for call_id in pending_call_ids if call_id not in outputs_by_id]
                if missing:
                    raise LLMToolCallResponseError(
                        code="tool_output_missing",
                        message="Azure continuation is missing a completed Tool output.",
                    )
                body["input"] = [
                    *[dict(item) for item in previous_output if isinstance(item, dict)],
                    *[
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": outputs_by_id[call_id],
                        }
                        for call_id in pending_call_ids
                    ],
                ]
        if checkpoint_payload is not None and checkpoint_payload["state"] in {
            "pending_search",
            "pending_tool_outputs",
        }:
            body["tools"] = _openai_continuation_request_tools(tool_request)
        else:
            body["tools"] = _openai_request_tools(tool_request)
        # Azure Responses shares the exact-request Tool-surface contract: the
        # replay items do not retain request-owned immediate declarations.
        body["tool_choice"] = tool_request.choice
        body["parallel_tool_calls"] = tool_request.max_calls > 1

        request_timeout = kw.get("request_timeout_s") or kw.get("timeout")
        normalized_base = str(host.base_url).rstrip("/")
        url = (
            f"{normalized_base}/responses"
            if normalized_base.endswith("/openai/v1")
            else f"{normalized_base}/openai/v1/responses"
        )
        r = await host._client.post(
            url,
            headers={"api-key": host.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=request_timeout,
        )
        metadata = checked_response_metadata("azure", model, "chat", r)
        data = r.json()
        return ProviderCallResult(
            (
                _openai_tool_call_response(
                    data,
                    tool_request=tool_request,
                    model=model,
                    provider="azure",
                ),
                data.get("usage", {}) or {},
            ),
            metadata,
        )

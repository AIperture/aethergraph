"""OpenAI Responses API methods (chat + stream + image generation)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import hashlib
import json
from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.tool_calling import (
    LLMToolCallResponseError,
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)
from aethergraph.services.llm.tool_discovery import (
    ToolDiscoveryEvent,
    ToolTransportCheckpoint,
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    GeneratedImage,
    ImageGenerationResult,
    LLMStructuredOutputRefusalError,
    LLMStructuredOutputTruncationError,
)
from aethergraph.services.llm.utils import (
    _guess_mime_from_format,
    _normalize_base_url_no_trailing_slash,
    _normalize_openai_responses_input,
)

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]


def _openai_function_tool(
    tool: ToolDefinition,
    *,
    defer_loading: bool | None = None,
) -> dict[str, Any]:
    """Encode one provider-neutral Tool as an OpenAI function Tool.

    The encoding preserves the canonical schema and adds deferred loading only
    when the caller explicitly requests it.

    Examples:
        Encode an immediate function:
            ```python
            value = _openai_function_tool(tool)
            assert value["type"] == "function"
            ```

        Encode a deferred function:
            ```python
            value = _openai_function_tool(tool, defer_loading=True)
            assert value["defer_loading"] is True
            ```

    Args:
        tool: Validated provider-neutral Tool definition.
        defer_loading: Optional exact OpenAI deferred-loading flag.

    Returns:
        dict[str, Any]: Detached Responses API function Tool object.

    Notes:
        Provider-only fields never enter the shared Tool definition contract.
    """

    value: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": False,
    }
    if defer_loading is not None:
        value["defer_loading"] = bool(defer_loading)
    return value


def _openai_request_tools(
    request: ToolCallRequest,
) -> list[dict[str, Any]]:
    """Encode one OpenAI Tool array with exact discovery framing.

    Namespaced definitions remain grouped for native discovery, while client
    search is appended as the final stable Tool declaration.

    Examples:
        Encode ordinary Tool calling:
            ```python
            values = _openai_request_tools(request)
            assert values[0]["type"] == "function"
            ```

        Encode native client discovery:
            ```python
            values = _openai_request_tools(discovery_request)
            assert values[-1]["type"] == "tool_search"
            ```

    Args:
        request: Validated provider-neutral Tool-call request.

    Returns:
        list[dict[str, Any]]: Ordered Responses API Tool declarations.

    Notes:
        Native hosted discovery is not encoded until its result bound is bindable.
    """

    discovery_mode = request.discovery.mode if request.discovery is not None else None
    active_names = set(request.active_tool_names)
    grouped: dict[str, dict[str, Any]] = {}
    result: list[dict[str, Any]] = []
    for tool in request.tools:
        if (
            discovery_mode == "native_client"
            and tool.exposure == "deferred"
            and tool.name not in active_names
        ):
            continue
        deferred = (
            discovery_mode == "native_hosted"
            and tool.exposure == "deferred"
            and tool.name not in active_names
        )
        encoded = _openai_function_tool(
            tool,
            defer_loading=True if deferred else None,
        )
        if discovery_mode is None or tool.namespace is None:
            result.append(encoded)
            continue
        namespace = grouped.get(tool.namespace.name)
        if namespace is None:
            namespace = {
                "type": "namespace",
                "name": tool.namespace.name,
                "description": tool.namespace.description,
                "tools": [],
            }
            grouped[tool.namespace.name] = namespace
            result.append(namespace)
        namespace["tools"].append(encoded)
    if discovery_mode == "native_client":
        result.append(
            {
                "type": "tool_search",
                "execution": "client",
                "description": "Find project tools needed to continue the task.",
                "parameters": {
                    "type": "object",
                    "properties": {"goal": {"type": "string"}},
                    "required": ["goal"],
                    "additionalProperties": False,
                },
            }
        )
    elif discovery_mode == "native_hosted":
        raise LLMToolCallResponseError(
            code="unsupported_discovery_mode",
            message="OpenAI hosted Tool search is not bound to an enforceable result limit.",
        )
    return result


def _openai_checkpoint(
    *,
    request: ToolCallRequest,
    model: str,
    response_id: str,
    state: str,
    call_id: str = "",
    provider: str = "openai",
    response_output: list[dict[str, Any]] | None = None,
    pending_call_ids: list[str] | None = None,
) -> ToolTransportCheckpoint:
    """Build one integrity-bound OpenAI Tool-continuation checkpoint.

    Only bounded replay facts are retained; raw response history and Tool
    schemas remain outside the checkpoint payload.

    Examples:
        Preserve a pending client search:
            ```python
            checkpoint = _openai_checkpoint(
                request=request,
                model="gpt-5.6",
                response_id="resp_1",
                state="pending_search",
                call_id="search_1",
            )
            assert checkpoint.revision == 1
            ```

        Await one function result:
            ```python
            checkpoint = _openai_checkpoint(
                request=continued_request,
                model="gpt-5.6",
                response_id="resp_2",
                state="pending_tool_outputs",
                pending_call_ids=["call_1"],
            )
            assert checkpoint.revision == 2
            ```

    Args:
        request: Exact same-turn discovery request.
        model: Exact Responses model binding.
        response_id: Provider response identity for continuation.
        state: Private replay state, pending_search or pending_tool_outputs.
        call_id: Pending client Tool-search call identity, when present.
        provider: Exact Responses transport provider identifier.
        response_output: Optional exact output replay required by the provider.
        pending_call_ids: Provider call identities awaiting Engine results.

    Returns:
        ToolTransportCheckpoint: Bounded latest same-turn replay checkpoint.

    Notes:
        The integrity digest covers the complete canonical opaque payload.
    """

    previous = request.transport_checkpoint
    revision = 1 if previous is None else previous.revision + 1
    payload = {
        "state": state,
        "response_id": response_id,
        "call_id": call_id,
        "active_tool_names": list(request.active_tool_names),
        "pending_call_ids": list(pending_call_ids or []),
    }
    if response_output is not None:
        payload["response_output"] = list(response_output)
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ToolTransportCheckpoint(
        checkpoint_id=f"{provider}_{revision}_{digest[:16]}",
        revision=revision,
        provider=provider,
        model=model,
        contract_version="responses.tool_search",
        turn_id=str(request.turn_id or ""),
        integrity_digest=digest,
        opaque_payload=payload,
    )


def _openai_checkpoint_payload(
    checkpoint: ToolTransportCheckpoint,
    *,
    provider: str = "openai",
) -> dict[str, Any]:
    """Validate and return one private OpenAI checkpoint payload.

    Adapter replay fails closed when provider, contract, payload, or digest
    identity differs from the exact Responses Tool-search contract.

    Examples:
        Restore a valid payload:
            ```python
            payload = _openai_checkpoint_payload(checkpoint)
            assert payload["state"] == "pending_search"
            ```

        Reject a foreign contract:
            ```python
            try:
                _openai_checkpoint_payload(foreign_checkpoint)
            except ValueError:
                pass
            ```

    Args:
        checkpoint: Candidate same-turn provider replay checkpoint.
        provider: Exact Responses transport provider identifier.

    Returns:
        dict[str, Any]: Detached validated private replay mapping.

    Notes:
        Validation never includes private payload contents in an error message.
    """

    if checkpoint.provider != provider or checkpoint.contract_version != "responses.tool_search":
        raise ValueError("Responses Tool checkpoint binding does not match")
    payload = dict(checkpoint.opaque_payload or {})
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != checkpoint.integrity_digest:
        raise ValueError("OpenAI Tool checkpoint integrity validation failed")
    if payload.get("state") not in {
        "pending_search",
        "pending_tool_outputs",
        "consumed",
    }:
        raise ValueError("OpenAI Tool checkpoint state is invalid")
    if payload["state"] == "pending_search" and (
        not str(payload.get("response_id") or "").strip()
        or not str(payload.get("call_id") or "").strip()
    ):
        raise ValueError("OpenAI pending Tool checkpoint identity is invalid")
    pending_call_ids = payload.get("pending_call_ids", [])
    if not isinstance(pending_call_ids, list) or not all(
        isinstance(call_id, str) and call_id.strip() for call_id in pending_call_ids
    ):
        raise ValueError("OpenAI Tool checkpoint pending-call state is invalid")
    if len(pending_call_ids) != len(set(pending_call_ids)):
        raise ValueError("OpenAI Tool checkpoint pending-call identities are not unique")
    if payload["state"] == "pending_tool_outputs" and not pending_call_ids:
        raise ValueError("OpenAI pending Tool-output checkpoint has no calls")
    active_names = payload.get("active_tool_names")
    if not isinstance(active_names, list) or not all(
        isinstance(name, str) and name.strip() for name in active_names
    ):
        raise ValueError("OpenAI Tool checkpoint activation state is invalid")
    return payload


def _openai_tool_call_response(
    data: dict[str, Any],
    *,
    tool_request: ToolCallRequest,
    model: str,
    provider: str = "openai",
) -> ToolCallResponse:
    """Normalize ordered OpenAI Tool and discovery response items.

    Client Tool-search calls become provider-neutral discovery events and
    function calls retain their exact provider order and call identity.

    Examples:
        Normalize a function call:
            ```python
            response = _openai_tool_call_response(data, tool_request=request, model=model)
            assert response.calls[0].call_id
            ```

        Normalize a client search:
            ```python
            response = _openai_tool_call_response(search_data, tool_request=request, model=model)
            assert response.discovery_events[0].source == "provider_client"
            ```

    Args:
        data: Detached OpenAI Responses payload.
        tool_request: Exact request used for this provider decision.
        model: Exact Responses model binding.
        provider: Exact Responses transport provider identifier.

    Returns:
        ToolCallResponse: Ordered provider-neutral items and latest checkpoint.

    Notes:
        Private replay state is carried only by `transport_checkpoint`.
    """

    items: list[ToolDiscoveryEvent | ToolCall] = []
    text_parts: list[str] = []
    search_call_ids: list[str] = []
    function_call_ids: list[str] = []
    for output_index, item in enumerate(list(data.get("output") or [])):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "tool_search_call":
            discovery = tool_request.discovery
            execution = str(item.get("execution") or "")
            if discovery is None or discovery.mode != "native_client" or execution != "client":
                raise LLMToolCallResponseError(
                    code="discovery_mode_mismatch",
                    message="OpenAI returned Tool search outside native client mode.",
                )
            call_id = str(item.get("call_id") or "").strip()
            if not call_id:
                raise LLMToolCallResponseError(
                    code="discovery_reference_missing",
                    message="OpenAI client Tool search omitted its call id.",
                )
            arguments = _openai_tool_arguments(
                item.get("arguments"),
                call_name="tool_search",
            )
            query = str(arguments.get("goal") or arguments.get("query") or "").strip()
            items.append(
                ToolDiscoveryEvent(
                    event_id=str(item.get("id") or call_id),
                    mode="native_client",
                    source="provider_client",
                    arguments=arguments,
                    query=query or None,
                    provider_reference_ids=(call_id,),
                )
            )
            search_call_ids.append(call_id)
            continue
        if item.get("type") == "function_call":
            arguments = _openai_tool_arguments(
                item.get("arguments"),
                call_name=str(item.get("name") or ""),
            )
            function_call_id = str(
                item.get("call_id") or item.get("id") or f"openai-call-{output_index}"
            )
            items.append(
                ToolCall(
                    call_id=function_call_id,
                    name=str(item.get("name") or ""),
                    arguments=arguments,
                    provider_metadata={
                        "item_id": str(item.get("id") or ""),
                        "status": str(item.get("status") or ""),
                        "output_index": output_index,
                    },
                )
            )
            function_call_ids.append(function_call_id)
            continue
        if item.get("type") != "message":
            continue
        for part in list(item.get("content") or []):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                raise LLMToolCallResponseError(
                    code="refused",
                    message=str(part.get("refusal") or "OpenAI refused Tool selection."),
                )
            if "text" in part:
                text_parts.append(str(part.get("text") or ""))
    if len(search_call_ids) > 1:
        raise LLMToolCallResponseError(
            code="discovery_cardinality_invalid",
            message="OpenAI returned more than one pending client Tool search.",
        )
    if search_call_ids and function_call_ids:
        raise LLMToolCallResponseError(
            code="discovery_order_invalid",
            message="OpenAI returned Tool calls before completing client Tool search.",
        )
    checkpoint: ToolTransportCheckpoint | None = None
    response_id = str(data.get("id") or "").strip()
    if search_call_ids:
        if not response_id:
            raise LLMToolCallResponseError(
                code="discovery_reference_missing",
                message="OpenAI client Tool search omitted its response id.",
            )
        checkpoint = _openai_checkpoint(
            request=tool_request,
            model=model,
            response_id=response_id,
            state="pending_search",
            call_id=search_call_ids[0],
            provider=provider,
            response_output=(
                [dict(item) for item in list(data.get("output") or []) if isinstance(item, dict)]
                if provider == "azure"
                else None
            ),
        )
    elif function_call_ids and provider == "openai" and tool_request.discovery is not None:
        if not response_id:
            raise LLMToolCallResponseError(
                code="tool_call_reference_missing",
                message="OpenAI Tool calls omitted their response id.",
            )
        checkpoint = _openai_checkpoint(
            request=tool_request,
            model=model,
            response_id=response_id,
            state="pending_tool_outputs",
            provider=provider,
            pending_call_ids=function_call_ids,
        )
    elif tool_request.transport_checkpoint is not None:
        prior_payload = _openai_checkpoint_payload(
            tool_request.transport_checkpoint,
            provider=provider,
        )
        if prior_payload["state"] == "pending_search":
            checkpoint = _openai_checkpoint(
                request=tool_request,
                model=model,
                response_id=response_id,
                state="consumed",
                provider=provider,
            )
    return ToolCallResponse(
        items=tuple(items),
        text="".join(text_parts),
        finish_reason=str(data.get("status") or ""),
        provider_metadata={
            "response_id": response_id,
            "output_item_count": len(list(data.get("output") or [])),
        },
        transport_checkpoint=checkpoint,
    )


def _openai_tool_arguments(value: Any, *, call_name: str) -> dict[str, Any]:
    """Decode one OpenAI function-call argument object."""

    if isinstance(value, dict):
        return dict(value)
    try:
        decoded = json.loads(str(value or ""))
    except json.JSONDecodeError as exc:
        raise LLMToolCallResponseError(
            code="invalid_arguments",
            message=(f"OpenAI Tool call '{call_name or '?'}' returned invalid JSON arguments."),
        ) from exc
    if not isinstance(decoded, dict):
        raise LLMToolCallResponseError(
            code="invalid_arguments",
            message=(
                f"OpenAI Tool call '{call_name or '?'}' arguments must decode " "to one object."
            ),
        )
    return decoded


class _OpenAIMixin:
    """Provider methods for OpenAI Responses API."""

    # ------------------------------------------------------------------
    # Chat – non-streaming
    # ------------------------------------------------------------------
    async def _chat_openai_responses(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        tool_request: ToolCallRequest | None = None,
        prompt_cache_fields: dict[str, Any] | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        await self._ensure_client()
        assert self._client is not None

        url = f"{self.base_url}/responses"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        input_messages = _normalize_openai_responses_input(messages)

        body: dict[str, Any] = {"model": model, "input": input_messages}
        structured_output_fields = kw.pop("structured_output_fields", None)
        if tool_request is not None and (
            structured_output_fields
            or output_format in {"json_object", "json_schema"}
            or tools is not None
            or tool_choice is not None
        ):
            raise ValueError(
                "Native Tool calling cannot be combined with structured output "
                "or legacy tools/tool_choice arguments"
            )
        if prompt_cache_fields:
            body.update(prompt_cache_fields)

        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        # Structured output
        if structured_output_fields:
            body.update(structured_output_fields)
        elif output_format == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif output_format == "json_schema":
            if json_schema is None:
                raise ValueError("output_format='json_schema' requires json_schema")
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": bool(strict_schema),
                }
            }

        checkpoint_payload: dict[str, Any] | None = None
        if tool_request is not None and tool_request.transport_checkpoint is not None:
            checkpoint_payload = _openai_checkpoint_payload(tool_request.transport_checkpoint)
            if checkpoint_payload["state"] == "pending_search":
                prior_active_names = {
                    str(name) for name in list(checkpoint_payload.get("active_tool_names") or [])
                }
                newly_active_names = set(tool_request.active_tool_names) - prior_active_names
                loaded_tools = [
                    tool
                    for tool in tool_request.tools
                    if tool.exposure == "deferred" and tool.name in newly_active_names
                ]
                if not loaded_tools:
                    raise LLMToolCallResponseError(
                        code="discovery_result_missing",
                        message="OpenAI client Tool search has no newly activated result.",
                    )
                assert tool_request.discovery is not None
                if len(loaded_tools) > tool_request.discovery.max_results:
                    raise LLMToolCallResponseError(
                        code="discovery_result_limit_exceeded",
                        message="OpenAI client Tool-search results exceed the request bound.",
                    )
                body["previous_response_id"] = str(checkpoint_payload.get("response_id") or "")
                body["input"] = [
                    {
                        "type": "tool_search_output",
                        "execution": "client",
                        "call_id": str(checkpoint_payload.get("call_id") or ""),
                        "status": "completed",
                        "tools": [
                            _openai_function_tool(tool, defer_loading=True) for tool in loaded_tools
                        ],
                    }
                ]
            elif checkpoint_payload["state"] == "pending_tool_outputs":
                pending_call_ids = tuple(
                    str(call_id)
                    for call_id in list(checkpoint_payload.get("pending_call_ids") or [])
                )
                outputs_by_id = {item.call_id: item.output for item in tool_request.tool_outputs}
                missing = [call_id for call_id in pending_call_ids if call_id not in outputs_by_id]
                if missing:
                    raise LLMToolCallResponseError(
                        code="tool_output_missing",
                        message="OpenAI continuation is missing a completed Tool output.",
                    )
                body["previous_response_id"] = str(checkpoint_payload.get("response_id") or "")
                body["input"] = [
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": outputs_by_id[call_id],
                    }
                    for call_id in pending_call_ids
                ]

        # Tools (Responses API style)
        if tool_request is not None:
            if checkpoint_payload is None or checkpoint_payload["state"] == "consumed":
                body["tools"] = _openai_request_tools(tool_request)
                body["tool_choice"] = tool_request.choice
                body["parallel_tool_calls"] = tool_request.max_calls > 1
        elif tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        request_timeout = kw.get("request_timeout_s")
        if request_timeout is None:
            request_timeout = kw.get("timeout")
        if request_timeout is None and max_output_tokens is not None and max_output_tokens >= 2048:
            request_timeout = max(float(self._timeout), 180.0)

        async def _call():
            r = await self._client.post(
                url,
                headers=headers,
                json=body,
                timeout=request_timeout,
            )
            metadata = checked_response_metadata("openai", model, "chat", r)

            data = r.json()
            usage = data.get("usage", {}) or {}
            if data.get("status") == "incomplete":
                detail = data.get("incomplete_details") or {}
                if tool_request is not None:
                    raise LLMToolCallResponseError(
                        code="truncated",
                        message=f"OpenAI Tool-call response was incomplete: {detail}",
                    )
                if structured_output_fields:
                    raise LLMStructuredOutputTruncationError(
                        "OpenAI structured response was incomplete: " f"{detail}"
                    )

            # If caller asked for raw provider payload, just return it as a JSON string
            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            # Existing parsing logic for message-only flows
            output = data.get("output")
            if tool_request is not None:
                return ProviderCallResult(
                    (
                        _openai_tool_call_response(
                            data,
                            tool_request=tool_request,
                            model=model,
                        ),
                        usage,
                    ),
                    metadata,
                )
            txt = ""

            if isinstance(output, list) and output:
                chunks: list[str] = []
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        parts = item.get("content") or []
                        for p in parts:
                            if isinstance(p, dict) and p.get("type") == "refusal":
                                raise LLMStructuredOutputRefusalError(
                                    str(p.get("refusal") or "OpenAI refused the request.")
                                )
                            if isinstance(p, dict) and "text" in p:
                                chunks.append(p["text"])
                txt = "".join(chunks)

            elif isinstance(output, dict) and output.get("type") == "message":
                msg = output.get("message") or output
                parts = msg.get("content") or []
                chunks: list[str] = []
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        chunks.append(p["text"])
                txt = "".join(chunks)

            elif isinstance(output, str):
                txt = output
            else:
                txt = ""

            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    # ------------------------------------------------------------------
    # Chat – streaming
    # ------------------------------------------------------------------
    async def _chat_openai_responses_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None,
        reasoning_summary: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
        fail_on_unsupported: bool,
        on_delta: DeltaCallback | None = None,
        on_thinking_delta: ThinkingDeltaCallback | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        """
        Stream text using OpenAI Responses API.

        Handles ``response.output_text.delta`` for content and
        ``response.reasoning_summary_text.delta`` for thinking/reasoning summaries.
        """
        await self._ensure_client()
        assert self._client is not None

        url = f"{self.base_url}/responses"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        input_messages = _normalize_openai_responses_input(messages)

        body: dict[str, Any] = {
            "model": model,
            "input": input_messages,
            "stream": True,
        }

        reasoning_cfg: dict[str, Any] = {}
        if reasoning_effort is not None:
            reasoning_cfg["effort"] = reasoning_effort
        if reasoning_summary is not None:
            reasoning_cfg["summary"] = reasoning_summary
        if reasoning_cfg:
            body["reasoning"] = reasoning_cfg
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        # Structured output config (same as non-streaming path)
        if output_format == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif output_format == "json_schema":
            if json_schema is None:
                raise ValueError("output_format='json_schema' requires json_schema")
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": bool(strict_schema),
                }
            }
        # else: default "text" format

        full_chunks: list[str] = []
        thinking_chunks: list[str] = []
        usage: dict[str, int] = {}

        async def _handle_event(evt: dict[str, Any]):
            nonlocal usage

            etype = evt.get("type")

            # Reasoning summary deltas (thinking tokens)
            if etype == "response.reasoning_summary_text.delta":
                delta = evt.get("delta") or ""
                if delta:
                    thinking_chunks.append(delta)
                    if on_thinking_delta is not None:
                        await on_thinking_delta(delta)

            # Main text deltas
            elif etype == "response.output_text.delta":
                delta = evt.get("delta") or ""
                if delta:
                    full_chunks.append(delta)
                    if on_delta is not None:
                        await on_delta(delta)

            # Finalization – grab usage from completed response if present
            elif etype in ("response.completed", "response.incomplete", "response.failed"):
                resp = evt.get("response") or {}
                # Usage may or may not be present, keep best-effort
                usage = resp.get("usage") or usage

            # Optional: basic error surface
            elif etype == "error":
                # in practice `error` may be structured differently; this is just a guardrail
                msg = evt.get("message") or "Unknown streaming error"
                raise RuntimeError(f"OpenAI streaming error: {msg}")

        async def _call():
            async with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=body,
            ) as r:
                if r.is_error:
                    await r.aread()
                metadata = checked_response_metadata("openai", model, "chat_stream", r)

                # SSE: each event line is "data: {...}" + blank lines between events
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if not data_str or data_str == "[DONE]":
                        # OpenAI ends stream with `data: [DONE]`
                        break

                    try:
                        evt = json.loads(data_str)
                    except Exception:
                        # best-effort: ignore malformed chunks
                        continue

                    await _handle_event(evt)

                return metadata

        metadata = await _call()
        return ProviderCallResult(("".join(full_chunks), usage), metadata)

    # ------------------------------------------------------------------
    # Image generation
    # ------------------------------------------------------------------
    async def _image_openai_generate(
        self,
        prompt: str,
        *,
        model: str,
        n: int,
        size: str | None,
        quality: str | None,
        style: str | None,
        output_format: Any | None,
        response_format: Any | None,
        background: str | None,
        **kw: Any,
    ) -> ProviderCallResult[ImageGenerationResult]:
        assert self._client is not None

        url = f"{_normalize_base_url_no_trailing_slash(self.base_url)}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
        }
        if size is not None:
            body["size"] = size
        if quality is not None:
            body["quality"] = quality
        if style is not None:
            body["style"] = style
        if output_format is not None:
            body["output_format"] = output_format
        if background is not None:
            body["background"] = background

        if response_format is not None:
            body["response_format"] = response_format

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            metadata = checked_response_metadata("openai", model, "image", r)

            data = r.json()
            imgs: list[GeneratedImage] = []
            for item in data.get("data", []) or []:
                imgs.append(
                    GeneratedImage(
                        b64=item.get("b64_json"),
                        url=item.get("url"),
                        mime_type=_guess_mime_from_format(output_format or "png")
                        if item.get("b64_json")
                        else None,
                        revised_prompt=item.get("revised_prompt"),
                    )
                )

            return ProviderCallResult(
                ImageGenerationResult(images=imgs, usage=data.get("usage", {}) or {}, raw=data),
                metadata,
            )

        return await _call()

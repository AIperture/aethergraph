"""Physical OpenAI Responses Chat adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import hashlib
import json
from typing import Any

from aethergraph.services.llm._tool_discovery_manifest import (
    render_tool_search_description,
)
from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.tool_calling import (
    AssistantOutput,
    LLMToolCallResponseError,
    ModelToolSpec,
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    assistant_output_identity,
)
from aethergraph.services.llm.tool_discovery import (
    ToolDiscoveryEvent,
    ToolTransportCheckpoint,
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    LLMStructuredOutputRefusalError,
    LLMStructuredOutputTruncationError,
)
from aethergraph.services.llm.utils import (
    _normalize_openai_responses_input,
)

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]


def _openai_function_tool(
    tool: ModelToolSpec,
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
        if discovery_mode is None or tool.path is None:
            result.append(encoded)
            continue
        namespace_name = _openai_namespace_name(tool.path.path)
        namespace = grouped.get(namespace_name)
        if namespace is None:
            namespace = {
                "type": "namespace",
                "name": namespace_name,
                "description": tool.path.description,
                "tools": [],
            }
            grouped[namespace_name] = namespace
            result.append(namespace)
        namespace["tools"].append(encoded)
    if discovery_mode == "native_client":
        if request.discovery is None or request.discovery.search_schema is None:
            raise LLMToolCallResponseError(
                code="tool_search_schema_missing",
                message="OpenAI client Tool search requires an Engine-authored schema.",
            )
        result.append(
            {
                "type": "tool_search",
                "execution": "client",
                "description": render_tool_search_description(request),
                "parameters": request.discovery.search_schema,
            }
        )
    elif discovery_mode == "native_hosted":
        result.append(
            {
                "type": "tool_search",
                "description": render_tool_search_description(request),
            }
        )
    return result


def _openai_hosted_tool_refs(item: dict[str, Any]) -> tuple[str, ...]:
    """Decode callable names selected by one hosted Tool-search output."""

    raw_results = item.get("tools")
    if raw_results is None:
        raw_results = item.get("results")
    if not isinstance(raw_results, list):
        return ()
    names: list[str] = []
    pending = list(raw_results)
    while pending:
        value = pending.pop(0)
        if isinstance(value, str):
            name = value.strip()
        elif isinstance(value, dict):
            name = str(value.get("name") or value.get("tool_name") or "").strip()
            nested = value.get("tools")
            if isinstance(nested, list):
                pending[0:0] = nested
                name = ""
        else:
            name = ""
        if name and name not in names:
            names.append(name)
    return tuple(names)


def _openai_namespace_name(path: str) -> str:
    """Project a Tool path to one reversible OpenAI-safe namespace name."""

    encoded_segments: list[str] = []
    for segment in str(path or "").split("."):
        encoded_segments.append(segment.replace("_", "_u").replace("-", "_h"))
    name = "tp_" + "_d".join(encoded_segments)
    if len(name) > 64:
        raise LLMToolCallResponseError(
            code="tool_path_projection_invalid",
            message=(f"Tool path {path!r} exceeds the OpenAI namespace projection limit."),
        )
    return name


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
    prompt_stable_message_count: int | None = None,
    prompt_stable_prefix_digest: str | None = None,
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
        prompt_stable_message_count: Number of stable prompt messages already
            represented by the response.
        prompt_stable_prefix_digest: Integrity digest of those stable messages.

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
    if prompt_stable_message_count is not None or prompt_stable_prefix_digest is not None:
        if prompt_stable_message_count is None or prompt_stable_prefix_digest is None:
            raise ValueError("OpenAI prompt continuation metadata is incomplete")
        payload["prompt_stable_message_count"] = prompt_stable_message_count
        payload["prompt_stable_prefix_digest"] = prompt_stable_prefix_digest
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
    prompt_count = payload.get("prompt_stable_message_count")
    prompt_digest = payload.get("prompt_stable_prefix_digest")
    if (prompt_count is None) != (prompt_digest is None):
        raise ValueError("OpenAI Tool checkpoint prompt state is incomplete")
    if prompt_count is not None:
        if isinstance(prompt_count, bool) or not isinstance(prompt_count, int) or prompt_count <= 0:
            raise ValueError("OpenAI Tool checkpoint prompt count is invalid")
        if not isinstance(prompt_digest, str) or len(prompt_digest) != 64:
            raise ValueError("OpenAI Tool checkpoint prompt digest is invalid")
    return payload


def _openai_prompt_prefix_digest(
    input_messages: list[dict[str, Any]],
    stable_message_count: int,
) -> str:
    """Return the canonical digest of one provider-visible stable prompt prefix."""

    if stable_message_count <= 0 or stable_message_count > len(input_messages):
        raise ValueError("OpenAI stable prompt message count is outside the request input")
    canonical = json.dumps(
        input_messages[:stable_message_count],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _openai_appended_prompt_input(
    input_messages: list[dict[str, Any]],
    *,
    stable_message_count: int | None,
    checkpoint_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return every prompt message appended after the prior stable prefix."""

    prior_count = checkpoint_payload.get("prompt_stable_message_count")
    prior_digest = checkpoint_payload.get("prompt_stable_prefix_digest")
    if prior_count is None:
        if stable_message_count is not None:
            raise LLMToolCallResponseError(
                code="prompt_continuation_state_missing",
                message="OpenAI continuation checkpoint has no stable prompt state.",
            )
        return []
    if stable_message_count is None:
        raise LLMToolCallResponseError(
            code="prompt_continuation_state_missing",
            message="OpenAI continuation request has no stable prompt state.",
        )
    if stable_message_count < prior_count or prior_count > len(input_messages):
        raise LLMToolCallResponseError(
            code="prompt_continuation_diverged",
            message="OpenAI continuation prompt no longer extends its stable prefix.",
        )
    current_prior_digest = _openai_prompt_prefix_digest(input_messages, prior_count)
    if current_prior_digest != prior_digest:
        raise LLMToolCallResponseError(
            code="prompt_continuation_diverged",
            message="OpenAI continuation prompt changed inside its stable prefix.",
        )
    return input_messages[prior_count:]


def _openai_tool_call_response(
    data: dict[str, Any],
    *,
    tool_request: ToolCallRequest,
    model: str,
    provider: str = "openai",
    prompt_stable_message_count: int | None = None,
    prompt_stable_prefix_digest: str | None = None,
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

    items: list[AssistantOutput | ToolDiscoveryEvent | ToolCall] = []
    search_call_ids: list[str] = []
    function_call_ids: list[str] = []
    response_id = str(data.get("id") or "").strip()
    output_items = list(data.get("output") or [])
    hosted_outputs = [
        value
        for value in output_items
        if isinstance(value, dict)
        and value.get("type") == "tool_search_output"
        and value.get("execution") == "server"
    ]
    hosted_output_index = 0
    for output_index, item in enumerate(output_items):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "tool_search_call":
            discovery = tool_request.discovery
            execution = str(item.get("execution") or "")
            if execution == "client":
                expected_mode = "native_client"
            elif execution == "server":
                expected_mode = "native_hosted"
            else:
                raise LLMToolCallResponseError(
                    code="discovery_execution_unsupported",
                    message=(
                        "OpenAI returned Tool search with unsupported execution " f"{execution!r}."
                    ),
                )
            if discovery is None or discovery.mode != expected_mode:
                raise LLMToolCallResponseError(
                    code="discovery_mode_mismatch",
                    message="OpenAI returned Tool search outside the configured mode.",
                )
            call_id = str(item.get("call_id") or "").strip()
            if expected_mode == "native_client" and not call_id:
                raise LLMToolCallResponseError(
                    code="discovery_reference_missing",
                    message="OpenAI client Tool search omitted its call id.",
                )
            arguments = (
                _openai_tool_arguments(
                    item.get("arguments"),
                    call_name="tool_search",
                )
                if item.get("arguments") is not None
                else {}
            )
            query = str(arguments.get("goal") or arguments.get("query") or "").strip()
            tool_refs: tuple[str, ...] = ()
            if expected_mode == "native_hosted":
                if hosted_output_index >= len(hosted_outputs):
                    raise LLMToolCallResponseError(
                        code="discovery_output_missing",
                        message="OpenAI hosted Tool search omitted its search output.",
                    )
                tool_refs = _openai_hosted_tool_refs(hosted_outputs[hosted_output_index])
                hosted_output_index += 1
            items.append(
                ToolDiscoveryEvent(
                    event_id=str(
                        item.get("id")
                        or call_id
                        or f"{response_id or 'response'}:tool-search:{output_index}"
                    ),
                    mode=expected_mode,
                    source=(
                        "provider_client" if expected_mode == "native_client" else "provider_hosted"
                    ),
                    arguments=arguments,
                    query=query or None,
                    tool_refs=tool_refs,
                    provider_reference_ids=((call_id,) if call_id else ()),
                )
            )
            if expected_mode == "native_client":
                search_call_ids.append(call_id)
            continue
        if item.get("type") == "tool_search_output":
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
        message_id = str(item.get("id") or f"openai-message-{output_index}")
        for part_index, part in enumerate(list(item.get("content") or [])):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                refusal_text = str(part.get("refusal") or "OpenAI refused Tool selection.")
                items.append(
                    AssistantOutput(
                        output_id=assistant_output_identity(
                            provider="openai",
                            response_id=response_id,
                            provider_item_id=message_id,
                            item_index=output_index,
                            content_index=part_index,
                            text=refusal_text,
                        ),
                        text=refusal_text,
                        content_type="refusal",
                        provider_metadata={
                            "item_id": message_id,
                            "output_index": output_index,
                            "content_index": part_index,
                        },
                    )
                )
                continue
            if "text" in part:
                output_text = str(part.get("text") or "")
                items.append(
                    AssistantOutput(
                        output_id=assistant_output_identity(
                            provider="openai",
                            response_id=response_id,
                            provider_item_id=message_id,
                            item_index=output_index,
                            content_index=part_index,
                            text=output_text,
                        ),
                        text=output_text,
                        provider_metadata={
                            "item_id": message_id,
                            "output_index": output_index,
                            "content_index": part_index,
                        },
                    )
                )
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
            prompt_stable_message_count=prompt_stable_message_count,
            prompt_stable_prefix_digest=prompt_stable_prefix_digest,
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
            prompt_stable_message_count=prompt_stable_message_count,
            prompt_stable_prefix_digest=prompt_stable_prefix_digest,
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
                prompt_stable_message_count=prompt_stable_message_count,
                prompt_stable_prefix_digest=prompt_stable_prefix_digest,
            )
    return ToolCallResponse(
        items=tuple(items),
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
            message=(f"OpenAI Tool call '{call_name or '?'}' arguments must decode to one object."),
        )
    return decoded


class OpenAIResponsesAdapter:
    """Physical adapter for the OpenAI Responses Chat endpoint."""

    # ------------------------------------------------------------------
    # Chat – non-streaming
    # ------------------------------------------------------------------
    @staticmethod
    async def invoke(
        host: Any,
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
        prompt_cache_stable_message_count: int | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Invoke one OpenAI Responses request.

        Intro:
            Projects structured output, prompt caching, native Tools, discovery,
            and same-turn continuation into one exact Responses request.

        Examples:
            Send a direct text request:
                ```python
                result = await OpenAIResponsesAdapter.invoke(
                    client,
                    messages,
                    model="gpt-test",
                    reasoning_effort=None,
                    max_output_tokens=256,
                    output_format="text",
                    json_schema=None,
                    schema_name="Response",
                    strict_schema=True,
                )
                ```

            Continue a native Tool request:
                ```python
                result = await OpenAIResponsesAdapter.invoke(
                    client,
                    messages,
                    model="gpt-test",
                    reasoning_effort="medium",
                    max_output_tokens=256,
                    output_format="text",
                    json_schema=None,
                    schema_name="Response",
                    strict_schema=True,
                    tool_request=continued_request,
                )
                ```

        Args:
            host: Bound generic client owning the OpenAI transport.
            messages: Provider-projected stable conversation messages.
            model: Exact configured OpenAI model identity.
            reasoning_effort: Optional normalized reasoning-depth override.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text, structured, or raw output mode.
            json_schema: Optional canonical JSON schema.
            schema_name: Stable structured-output schema name.
            strict_schema: Whether native schema enforcement is strict.
            tools: Optional legacy Responses Tool declarations.
            tool_choice: Optional legacy Tool-selection policy.
            tool_request: Optional canonical native Tool request and continuation.
            prompt_cache_fields: Optional prepared native cache fields.
            prompt_cache_stable_message_count: Optional stable prompt-prefix size.
            **kw: Additional bounded OpenAI request options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Normalized response, raw usage, and transport metadata.

        Notes:
            Retry, quota accounting, metering, and observations remain owned by
            the shared invocation lifecycle. This adapter makes one attempt.
        """

        await host._ensure_client()
        assert host._client is not None

        url = f"{host.base_url}/responses"
        headers = {"Authorization": f"Bearer {host.api_key}", "Content-Type": "application/json"}

        input_messages = _normalize_openai_responses_input(messages)
        prompt_stable_prefix_digest = (
            _openai_prompt_prefix_digest(
                input_messages,
                prompt_cache_stable_message_count,
            )
            if prompt_cache_stable_message_count is not None
            else None
        )

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
            appended_prompt_input = _openai_appended_prompt_input(
                input_messages,
                stable_message_count=prompt_cache_stable_message_count,
                checkpoint_payload=checkpoint_payload,
            )
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
                    },
                    *appended_prompt_input,
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
                ] + appended_prompt_input

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
            request_timeout = max(float(host._timeout), 180.0)

        async def _call():
            r = await host._client.post(
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
                        f"OpenAI structured response was incomplete: {detail}"
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
                            prompt_stable_message_count=prompt_cache_stable_message_count,
                            prompt_stable_prefix_digest=prompt_stable_prefix_digest,
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
    @staticmethod
    async def stream(
        host: Any,
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
        on_usage_update: Callable[[dict[str, int]], Awaitable[None]] | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        """Stream text and reasoning summaries through OpenAI Responses.

        Intro:
            Parses Responses SSE events into assistant, reasoning-summary, and
            cumulative usage callbacks while retaining one terminal raw receipt.

        Examples:
            Stream assistant text:
                ```python
                result = await OpenAIResponsesAdapter.stream(
                    client,
                    messages,
                    model="gpt-test",
                    reasoning_effort=None,
                    reasoning_summary=None,
                    max_output_tokens=256,
                    output_format="text",
                    json_schema=None,
                    schema_name="Response",
                    strict_schema=True,
                    fail_on_unsupported=True,
                    on_delta=on_delta,
                )
                ```

            Observe cumulative usage:
                ```python
                result = await OpenAIResponsesAdapter.stream(
                    client,
                    messages,
                    model="gpt-test",
                    reasoning_effort="medium",
                    reasoning_summary="auto",
                    max_output_tokens=256,
                    output_format="text",
                    json_schema=None,
                    schema_name="Response",
                    strict_schema=True,
                    fail_on_unsupported=True,
                    on_usage_update=on_usage_update,
                )
                ```

        Args:
            host: Bound generic client owning the OpenAI transport.
            messages: Provider-projected stable conversation messages.
            model: Exact configured OpenAI model identity.
            reasoning_effort: Optional normalized reasoning-depth override.
            reasoning_summary: Optional displayable reasoning-summary mode.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text or structured output mode.
            json_schema: Optional canonical JSON schema.
            schema_name: Stable structured-output schema name.
            strict_schema: Whether native schema enforcement is strict.
            fail_on_unsupported: Whether unsupported native features must fail.
            on_delta: Optional async assistant-text callback.
            on_thinking_delta: Optional async reasoning-summary callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded OpenAI request options.

        Returns:
            ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated text,
                terminal provider usage, and transport metadata.

        Notes:
            Usage callbacks are cumulative observations. Shared accounting and
            metering consume only the returned terminal receipt.
        """
        await host._ensure_client()
        assert host._client is not None

        url = f"{host.base_url}/responses"
        headers = {"Authorization": f"Bearer {host.api_key}", "Content-Type": "application/json"}

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
            """Apply one parsed OpenAI Responses stream event.

            Intro:
                Routes one provider event to its exact text, reasoning, usage,
                completion, or error behavior without owning transport retries.

            Examples:
                Apply a text delta:
                    ```python
                    await _handle_event(
                        {"type": "response.output_text.delta", "delta": "Hi"}
                    )
                    ```

                Apply terminal usage:
                    ```python
                    await _handle_event(
                        {
                            "type": "response.completed",
                            "response": {"usage": {"input_tokens": 1}},
                        }
                    )
                    ```

            Args:
                evt: Parsed provider Responses event.

            Returns:
                None: Completes after routing the event.

            Notes:
                Malformed transport frames are filtered by the surrounding SSE
                loop before this semantic dispatcher is called.
            """

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
                if usage and on_usage_update is not None:
                    await on_usage_update(dict(usage))

            # Optional: basic error surface
            elif etype == "error":
                # in practice `error` may be structured differently; this is just a guardrail
                msg = evt.get("message") or "Unknown streaming error"
                raise RuntimeError(f"OpenAI streaming error: {msg}")

        async def _call():
            async with host._client.stream(
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

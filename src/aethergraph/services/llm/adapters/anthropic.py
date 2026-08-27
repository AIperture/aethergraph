"""Physical Anthropic Messages Chat adapter."""

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
    LLMToolCallCapabilityError,
    LLMToolCallResponseError,
    ModelToolSpec,
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    assistant_output_identity,
)
from aethergraph.services.llm.tool_discovery import (
    ToolDiscoveryError,
    ToolDiscoveryEvent,
    ToolTransportCheckpoint,
)
from aethergraph.services.llm.types import ChatOutputFormat, LLMUnsupportedFeatureError
from aethergraph.services.llm.utils import _to_anthropic_blocks

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]


def _anthropic_function_tool(tool: ModelToolSpec) -> dict[str, Any]:
    """Encode one shared Tool definition for Anthropic Messages.

    Deferred Tools retain their complete schema in the physical request while
    `defer_loading` keeps them out of the rendered prompt prefix.

    Examples:
        Encode an immediate Tool:
            ```python
            value = _anthropic_function_tool(immediate)
            assert "defer_loading" not in value
            ```

        Encode a deferred Tool:
            ```python
            value = _anthropic_function_tool(deferred)
            assert value["defer_loading"] is True
            ```

    Args:
        tool: Validated provider-neutral Tool definition.

    Returns:
        dict[str, Any]: Detached Anthropic Tool definition.

    Notes:
        Deferred Tool definitions never receive `cache_control`.
    """

    value: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }
    if tool.exposure == "deferred":
        value["defer_loading"] = True
    return value


def _anthropic_request_tools(request: ToolCallRequest) -> list[dict[str, Any]]:
    """Encode one stable Anthropic discovery Tool array.

    The selected search Tool remains non-deferred and every catalog Tool keeps
    the same physical definition across activation cycles.

    Examples:
        Encode hosted BM25 discovery:
            ```python
            values = _anthropic_request_tools(hosted_request)
            assert values[0]["type"].startswith("tool_search_tool_bm25")
            ```

        Encode custom client discovery:
            ```python
            values = _anthropic_request_tools(client_request)
            assert values[0]["name"] == "tool_search"
            ```

    Args:
        request: Exact provider-neutral Tool-call request.

    Returns:
        list[dict[str, Any]]: Stable ordered Messages API Tool array.

    Notes:
        At least the selected search Tool is always non-deferred.
    """

    discovery = request.discovery
    result: list[dict[str, Any]] = []
    if discovery is not None and discovery.mode == "native_hosted":
        result.append(
            {
                "type": "tool_search_tool_bm25_20251119",
                "name": "tool_search_tool_bm25",
            }
        )
    elif discovery is not None and discovery.mode == "native_client":
        if discovery.search_schema is None:
            raise LLMToolCallResponseError(
                code="tool_search_schema_missing",
                message="Anthropic client Tool search requires an Engine-authored schema.",
            )
        result.append(
            {
                "name": "tool_search",
                "description": render_tool_search_description(request),
                "input_schema": discovery.search_schema,
            }
        )
    result.extend(_anthropic_function_tool(tool) for tool in request.tools)
    return result


def _anthropic_checkpoint(
    *,
    request: ToolCallRequest,
    model: str,
    state: str,
    assistant_content: list[dict[str, Any]] | None = None,
    search_call_id: str = "",
    pending_call_ids: list[str] | None = None,
) -> ToolTransportCheckpoint:
    """Build one integrity-bound Anthropic full-history checkpoint.

    Pending client search retains the assistant blocks unchanged so the next
    request can append exact `tool_reference` results.

    Examples:
        Preserve a pending custom search:
            ```python
            checkpoint = _anthropic_checkpoint(
                request=request,
                model=model,
                state="pending_search",
                assistant_content=blocks,
                search_call_id="toolu_1",
            )
            assert checkpoint.revision == 1
            ```

        Mark replay as consumed:
            ```python
            checkpoint = _anthropic_checkpoint(
                request=continued_request,
                model=model,
                state="consumed",
            )
            assert checkpoint.revision == 2
            ```

    Args:
        request: Exact same-turn discovery request.
        model: Exact Anthropic model binding.
        state: Private replay state: pending_search, pending_tool_outputs, or consumed.
        assistant_content: Exact prior assistant blocks for pending replay.
        search_call_id: Exact custom search Tool-use identity.
        pending_call_ids: Exact ordinary Tool-use identities awaiting results.

    Returns:
        ToolTransportCheckpoint: Bounded latest full-history checkpoint.

    Notes:
        Raw assistant blocks remain private and are never rendered in observations.
    """

    previous = request.transport_checkpoint
    revision = 1 if previous is None else previous.revision + 1
    payload = {
        "state": state,
        "assistant_content": list(assistant_content or []),
        "search_call_id": search_call_id,
        "pending_call_ids": list(pending_call_ids or []),
        "active_tool_names": list(request.active_tool_names),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ToolTransportCheckpoint(
        checkpoint_id=f"anthropic_{revision}_{digest[:16]}",
        revision=revision,
        provider="anthropic",
        model=model,
        contract_version="messages.tool_reference",
        turn_id=str(request.turn_id or ""),
        integrity_digest=digest,
        purpose={
            "pending_search": "pending_discovery_result",
            "pending_tool_outputs": "pending_tool_outputs",
            "consumed": "consumed",
        }[state],
        opaque_payload=payload,
    )


def _anthropic_checkpoint_payload(
    checkpoint: ToolTransportCheckpoint,
) -> dict[str, Any]:
    """Validate and return one private Anthropic replay payload.

    Replay rejects foreign contracts, modified assistant history, and incomplete
    pending identities before any provider traffic occurs.

    Examples:
        Restore valid pending history:
            ```python
            payload = _anthropic_checkpoint_payload(checkpoint)
            assert payload["state"] == "pending_search"
            ```

        Reject a modified payload:
            ```python
            try:
                _anthropic_checkpoint_payload(modified_checkpoint)
            except ValueError:
                pass
            ```

    Args:
        checkpoint: Candidate Anthropic same-turn replay checkpoint.

    Returns:
        dict[str, Any]: Detached validated private replay mapping.

    Notes:
        Error messages never include private assistant content.
    """

    if (
        checkpoint.provider != "anthropic"
        or checkpoint.contract_version != "messages.tool_reference"
    ):
        raise ValueError("Anthropic Tool checkpoint binding does not match")
    payload = dict(checkpoint.opaque_payload or {})
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != checkpoint.integrity_digest:
        raise ValueError("Anthropic Tool checkpoint integrity validation failed")
    if payload.get("state") not in {
        "pending_search",
        "pending_tool_outputs",
        "consumed",
    }:
        raise ValueError("Anthropic Tool checkpoint state is invalid")
    expected_purpose = {
        "pending_search": "pending_discovery_result",
        "pending_tool_outputs": "pending_tool_outputs",
        "consumed": "consumed",
    }[payload["state"]]
    if checkpoint.purpose != expected_purpose:
        raise ValueError("Anthropic Tool checkpoint purpose does not match replay state")
    if not isinstance(payload.get("active_tool_names"), list):
        raise ValueError("Anthropic Tool checkpoint activation state is invalid")
    if payload["state"] == "pending_search" and (
        not str(payload.get("search_call_id") or "").strip()
        or not isinstance(payload.get("assistant_content"), list)
        or not payload["assistant_content"]
    ):
        raise ValueError("Anthropic pending Tool checkpoint identity is invalid")
    pending_call_ids = payload.get("pending_call_ids", [])
    if not isinstance(pending_call_ids, list) or not all(
        isinstance(call_id, str) and call_id.strip() for call_id in pending_call_ids
    ):
        raise ValueError("Anthropic Tool checkpoint pending-call state is invalid")
    if len(pending_call_ids) != len(set(pending_call_ids)):
        raise ValueError("Anthropic Tool checkpoint pending-call identities are not unique")
    if payload["state"] == "pending_tool_outputs" and (
        not pending_call_ids
        or not isinstance(payload.get("assistant_content"), list)
        or not payload["assistant_content"]
    ):
        raise ValueError("Anthropic pending Tool-output checkpoint is invalid")
    return payload


def _anthropic_tool_call_response(
    data: dict[str, Any],
    *,
    tool_request: ToolCallRequest,
    model: str,
) -> ToolCallResponse:
    """Normalize ordered Anthropic discovery and Tool-use blocks.

    Hosted search results become exact referenced discovery events; custom
    client search calls become Engine-resolved events with private replay state.

    Examples:
        Normalize hosted discovery and a call:
            ```python
            response = _anthropic_tool_call_response(data, tool_request=request, model=model)
            assert response.discovery_events
            ```

        Normalize a custom client search:
            ```python
            response = _anthropic_tool_call_response(client_data, tool_request=request, model=model)
            assert response.transport_checkpoint is not None
            ```

    Args:
        data: Detached Anthropic Messages response payload.
        tool_request: Exact request used for this decision.
        model: Exact Anthropic model binding.

    Returns:
        ToolCallResponse: Ordered provider-neutral items and private checkpoint.

    Notes:
        Server search blocks never become executable Engine Tool calls.
    """

    items: list[AssistantOutput | ToolDiscoveryEvent | ToolCall] = []
    blocks = list(data.get("content") or [])
    response_id = str(data.get("id") or "").strip()
    hosted_calls: dict[str, dict[str, Any]] = {}
    client_search_ids: list[str] = []
    ordinary_call_ids: list[str] = []
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            output_text = str(block.get("text") or "")
            provider_item_id = str(block.get("id") or "")
            items.append(
                AssistantOutput(
                    output_id=assistant_output_identity(
                        provider="anthropic",
                        response_id=response_id,
                        provider_item_id=provider_item_id,
                        item_index=block_index,
                        text=output_text,
                    ),
                    text=output_text,
                    provider_metadata={
                        "provider_item_id": provider_item_id,
                        "content_block_index": block_index,
                    },
                )
            )
            continue
        block_type = block.get("type")
        if block_type == "server_tool_use":
            if (
                tool_request.discovery is None
                or tool_request.discovery.mode != "native_hosted"
                or block.get("name") != "tool_search_tool_bm25"
            ):
                raise LLMToolCallResponseError(
                    code="discovery_mode_mismatch",
                    message="Anthropic returned server Tool search outside hosted mode.",
                )
            server_id = str(block.get("id") or "").strip()
            arguments = block.get("input")
            if not server_id or not isinstance(arguments, dict):
                raise LLMToolCallResponseError(
                    code="discovery_reference_missing",
                    message="Anthropic hosted Tool search omitted its identity or input.",
                )
            hosted_calls[server_id] = dict(arguments)
            continue
        if block_type == "tool_search_tool_result":
            server_id = str(block.get("tool_use_id") or "").strip()
            arguments = hosted_calls.pop(server_id, None)
            if arguments is None:
                raise LLMToolCallResponseError(
                    code="discovery_reference_missing",
                    message="Anthropic Tool-search result has no matching server call.",
                )
            content = block.get("content")
            if not isinstance(content, dict):
                raise LLMToolCallResponseError(
                    code="invalid_discovery_response",
                    message="Anthropic Tool-search result content must be an object.",
                )
            if content.get("type") == "tool_search_tool_result_error":
                items.append(
                    ToolDiscoveryEvent(
                        event_id=server_id,
                        mode="native_hosted",
                        source="provider_hosted",
                        arguments=arguments,
                        query=str(arguments.get("query") or "") or None,
                        status="failed",
                        error=ToolDiscoveryError(
                            code=str(content.get("error_code") or "search_failed"),
                            summary=str(
                                content.get("error_message") or "Anthropic Tool search failed."
                            ),
                            retryable=content.get("error_code")
                            in {"unavailable", "too_many_requests"},
                        ),
                        provider_reference_ids=(server_id,),
                    )
                )
                continue
            references = content.get("tool_references")
            if not isinstance(references, list):
                raise LLMToolCallResponseError(
                    code="invalid_discovery_response",
                    message="Anthropic Tool-search references must be an array.",
                )
            tool_refs = tuple(
                str(reference.get("tool_name") or "").strip()
                for reference in references
                if isinstance(reference, dict)
                and reference.get("type") == "tool_reference"
                and str(reference.get("tool_name") or "").strip()
            )
            items.append(
                ToolDiscoveryEvent(
                    event_id=server_id,
                    mode="native_hosted",
                    source="provider_hosted",
                    arguments=arguments,
                    query=str(arguments.get("query") or "") or None,
                    tool_refs=tool_refs,
                    provider_reference_ids=(server_id,),
                )
            )
            continue
        if block_type != "tool_use":
            continue
        arguments = block.get("input")
        if not isinstance(arguments, dict):
            raise LLMToolCallResponseError(
                code="invalid_arguments",
                message=(
                    f"Anthropic Tool call '{block.get('name') or '?'}' input must be an object."
                ),
            )
        call_id = str(block.get("id") or f"anthropic-call-{block_index}")
        if (
            tool_request.discovery is not None
            and tool_request.discovery.mode == "native_client"
            and block.get("name") == "tool_search"
        ):
            items.append(
                ToolDiscoveryEvent(
                    event_id=call_id,
                    mode="native_client",
                    source="provider_client",
                    arguments=dict(arguments),
                    query=str(arguments.get("query") or arguments.get("goal") or "") or None,
                    provider_reference_ids=(call_id,),
                )
            )
            client_search_ids.append(call_id)
            continue
        items.append(
            ToolCall(
                call_id=call_id,
                name=str(block.get("name") or ""),
                arguments=dict(arguments),
                provider_metadata={"content_block_index": block_index},
            )
        )
        ordinary_call_ids.append(call_id)
    if hosted_calls:
        raise LLMToolCallResponseError(
            code="invalid_discovery_response",
            message="Anthropic hosted Tool search omitted its result block.",
        )
    if len(client_search_ids) > 1:
        raise LLMToolCallResponseError(
            code="discovery_cardinality_invalid",
            message="Anthropic returned more than one pending client Tool search.",
        )
    if client_search_ids and ordinary_call_ids:
        raise LLMToolCallResponseError(
            code="discovery_order_invalid",
            message="Anthropic returned Tool calls before completing client Tool search.",
        )
    checkpoint: ToolTransportCheckpoint | None = None
    if client_search_ids:
        checkpoint = _anthropic_checkpoint(
            request=tool_request,
            model=model,
            state="pending_search",
            assistant_content=[dict(block) for block in blocks if isinstance(block, dict)],
            search_call_id=client_search_ids[0],
        )
    elif ordinary_call_ids and str(tool_request.turn_id or "").strip():
        checkpoint = _anthropic_checkpoint(
            request=tool_request,
            model=model,
            state="pending_tool_outputs",
            assistant_content=[dict(block) for block in blocks if isinstance(block, dict)],
            pending_call_ids=ordinary_call_ids,
        )
    elif tool_request.transport_checkpoint is not None:
        prior = _anthropic_checkpoint_payload(tool_request.transport_checkpoint)
        if prior["state"] in {"pending_search", "pending_tool_outputs"}:
            checkpoint = _anthropic_checkpoint(
                request=tool_request,
                model=model,
                state="consumed",
            )
    return ToolCallResponse(
        items=tuple(items),
        finish_reason=str(data.get("stop_reason") or ""),
        provider_metadata={
            "response_id": response_id,
            "content_block_count": len(blocks),
        },
        transport_checkpoint=checkpoint,
    )


def _anthropic_system_payload(
    messages: list[dict[str, Any]],
    *,
    output_format: ChatOutputFormat,
) -> str | list[dict[str, Any]] | None:
    directive = (
        "Return ONLY valid JSON. No markdown, no commentary."
        if output_format == "json_object"
        else ""
    )
    system_messages = [m for m in list(messages or []) if m.get("role") == "system"]
    if not directive and not system_messages:
        return None
    if not any(_message_has_cache_control(m) for m in system_messages):
        sys_msgs = [directive] if directive else []
        for message in system_messages:
            content = message.get("content")
            sys_msgs.append(content if isinstance(content, str) else str(content))
        return "\n\n".join(sys_msgs) if sys_msgs else None

    blocks: list[dict[str, Any]] = []
    if directive:
        blocks.append({"type": "text", "text": directive})
    for message in system_messages:
        blocks.extend(_anthropic_content_blocks(message))
    return blocks or None


def _anthropic_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = _to_anthropic_blocks(message.get("content"))
    cache_control = _anthropic_cache_control(message.get("cache_control"))
    if cache_control:
        if not blocks:
            blocks = [{"type": "text", "text": ""}]
        last = dict(blocks[-1])
        existing = _anthropic_cache_control(last.get("cache_control"))
        if existing is not None and existing != cache_control:
            raise ValueError(
                "conflicting Anthropic cache_control on message and final content block"
            )
        last["cache_control"] = cache_control
        blocks[-1] = last
    return blocks


def _anthropic_cache_control(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("Anthropic cache_control must be a dict")
    if not value:
        return None
    return dict(value)


def _message_has_cache_control(message: dict[str, Any]) -> bool:
    if _anthropic_cache_control(message.get("cache_control")):
        return True
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and _anthropic_cache_control(item.get("cache_control"))
        for item in content
    )


def _validate_anthropic_cache_breakpoints(payload: dict[str, Any]) -> None:
    count = 1 if _anthropic_cache_control(payload.get("cache_control")) else 0
    system = payload.get("system")
    if isinstance(system, list):
        count += _count_cache_control_blocks(system)
    for message in list(payload.get("messages") or []):
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            count += _count_cache_control_blocks(content)
    if count > 4:
        raise ValueError(
            f"Anthropic prompt caching supports at most 4 cache breakpoints; got {count}"
        )


def _count_cache_control_blocks(blocks: list[Any]) -> int:
    return sum(
        1
        for block in blocks
        if isinstance(block, dict) and _anthropic_cache_control(block.get("cache_control"))
    )


class AnthropicMessagesAdapter:
    """Physical adapter for the Anthropic Messages Chat endpoint."""

    # ------------------------------------------------------------------
    # Chat – non-streaming
    # ------------------------------------------------------------------
    @staticmethod
    async def invoke(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
        thinking_mode: str | None = None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_request: ToolCallRequest | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Invoke one Anthropic Messages request.

        Intro:
            Projects system blocks, thinking, prompt caching, structured output,
            native Tools, discovery, and continuation into one Messages request.

        Examples:
            Send a direct text request:
                ```python
                result = await AnthropicMessagesAdapter.invoke(
                    client,
                    messages,
                    model="claude-test",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=True,
                )
                ```

            Continue native Tool discovery:
                ```python
                result = await AnthropicMessagesAdapter.invoke(
                    client,
                    messages,
                    model="claude-test",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=True,
                    tool_request=continued_request,
                )
                ```

        Args:
            host: Bound generic client owning the Anthropic transport.
            messages: Provider-projected stable conversation messages.
            model: Exact configured Anthropic model identity.
            reasoning_effort: Optional adaptive reasoning-depth override.
            max_output_tokens: Optional maximum generated tokens.
            thinking_budget: Optional extended-thinking token budget.
            thinking_mode: Optional explicit thinking on/off mode.
            output_format: Requested text, structured, or raw output mode.
            json_schema: Optional canonical JSON schema.
            fail_on_unsupported: Whether unsupported native fields must fail.
            tools: Optional legacy Tool declarations, rejected rather than dropped.
            tool_request: Optional canonical native Tool request and continuation.
            **kw: Additional bounded Anthropic sampling and cache options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Normalized response, raw usage, and transport metadata.

        Notes:
            Retry, quota accounting, metering, and observations remain owned by
            the shared invocation lifecycle. This adapter makes one attempt.
        """

        await host._ensure_client()
        assert host._client is not None

        if tools is not None:
            raise LLMUnsupportedFeatureError(
                host.provider,
                model,
                "provider-neutral tools",
                "Anthropic tool translation is not wired yet; refusing to drop tools silently.",
            )
        structured_output_fields = kw.pop("structured_output_fields", None)
        if tool_request is not None and (
            structured_output_fields or output_format in {"json_object", "json_schema"}
        ):
            raise ValueError("Native Tool calling cannot be combined with structured output")

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        system_payload = _anthropic_system_payload(messages, output_format=output_format)

        # Convert messages to Anthropic format (blocks)
        conv: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            anthro_role = "assistant" if role == "assistant" else "user"
            content_blocks = _anthropic_content_blocks(m)
            conv.append({"role": anthro_role, "content": content_blocks})

        checkpoint_payload: dict[str, Any] | None = None
        if tool_request is not None and tool_request.transport_checkpoint is not None:
            checkpoint_payload = _anthropic_checkpoint_payload(tool_request.transport_checkpoint)
            if checkpoint_payload["state"] == "pending_search":
                prior_active_names = {
                    str(name) for name in list(checkpoint_payload.get("active_tool_names") or [])
                }
                newly_active_names = tuple(
                    name
                    for name in tool_request.active_tool_names
                    if name not in prior_active_names
                )
                if not newly_active_names:
                    raise LLMToolCallResponseError(
                        code="discovery_result_missing",
                        message="Anthropic client Tool search has no newly activated result.",
                    )
                assert tool_request.discovery is not None
                if len(newly_active_names) > tool_request.discovery.max_results:
                    raise LLMToolCallResponseError(
                        code="discovery_result_limit_exceeded",
                        message="Anthropic client Tool-search results exceed the request bound.",
                    )
                conv.append(
                    {
                        "role": "assistant",
                        "content": list(checkpoint_payload["assistant_content"]),
                    }
                )
                conv.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(checkpoint_payload.get("search_call_id") or ""),
                                "content": [
                                    {
                                        "type": "tool_reference",
                                        "tool_name": name,
                                    }
                                    for name in newly_active_names
                                ],
                            }
                        ],
                    }
                )
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
                        message="Anthropic continuation is missing a completed Tool output.",
                    )
                conv.append(
                    {
                        "role": "assistant",
                        "content": list(checkpoint_payload["assistant_content"]),
                    }
                )
                conv.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": outputs_by_id[call_id],
                            }
                            for call_id in pending_call_ids
                        ],
                    }
                )

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens or kw.get("max_tokens", 1024),
            "messages": conv,
            "temperature": temperature,
            "top_p": top_p,
        }
        request_cache_control = _anthropic_cache_control(kw.get("cache_control"))
        if request_cache_control:
            payload["cache_control"] = request_cache_control
        if system_payload:
            payload["system"] = system_payload
        if structured_output_fields:
            payload.update(structured_output_fields)
        elif output_format == "json_schema":
            if json_schema is None:
                raise ValueError("output_format='json_schema' requires json_schema")
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": json_schema,
                }
            }
        if thinking_mode == "off":
            pass
        elif reasoning_effort is not None:
            payload["thinking"] = {"type": "adaptive", "effort": reasoning_effort}
        elif thinking_mode == "on":
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget or 4096}
        if tool_request is not None:
            payload["tools"] = _anthropic_request_tools(tool_request)
            choice_type = {
                "auto": "auto",
                "required": "any",
                "none": "none",
            }[tool_request.choice]
            if choice_type == "any" and (reasoning_effort is not None or thinking_mode == "on"):
                raise LLMToolCallCapabilityError(
                    provider="anthropic",
                    model=model,
                    feature="required_tool_choice_with_thinking",
                )
            payload["tool_choice"] = {
                "type": choice_type,
                "disable_parallel_tool_use": tool_request.max_calls == 1,
            }
        _validate_anthropic_cache_breakpoints(payload)

        async def _call():
            r = await host._client.post(
                f"{host.base_url}/v1/messages",
                headers={
                    "x-api-key": host.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            metadata = checked_response_metadata("anthropic", model, "chat", r)

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            if tool_request is not None:
                if data.get("stop_reason") == "max_tokens":
                    raise LLMToolCallResponseError(
                        code="truncated",
                        message=(
                            "Anthropic stopped at max_tokens before completing "
                            "native Tool selection."
                        ),
                    )
                return ProviderCallResult(
                    (
                        _anthropic_tool_call_response(
                            data,
                            tool_request=tool_request,
                            model=model,
                        ),
                        usage,
                    ),
                    metadata,
                )

            blocks = data.get("content") or []
            txt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    # ------------------------------------------------------------------
    # Chat – streaming (with extended thinking support)
    # ------------------------------------------------------------------
    @staticmethod
    async def stream(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        thinking_budget: int | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        on_delta: DeltaCallback | None = None,
        on_thinking_delta: ThinkingDeltaCallback | None = None,
        on_usage_update: Callable[[dict[str, int]], Awaitable[None]] | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        """Stream text and thinking summaries through Anthropic Messages.

        Intro:
            Parses Messages SSE blocks into assistant, displayable thinking, and
            cumulative usage callbacks while preserving cache-related counters.

        Examples:
            Stream assistant text:
                ```python
                result = await AnthropicMessagesAdapter.stream(
                    client,
                    messages,
                    model="claude-test",
                    thinking_budget=None,
                    max_output_tokens=256,
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=True,
                    on_delta=on_delta,
                )
                ```

            Observe thinking and usage:
                ```python
                result = await AnthropicMessagesAdapter.stream(
                    client,
                    messages,
                    model="claude-test",
                    thinking_budget=1024,
                    max_output_tokens=256,
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=True,
                    on_thinking_delta=on_thinking_delta,
                    on_usage_update=on_usage_update,
                )
                ```

        Args:
            host: Bound generic client owning the Anthropic transport.
            messages: Provider-projected stable conversation messages.
            model: Exact configured Anthropic model identity.
            thinking_budget: Optional extended-thinking token budget.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text or structured output mode.
            json_schema: Optional canonical JSON schema.
            fail_on_unsupported: Whether unsupported native features must fail.
            on_delta: Optional async assistant-text callback.
            on_thinking_delta: Optional async thinking-summary callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded Anthropic sampling and cache options.

        Returns:
            ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated text,
                cumulative provider usage, and transport metadata.

        Notes:
            Usage callbacks are cumulative observations. Shared accounting and
            metering consume only the returned terminal receipt.
        """
        await host._ensure_client()
        assert host._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        system_payload = _anthropic_system_payload(messages, output_format=output_format)

        # Convert messages to Anthropic format (blocks)
        conv: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            anthro_role = "assistant" if role == "assistant" else "user"
            content_blocks = _anthropic_content_blocks(m)
            conv.append({"role": anthro_role, "content": content_blocks})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens or kw.get("max_tokens", 4096),
            "messages": conv,
            "stream": True,
        }

        if output_format == "json_schema":
            raise RuntimeError(
                "Anthropic json_schema streaming is intentionally unsupported; use chat() instead."
            )

        if thinking_budget is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        elif kw.get("reasoning_effort") is not None:
            payload["thinking"] = {"type": "adaptive", "effort": kw.get("reasoning_effort")}
        else:
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        request_cache_control = _anthropic_cache_control(kw.get("cache_control"))
        if request_cache_control:
            payload["cache_control"] = request_cache_control
        if system_payload:
            payload["system"] = system_payload
        _validate_anthropic_cache_breakpoints(payload)

        headers: dict[str, str] = {
            "x-api-key": host.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if thinking_budget is not None:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        text_chunks: list[str] = []
        usage: dict[str, int] = {}

        async def _call():
            nonlocal usage

            async with host._client.stream(
                "POST",
                f"{host.base_url}/v1/messages",
                headers=headers,
                json=payload,
            ) as r:
                if r.is_error:
                    await r.aread()
                metadata = checked_response_metadata("anthropic", model, "chat_stream", r)

                # Anthropic SSE uses two-line format: "event: <type>\ndata: <json>"
                pending_event_type: str | None = None

                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue

                    # Parse event type line
                    if line.startswith("event:"):
                        pending_event_type = line[len("event:") :].strip()
                        continue

                    # Parse data line
                    if line.startswith("data:"):
                        data_str = line[len("data:") :].strip()
                        if not data_str:
                            continue

                        try:
                            data = json.loads(data_str)
                        except Exception:
                            continue

                        event_type = pending_event_type or data.get("type", "")
                        pending_event_type = None

                        await _handle_sse_event(event_type, data)

                return metadata

        async def _handle_sse_event(event_type: str, data: dict[str, Any]):
            """Apply one parsed Anthropic Messages stream event.

            Intro:
                Routes message usage, text, thinking, and error events while
                retaining one cumulative raw usage receipt.

            Examples:
                Apply starting input usage:
                    ```python
                    await _handle_sse_event(
                        "message_start",
                        {"message": {"usage": {"input_tokens": 2}}},
                    )
                    ```

                Apply an output usage update:
                    ```python
                    await _handle_sse_event(
                        "message_delta",
                        {"usage": {"output_tokens": 1}},
                    )
                    ```

            Args:
                event_type: Exact Anthropic SSE event name.
                data: Parsed event payload.

            Returns:
                None: Completes after routing the event.

            Notes:
                Signature deltas remain provider integrity material and are not
                exposed as displayable reasoning text.
            """

            nonlocal usage

            if event_type == "message_start":
                msg = data.get("message", {})
                msg_usage = msg.get("usage", {})
                if msg_usage:
                    usage.update(msg_usage)
                    if on_usage_update is not None:
                        await on_usage_update(dict(usage))

            elif event_type == "content_block_delta":
                delta = data.get("delta", {})
                delta_type = delta.get("type", "")

                if delta_type == "thinking_delta":
                    chunk = delta.get("thinking", "")
                    if chunk and on_thinking_delta is not None:
                        await on_thinking_delta(chunk)

                elif delta_type == "text_delta":
                    chunk = delta.get("text", "")
                    if chunk:
                        text_chunks.append(chunk)
                        if on_delta is not None:
                            await on_delta(chunk)

                # signature_delta: ignore (integrity check for thinking blocks)

            elif event_type == "message_delta":
                delta_usage = data.get("usage", {})
                if delta_usage:
                    usage.update(delta_usage)
                    if on_usage_update is not None:
                        await on_usage_update(dict(usage))

            # content_block_start, content_block_stop, message_stop, ping: no action needed

            elif event_type == "error":
                err = data.get("error", {})
                msg = err.get("message", "Unknown Anthropic streaming error")
                raise RuntimeError(f"Anthropic streaming error: {msg}")

        metadata = await _call()
        return ProviderCallResult(("".join(text_chunks), usage), metadata)

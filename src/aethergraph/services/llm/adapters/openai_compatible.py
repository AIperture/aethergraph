"""Physical OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.tool_calling import (
    AssistantOutput,
    LLMToolCallResponseError,
    ToolCall,
    ToolCallRequest,
    ToolCallResponse,
    assistant_output_identity,
    tool_call_request_fingerprint,
)
from aethergraph.services.llm.tool_discovery import ToolTransportCheckpoint
from aethergraph.services.llm.types import ChatOutputFormat
from aethergraph.services.llm.utils import _ensure_system_json_directive


def _first_text(choices):
    """Extract text and usage from OpenAI-style choices list."""
    if not choices:
        return "", {}
    c = choices[0]
    text = (c.get("message", {}) or {}).get("content") or c.get("text") or ""
    usage = {}
    return text, usage


def _openai_like_tool_call_response(
    data: dict[str, Any],
    *,
    tool_request: ToolCallRequest,
    provider: str,
    model: str,
    stable_messages: list[dict[str, Any]],
    replay_messages: tuple[dict[str, Any], ...] = (),
) -> ToolCallResponse:
    """Normalize one OpenAI-compatible Chat Completions Tool response.

    Intro:
        The normalizer preserves ordered assistant output and Tool calls. When
        the request has a semantic turn identity, pending calls also produce an
        integrity-bound private checkpoint containing exact replay messages.

    Examples:
        Normalize an initial Tool call:
            ```python
            response = _openai_like_tool_call_response(
                payload,
                tool_request=request,
                provider="openrouter",
                model="openai/gpt-test",
                stable_messages=messages,
            )
            ```

        Normalize a continued Tool decision:
            ```python
            response = _openai_like_tool_call_response(
                payload,
                tool_request=continued_request,
                provider="ollama",
                model="local-model",
                stable_messages=messages,
                replay_messages=prior_replay,
            )
            ```

    Args:
        data: Parsed Chat Completions response payload.
        tool_request: Exact provider-neutral Tool request.
        provider: Exact registered provider identity.
        model: Exact configured model identity.
        stable_messages: Canonical prompt messages bound to the checkpoint.
        replay_messages: Validated same-turn replay messages sent in this call.

    Returns:
        ToolCallResponse: Ordered normalized items and optional continuation.

    Notes:
        Private replay messages are omitted from observations and public response
        metadata. A direct completion has no continuation checkpoint.
    """

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMToolCallResponseError(
            code="missing_choice",
            message=f"{provider} Tool-call response omitted its first choice.",
        )
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LLMToolCallResponseError(
            code="missing_message",
            message=f"{provider} Tool-call response omitted its assistant message.",
        )

    response_id = str(data.get("id") or "")
    choice_index = int(choice.get("index") or 0)
    items: list[AssistantOutput | ToolCall] = []
    content = message.get("content")
    if content is not None:
        if not isinstance(content, str):
            raise LLMToolCallResponseError(
                code="invalid_assistant_content",
                message=f"{provider} assistant content must be text or null in Tool mode.",
            )
        if content:
            items.append(
                AssistantOutput(
                    output_id=assistant_output_identity(
                        provider=provider,
                        response_id=response_id,
                        provider_item_id=str(message.get("id") or ""),
                        item_index=choice_index,
                        content_index=0,
                        text=content,
                    ),
                    text=content,
                    provider_metadata={
                        "choice_index": choice_index,
                        "role": str(message.get("role") or "assistant"),
                    },
                )
            )
    refusal = message.get("refusal")
    if refusal is not None:
        refusal_text = str(refusal)
        if refusal_text:
            items.append(
                AssistantOutput(
                    output_id=assistant_output_identity(
                        provider=provider,
                        response_id=response_id,
                        provider_item_id=str(message.get("id") or ""),
                        item_index=choice_index,
                        content_index=1,
                        text=refusal_text,
                    ),
                    text=refusal_text,
                    content_type="refusal",
                    provider_metadata={"choice_index": choice_index},
                )
            )

    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise LLMToolCallResponseError(
            code="invalid_tool_calls",
            message=f"{provider} assistant tool_calls must be an array.",
        )
    if len(raw_calls) > tool_request.max_calls:
        raise LLMToolCallResponseError(
            code="tool_call_cardinality_exceeded",
            message=(
                f"{provider} returned {len(raw_calls)} Tool calls, exceeding the "
                f"request maximum of {tool_request.max_calls}."
            ),
        )
    allowed_names = {tool.name for tool in tool_request.tools}
    for call_index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict) or raw_call.get("type", "function") != "function":
            raise LLMToolCallResponseError(
                code="invalid_tool_call",
                message=f"{provider} returned a non-function Tool call.",
            )
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise LLMToolCallResponseError(
                code="invalid_tool_call",
                message=f"{provider} function Tool call omitted its function payload.",
            )
        name = str(function.get("name") or "").strip()
        if name not in allowed_names:
            raise LLMToolCallResponseError(
                code="unknown_tool",
                message=f"{provider} returned unknown Tool {name or '?'}.",
            )
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            decoded_arguments = dict(arguments)
        else:
            try:
                decoded_arguments = json.loads(str(arguments or ""))
            except json.JSONDecodeError as exc:
                raise LLMToolCallResponseError(
                    code="invalid_arguments",
                    message=f"{provider} Tool call {name!r} returned invalid JSON arguments.",
                ) from exc
        if not isinstance(decoded_arguments, dict):
            raise LLMToolCallResponseError(
                code="invalid_arguments",
                message=f"{provider} Tool call {name!r} arguments must be an object.",
            )
        call_id = str(raw_call.get("id") or "").strip()
        if not call_id:
            call_id = f"{response_id or provider}-call-{call_index}"
        items.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=decoded_arguments,
                provider_metadata={
                    "choice_index": choice_index,
                    "tool_call_index": call_index,
                },
            )
        )

    checkpoint = None
    if raw_calls and tool_request.turn_id:
        checkpoint = _openai_like_checkpoint(
            request=tool_request,
            provider=provider,
            model=model,
            stable_messages=stable_messages,
            replay_messages=(*replay_messages, dict(message)),
            pending_call_ids=tuple(call.call_id for call in items if isinstance(call, ToolCall)),
        )
    return ToolCallResponse(
        items=tuple(items),
        finish_reason=str(choice.get("finish_reason") or ""),
        provider_metadata={
            "response_id": response_id,
            "choice_index": choice_index,
            "tool_call_count": len(raw_calls),
        },
        transport_checkpoint=checkpoint,
    )


def _openai_like_checkpoint(
    *,
    request: ToolCallRequest,
    provider: str,
    model: str,
    stable_messages: list[dict[str, Any]],
    replay_messages: tuple[dict[str, Any], ...],
    pending_call_ids: tuple[str, ...],
) -> ToolTransportCheckpoint:
    """Build one integrity-bound Chat Completions continuation checkpoint.

    Intro:
        The checkpoint binds the exact provider, model, turn, Tool contract,
        stable prompt, replay transcript, and outstanding provider call IDs.

    Examples:
        Preserve the first Tool decision:
            ```python
            checkpoint = _openai_like_checkpoint(
                request=request,
                provider="openrouter",
                model="openai/gpt-test",
                stable_messages=messages,
                replay_messages=(assistant_message,),
                pending_call_ids=("call_1",),
            )
            ```

        Advance a continued decision:
            ```python
            checkpoint = _openai_like_checkpoint(
                request=continued_request,
                provider="ollama",
                model="local-model",
                stable_messages=messages,
                replay_messages=continued_replay,
                pending_call_ids=("call_2",),
            )
            assert checkpoint.revision == 2
            ```

    Args:
        request: Exact same-turn Tool request.
        provider: Exact registered provider identity.
        model: Exact configured model identity.
        stable_messages: Original prompt messages retained by the caller.
        replay_messages: Complete provider replay transcript after the response.
        pending_call_ids: Exact Tool calls awaiting Engine results.

    Returns:
        ToolTransportCheckpoint: Bounded opaque same-turn replay state.

    Notes:
        The checkpoint payload is private and its canonical digest is validated
        before any continuation request reaches a provider.
    """

    previous = request.transport_checkpoint
    revision = 1 if previous is None else previous.revision + 1
    prompt_digest = _openai_like_messages_digest(stable_messages)
    payload = {
        "state": "pending_tool_outputs",
        "tool_contract_fingerprint": tool_call_request_fingerprint(request),
        "prompt_message_count": len(stable_messages),
        "prompt_digest": prompt_digest,
        "replay_messages": list(replay_messages),
        "pending_call_ids": list(pending_call_ids),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ToolTransportCheckpoint(
        checkpoint_id=f"{provider}_chat_completions_{revision}_{digest[:16]}",
        revision=revision,
        provider=provider,
        model=model,
        contract_version="chat.completions.tool_results/v1",
        turn_id=str(request.turn_id or ""),
        integrity_digest=digest,
        purpose="pending_tool_outputs",
        opaque_payload=payload,
    )


def _openai_like_checkpoint_payload(
    checkpoint: ToolTransportCheckpoint,
    *,
    request: ToolCallRequest,
    provider: str,
    model: str,
    stable_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and detach one Chat Completions replay checkpoint.

    Intro:
        Validation rejects foreign bindings, modified payloads, changed prompts,
        changed Tool contracts, and incomplete pending-call state before I/O.

    Examples:
        Restore a valid checkpoint:
            ```python
            payload = _openai_like_checkpoint_payload(
                checkpoint,
                request=continued_request,
                provider="openrouter",
                model="openai/gpt-test",
                stable_messages=messages,
            )
            ```

        Reject a foreign provider binding:
            ```python
            try:
                _openai_like_checkpoint_payload(
                    checkpoint,
                    request=request,
                    provider="ollama",
                    model="local-model",
                    stable_messages=messages,
                )
            except ValueError:
                pass
            ```

    Args:
        checkpoint: Candidate private provider replay state.
        request: Exact continued Tool request.
        provider: Exact registered provider identity.
        model: Exact configured model identity.
        stable_messages: Current original prompt messages.

    Returns:
        dict[str, Any]: Detached validated opaque payload.

    Notes:
        Failure diagnostics never include replay message content or Tool outputs.
    """

    if (
        checkpoint.provider != provider
        or checkpoint.model != model
        or checkpoint.contract_version != "chat.completions.tool_results/v1"
    ):
        raise ValueError("Chat Completions Tool checkpoint binding does not match")
    payload = dict(checkpoint.opaque_payload or {})
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != checkpoint.integrity_digest:
        raise ValueError("Chat Completions Tool checkpoint integrity validation failed")
    if payload.get("state") != "pending_tool_outputs":
        raise ValueError("Chat Completions Tool checkpoint state is invalid")
    if payload.get("tool_contract_fingerprint") != tool_call_request_fingerprint(request):
        raise ValueError("Chat Completions Tool checkpoint contract changed")
    if payload.get("prompt_message_count") != len(stable_messages) or payload.get(
        "prompt_digest"
    ) != _openai_like_messages_digest(stable_messages):
        raise ValueError("Chat Completions Tool checkpoint prompt changed")
    pending_call_ids = payload.get("pending_call_ids")
    if not isinstance(pending_call_ids, list) or not pending_call_ids:
        raise ValueError("Chat Completions Tool checkpoint has no pending calls")
    if not all(isinstance(call_id, str) and call_id.strip() for call_id in pending_call_ids):
        raise ValueError("Chat Completions Tool checkpoint call identity is invalid")
    if len(pending_call_ids) != len(set(pending_call_ids)):
        raise ValueError("Chat Completions Tool checkpoint call identities are not unique")
    replay_messages = payload.get("replay_messages")
    if (
        not isinstance(replay_messages, list)
        or not replay_messages
        or not all(isinstance(message, dict) for message in replay_messages)
    ):
        raise ValueError("Chat Completions Tool checkpoint replay state is invalid")
    return payload


def _openai_like_messages_digest(messages: list[dict[str, Any]]) -> str:
    """Return the canonical identity of one stable Chat Completions prompt.

    Intro:
        The digest binds continuation replay to the exact provider-projected
        prompt without persisting a second public conversation representation.

    Examples:
        Digest one prompt:
            ```python
            digest = _openai_like_messages_digest(messages)
            assert len(digest) == 64
            ```

        Observe prompt changes:
            ```python
            assert _openai_like_messages_digest(first) != _openai_like_messages_digest(second)
            ```

    Args:
        messages: Provider-projected stable conversation messages.

    Returns:
        str: Lowercase SHA-256 digest of canonical JSON.

    Notes:
        Prompt contents remain outside the checkpoint payload.
    """

    canonical = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _openai_like_continuation_messages(
    messages: list[dict[str, Any]],
    *,
    tool_request: ToolCallRequest,
    provider: str,
    model: str,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    """Append validated Tool results to one compatible replay transcript.

    Intro:
        Standard assistant Tool-call messages are replayed before one `tool`
        message per pending call, in provider order and with exact call IDs.

    Examples:
        Prepare an initial Tool request:
            ```python
            provider_messages, replay = _openai_like_continuation_messages(
                messages,
                tool_request=request,
                provider="openrouter",
                model="openai/gpt-test",
            )
            assert replay == ()
            ```

        Prepare a continued Tool request:
            ```python
            provider_messages, replay = _openai_like_continuation_messages(
                messages,
                tool_request=continued_request,
                provider="ollama",
                model="local-model",
            )
            assert replay[-1]["role"] == "tool"
            ```

    Args:
        messages: Stable provider-projected prompt messages.
        tool_request: Exact current Tool request and optional checkpoint/results.
        provider: Exact registered provider identity.
        model: Exact configured model identity.

    Returns:
        tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]: Complete provider
            messages and the private replay suffix used for the next checkpoint.

    Notes:
        Output IDs must exactly equal pending IDs; partial and extra results fail
        before transport rather than being ignored or retried.
    """

    checkpoint = tool_request.transport_checkpoint
    if checkpoint is None:
        return messages, ()
    payload = _openai_like_checkpoint_payload(
        checkpoint,
        request=tool_request,
        provider=provider,
        model=model,
        stable_messages=messages,
    )
    pending_call_ids = tuple(str(call_id) for call_id in payload["pending_call_ids"])
    outputs_by_id = {output.call_id: output.output for output in tool_request.tool_outputs}
    if set(outputs_by_id) != set(pending_call_ids):
        raise LLMToolCallResponseError(
            code="tool_output_mismatch",
            message="Chat Completions continuation requires exactly every pending Tool output.",
        )
    replay_messages = tuple(dict(message) for message in payload["replay_messages"])
    result_messages = tuple(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": outputs_by_id[call_id],
        }
        for call_id in pending_call_ids
    )
    replay = (*replay_messages, *result_messages)
    return [*messages, *replay], replay


async def _stream_openai_like_chat_completions(
    *,
    http_client: Any,
    provider: str,
    model: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    on_delta: Any = None,
    on_usage_update: Any = None,
) -> ProviderCallResult[tuple[str, dict[str, int]]]:
    """Execute one OpenAI-compatible Chat Completions SSE request.

    Intro:
        The shared parser preserves ordered text deltas and the latest provider
        usage receipt while endpoint adapters retain URL, headers, and body ownership.

    Examples:
        Stream a compatible endpoint:
            ```python
            result = await _stream_openai_like_chat_completions(
                http_client=client,
                provider="openrouter",
                model="openai/gpt-test",
                url=url,
                headers=headers,
                body=body,
            )
            ```

        Forward deltas:
            ```python
            result = await _stream_openai_like_chat_completions(
                http_client=client,
                provider="azure",
                model="deployment-a",
                url=url,
                headers=headers,
                body=body,
                on_delta=on_delta,
                on_usage_update=on_usage_update,
            )
            ```

    Args:
        http_client: Bound asynchronous HTTP client.
        provider: Exact registered provider identity.
        model: Exact configured model or deployment identity.
        url: Exact adapter-owned Chat Completions URL.
        headers: Exact adapter-owned authentication and content headers.
        body: Complete adapter-owned streaming request body.
        on_delta: Optional async assistant-text callback.
        on_usage_update: Optional async cumulative usage callback.

    Returns:
        ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated text, latest
            usage receipt, and sanitized transport metadata.

    Notes:
        Malformed non-data SSE frames are ignored. Provider error responses and
        rate-limit metadata are handled by the shared transport classifier.
    """

    chunks: list[str] = []
    usage: dict[str, int] = {}
    async with http_client.stream(
        "POST",
        url,
        headers=headers,
        json=body,
    ) as response:
        if response.is_error:
            await response.aread()
        metadata = checked_response_metadata(provider, model, "chat_stream", response)

        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if not data_str or data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except (TypeError, ValueError):
                continue
            choices = event.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    chunks.append(delta)
                    if on_delta is not None:
                        await on_delta(delta)
            event_usage = event.get("usage")
            if isinstance(event_usage, dict) and event_usage:
                usage = dict(event_usage)
                if on_usage_update is not None:
                    await on_usage_update(dict(usage))

    return ProviderCallResult(("".join(chunks), usage), metadata)


def _openai_like_tool_definitions(tool_request: ToolCallRequest) -> list[dict[str, Any]]:
    """Project canonical Tools into Chat Completions function declarations."""

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tool_request.tools
    ]


class OpenAICompatibleChatAdapter:
    """Physical adapter for OpenAI-compatible Chat Completions endpoints."""

    @staticmethod
    async def invoke(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tool_request: ToolCallRequest | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Invoke one OpenAI-compatible Chat Completions request.

        Intro:
            The adapter projects text, structured output, native Tools, and
            integrity-bound Tool-result continuation into one compatible call.

        Examples:
            Send a direct request:
                ```python
                result = await OpenAICompatibleChatAdapter.invoke(
                    client,
                    messages,
                    model="local-model",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=False,
                )
                ```

            Continue a Tool request:
                ```python
                result = await OpenAICompatibleChatAdapter.invoke(
                    client,
                    messages,
                    model="local-model",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=False,
                    tool_request=continued_request,
                )
                ```

        Args:
            host: Bound generic client owning compatible transport primitives.
            messages: Provider-projected stable conversation messages.
            model: Exact configured model identity.
            reasoning_effort: Optional provider reasoning control.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text, JSON, or raw response mode.
            json_schema: Optional canonical JSON schema.
            fail_on_unsupported: Whether unsupported native formatting fails.
            tools: Optional legacy provider Tool declarations.
            tool_choice: Optional legacy provider Tool-selection policy.
            tool_request: Optional canonical native Tool request and continuation.
            **kw: Additional bounded provider options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Normalized response and raw provider usage with transport metadata.

        Notes:
            Each invocation is single-attempt at this adapter boundary. Shared
            retry, rate gating, accounting, and observations remain caller-owned.
        """

        await host._ensure_client()
        assert host._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        replay_messages: tuple[dict[str, Any], ...] = ()
        if tool_request is not None:
            msg_for_provider, replay_messages = _openai_like_continuation_messages(
                messages,
                tool_request=tool_request,
                provider=host.provider,
                model=model,
            )
        response_format = None
        structured_output_fields = kw.pop("structured_output_fields", None)

        if structured_output_fields:
            response_format = structured_output_fields.get("response_format")
        elif output_format == "json_object":
            if host.provider == "lmstudio":
                if fail_on_unsupported:
                    raise RuntimeError(
                        "LM Studio does not support response_format.type='json_object'; "
                        "use compatibility mode for text-mode JSON with local validation"
                    )
                response_format = {"type": "text"}
            else:
                response_format = {"type": "json_object"}
            msg_for_provider = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            if host.provider == "lmstudio":
                if json_schema is None:
                    raise ValueError("output_format='json_schema' requires json_schema")
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": kw.get("schema_name", "output"),
                        "schema": json_schema,
                        "strict": bool(kw.get("strict_schema", True)),
                    },
                }
            elif fail_on_unsupported:
                raise RuntimeError(f"provider {host.provider} does not support native json_schema")
            msg_for_provider = _ensure_system_json_directive(messages, schema=json_schema)

        async def _call():
            body: dict[str, Any] = {
                "model": model,
                "messages": msg_for_provider,
                "temperature": temperature,
                "top_p": top_p,
            }
            if max_output_tokens is not None:
                body["max_tokens"] = max_output_tokens
            if reasoning_effort is not None and host.provider == "deepseek":
                body["reasoning_effort"] = host._map_deepseek_reasoning_effort(reasoning_effort)
            if host.provider == "deepseek":
                body.update(host._deepseek_thinking_body(**kw))
            if response_format is not None:
                body["response_format"] = response_format
            if tool_request is not None:
                body["tools"] = _openai_like_tool_definitions(tool_request)
                body["tool_choice"] = tool_request.choice
                body["parallel_tool_calls"] = tool_request.max_calls > 1
            elif tools is not None:
                body["tools"] = tools
            if tool_request is None and tool_choice is not None:
                body["tool_choice"] = tool_choice

            r = await host._client.post(
                f"{host.base_url}/chat/completions",
                headers=host._headers_openai_like(),
                json=body,
            )
            metadata = checked_response_metadata(host.provider, model, "chat", r)

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            if tool_request is not None:
                response = _openai_like_tool_call_response(
                    data,
                    tool_request=tool_request,
                    provider=host.provider,
                    model=model,
                    stable_messages=messages,
                    replay_messages=replay_messages,
                )
                return ProviderCallResult((response, usage), metadata)

            txt, _ = _first_text(data.get("choices", []))
            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    @staticmethod
    async def stream(
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
        """Stream one OpenAI-compatible Chat Completions request.

        Intro:
            Builds the exact compatible request and delegates only SSE parsing to
            the adjacent shared Chat Completions stream helper.

        Examples:
            Stream default text:
                ```python
                result = await OpenAICompatibleChatAdapter.stream(
                    client,
                    messages,
                    model="local-model",
                )
                ```

            Stream with a token ceiling:
                ```python
                result = await OpenAICompatibleChatAdapter.stream(
                    client,
                    messages,
                    model="local-model",
                    max_output_tokens=256,
                    on_delta=on_delta,
                    on_usage_update=on_usage_update,
                )
                ```

        Args:
            host: Bound generic client owning compatible transport primitives.
            messages: Provider-projected stable conversation messages.
            model: Exact configured model identity.
            reasoning_effort: Optional compatible reasoning-depth override.
            max_output_tokens: Optional maximum generated tokens.
            on_delta: Optional async assistant-text callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded compatible generation options.

        Returns:
            ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated text,
                provider usage, and sanitized transport metadata.

        Notes:
            Retry, rate gating, quota accounting, and observations remain owned by
            the shared invocation lifecycle.
        """

        await host._ensure_client()
        assert host._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        if reasoning_effort is not None and host.provider == "deepseek":
            body["reasoning_effort"] = host._map_deepseek_reasoning_effort(reasoning_effort)
        if host.provider == "deepseek":
            body.update(host._deepseek_thinking_body(**kw))

        return await _stream_openai_like_chat_completions(
            http_client=host._client,
            provider=host.provider,
            model=model,
            url=f"{host.base_url}/chat/completions",
            headers=host._headers_openai_like(),
            body=body,
            on_delta=on_delta,
            on_usage_update=on_usage_update,
        )

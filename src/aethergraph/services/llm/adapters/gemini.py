"""Physical Google Gemini GenerateContent Chat adapter."""

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
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    LLMUnsupportedFeatureError,
)
from aethergraph.services.llm.utils import (
    _to_gemini_parts,
)


def _gemini_function_parameters(schema: Any) -> Any:
    """Project canonical JSON Schema into Gemini's FunctionDeclaration subset."""

    if isinstance(schema, dict):
        return {
            key: _gemini_function_parameters(value)
            for key, value in schema.items()
            if key != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_gemini_function_parameters(value) for value in schema]
    return schema


def _gemini_stable_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project the caller-owned stable conversation into Gemini contents."""

    system_parts: list[str] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            system_parts.append(content if isinstance(content, str) else str(content))
    system = "\n".join(system_parts)
    contents = [
        {
            "role": "user" if message.get("role") == "user" else "model",
            "parts": _to_gemini_parts(message.get("content")),
        }
        for message in messages
        if message.get("role") != "system"
    ]
    if system:
        contents.insert(
            0,
            {"role": "user", "parts": [{"text": f"System instructions: {system}"}]},
        )
    return contents


def _gemini_messages_digest(messages: list[dict[str, Any]]) -> str:
    """Bind private Gemini replay state to the exact stable prompt."""

    canonical = json.dumps(messages, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _gemini_checkpoint(
    *,
    request: ToolCallRequest,
    model: str,
    stable_messages: list[dict[str, Any]],
    replay_contents: tuple[dict[str, Any], ...],
    pending_calls: tuple[dict[str, str], ...],
) -> ToolTransportCheckpoint:
    """Build one integrity-bound Gemini function-result checkpoint."""

    previous = request.transport_checkpoint
    revision = 1 if previous is None else previous.revision + 1
    payload = {
        "state": "pending_tool_outputs",
        "tool_contract_fingerprint": tool_call_request_fingerprint(request),
        "prompt_message_count": len(stable_messages),
        "prompt_digest": _gemini_messages_digest(stable_messages),
        "replay_contents": list(replay_contents),
        "pending_calls": list(pending_calls),
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ToolTransportCheckpoint(
        checkpoint_id=f"google_generate_content_{revision}_{digest[:16]}",
        revision=revision,
        provider="google",
        model=model,
        contract_version="generate_content.tool_results/v1",
        turn_id=str(request.turn_id or ""),
        integrity_digest=digest,
        purpose="pending_tool_outputs",
        opaque_payload=payload,
    )


def _gemini_checkpoint_payload(
    checkpoint: ToolTransportCheckpoint,
    *,
    request: ToolCallRequest,
    model: str,
    stable_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and detach one Gemini function-result checkpoint."""

    if (
        checkpoint.provider != "google"
        or checkpoint.model != model
        or checkpoint.contract_version != "generate_content.tool_results/v1"
    ):
        raise LLMToolCallResponseError(
            code="model_continuation_binding_mismatch",
            message="Gemini Tool checkpoint binding does not match.",
        )
    payload = dict(checkpoint.opaque_payload or {})
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if digest != checkpoint.integrity_digest:
        raise LLMToolCallResponseError(
            code="model_continuation_integrity_invalid",
            message="Gemini Tool checkpoint integrity validation failed.",
        )
    if payload.get("state") != "pending_tool_outputs":
        raise LLMToolCallResponseError(
            code="model_continuation_state_invalid",
            message="Gemini Tool checkpoint state is invalid.",
        )
    if payload.get("tool_contract_fingerprint") != tool_call_request_fingerprint(request):
        raise LLMToolCallResponseError(
            code="model_exchange_tool_contract_changed",
            message="Gemini Tool checkpoint contract changed.",
        )
    if payload.get("prompt_message_count") != len(stable_messages) or payload.get(
        "prompt_digest"
    ) != _gemini_messages_digest(stable_messages):
        raise LLMToolCallResponseError(
            code="prompt_continuation_diverged",
            message="Gemini Tool checkpoint prompt changed.",
        )
    pending_calls = payload.get("pending_calls")
    if not isinstance(pending_calls, list) or not pending_calls:
        raise LLMToolCallResponseError(
            code="model_continuation_pending_calls_invalid",
            message="Gemini Tool checkpoint has no pending calls.",
        )
    call_ids = [str(item.get("call_id") or "") for item in pending_calls if isinstance(item, dict)]
    if len(call_ids) != len(pending_calls) or any(not call_id for call_id in call_ids):
        raise LLMToolCallResponseError(
            code="model_continuation_pending_calls_invalid",
            message="Gemini Tool checkpoint call identity is invalid.",
        )
    if len(call_ids) != len(set(call_ids)):
        raise LLMToolCallResponseError(
            code="model_continuation_pending_calls_invalid",
            message="Gemini Tool checkpoint call identities are not unique.",
        )
    if any(
        not isinstance(item.get("name"), str) or not str(item.get("name") or "").strip()
        for item in pending_calls
    ):
        raise LLMToolCallResponseError(
            code="model_continuation_pending_calls_invalid",
            message="Gemini Tool checkpoint call name is invalid.",
        )
    replay_contents = payload.get("replay_contents")
    if (
        not isinstance(replay_contents, list)
        or not replay_contents
        or not all(isinstance(content, dict) for content in replay_contents)
    ):
        raise LLMToolCallResponseError(
            code="model_continuation_replay_invalid",
            message="Gemini Tool checkpoint replay state is invalid.",
        )
    return payload


def _gemini_tool_output_response(output: str) -> dict[str, Any]:
    """Convert one opaque Engine Tool result into Gemini's required object."""

    try:
        decoded = json.loads(output)
    except json.JSONDecodeError:
        decoded = output
    return decoded if isinstance(decoded, dict) else {"output": decoded}


def _gemini_continuation_contents(
    messages: list[dict[str, Any]],
    *,
    tool_request: ToolCallRequest,
    model: str,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    """Append exact Gemini function calls and matching function responses."""

    stable_contents = _gemini_stable_contents(messages)
    checkpoint = tool_request.transport_checkpoint
    if checkpoint is None:
        return stable_contents, ()
    payload = _gemini_checkpoint_payload(
        checkpoint,
        request=tool_request,
        model=model,
        stable_messages=messages,
    )
    pending_calls = tuple(dict(item) for item in payload["pending_calls"])
    outputs_by_id = {output.call_id: output.output for output in tool_request.tool_outputs}
    pending_ids = {str(item["call_id"]) for item in pending_calls}
    if set(outputs_by_id) != pending_ids:
        raise LLMToolCallResponseError(
            code="tool_output_mismatch",
            message="Gemini continuation requires exactly every pending Tool output.",
        )
    result_parts: list[dict[str, Any]] = []
    for item in pending_calls:
        response: dict[str, Any] = {
            "name": str(item["name"]),
            "response": _gemini_tool_output_response(outputs_by_id[str(item["call_id"])]),
        }
        provider_call_id = str(item.get("provider_call_id") or "").strip()
        if provider_call_id:
            response["id"] = provider_call_id
        result_parts.append({"functionResponse": response})
    replay = (
        *(dict(content) for content in payload["replay_contents"]),
        {"role": "user", "parts": result_parts},
    )
    return [*stable_contents, *replay], replay


def _gemini_tool_call_response(
    candidate: dict[str, Any],
    *,
    tool_request: ToolCallRequest,
    model: str,
    stable_messages: list[dict[str, Any]],
    replay_contents: tuple[dict[str, Any], ...],
) -> ToolCallResponse:
    """Normalize Gemini function-call parts without flattening part boundaries."""

    items: list[AssistantOutput | ToolCall] = []
    pending_calls: list[dict[str, str]] = []
    content = candidate.get("content")
    content = dict(content) if isinstance(content, dict) else {}
    parts = list(content.get("parts") or [])
    allowed_names = {tool.name for tool in tool_request.tools}
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        if "text" in part:
            output_text = str(part.get("text") or "")
            provider_item_id = str(part.get("id") or "")
            items.append(
                AssistantOutput(
                    output_id=assistant_output_identity(
                        provider="gemini",
                        provider_item_id=provider_item_id,
                        item_index=int(candidate.get("index") or 0),
                        content_index=part_index,
                        text=output_text,
                    ),
                    text=output_text,
                    provider_metadata={
                        "provider_item_id": provider_item_id,
                        "part_index": part_index,
                    },
                )
            )
        function_call = part.get("functionCall") or part.get("function_call")
        if not isinstance(function_call, dict):
            continue
        arguments = function_call.get("args")
        if not isinstance(arguments, dict):
            raise LLMToolCallResponseError(
                code="invalid_arguments",
                message=(
                    f"Gemini Tool call '{function_call.get('name') or '?'}' "
                    "arguments must be an object."
                ),
            )
        name = str(function_call.get("name") or "").strip()
        if name not in allowed_names:
            raise LLMToolCallResponseError(
                code="unknown_tool",
                message=f"Gemini returned unknown Tool {name or '?'}.",
            )
        metadata: dict[str, Any] = {"part_index": part_index}
        thought_signature = part.get("thoughtSignature") or part.get("thought_signature")
        if thought_signature is not None:
            metadata["thought_signature"] = thought_signature
        provider_call_id = str(function_call.get("id") or part.get("id") or "").strip()
        call_id = provider_call_id or f"gemini-call-{part_index}"
        items.append(
            ToolCall(
                call_id=call_id,
                name=name,
                arguments=dict(arguments),
                provider_metadata=metadata,
            )
        )
        pending_call = {"call_id": call_id, "name": name}
        if provider_call_id:
            pending_call["provider_call_id"] = provider_call_id
        pending_calls.append(pending_call)
    if len(pending_calls) > tool_request.max_calls:
        raise LLMToolCallResponseError(
            code="tool_call_cardinality_exceeded",
            message=(
                f"Gemini returned {len(pending_calls)} Tool calls, exceeding the "
                f"request maximum of {tool_request.max_calls}."
            ),
        )
    checkpoint = None
    if pending_calls and tool_request.turn_id:
        content["role"] = str(content.get("role") or "model")
        checkpoint = _gemini_checkpoint(
            request=tool_request,
            model=model,
            stable_messages=stable_messages,
            replay_contents=(*replay_contents, content),
            pending_calls=tuple(pending_calls),
        )
    return ToolCallResponse(
        items=tuple(items),
        finish_reason=str(candidate.get("finishReason") or ""),
        provider_metadata={
            "candidate_index": int(candidate.get("index") or 0),
            "part_count": len(parts),
        },
        transport_checkpoint=checkpoint,
    )


def _gemini_generate_content_payload(
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    top_p: float,
    max_output_tokens: int | None,
    thinking_config: dict[str, Any] | None,
    output_format: ChatOutputFormat,
    json_schema: dict[str, Any] | None,
    structured_output_fields: dict[str, Any] | None,
    tool_request: ToolCallRequest | None,
    provider_contents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one shared Gemini GenerateContent request payload.

    Intro:
        Unary and SSE adapters use the same message, generation, structured-output,
        thinking, and native function-declaration projection.

    Examples:
        Build a text request:
            ```python
            payload = _gemini_generate_content_payload(
                messages,
                temperature=0.5,
                top_p=1.0,
                max_output_tokens=None,
                thinking_config=None,
                output_format="text",
                json_schema=None,
                structured_output_fields=None,
                tool_request=None,
            )
            ```

        Build a native Tool request:
            ```python
            payload = _gemini_generate_content_payload(
                messages,
                temperature=0.5,
                top_p=1.0,
                max_output_tokens=256,
                thinking_config={"thinkingLevel": "high"},
                output_format="text",
                json_schema=None,
                structured_output_fields=None,
                tool_request=tool_request,
            )
            ```

    Args:
        messages: Provider-projected stable conversation messages.
        temperature: Exact normalized sampling temperature.
        top_p: Exact normalized nucleus-sampling value.
        max_output_tokens: Optional maximum generated tokens.
        thinking_config: Optional Gemini thinking configuration.
        output_format: Requested text, JSON, schema, or raw mode.
        json_schema: Optional canonical JSON schema.
        structured_output_fields: Optional prepared native structured fields.
        tool_request: Optional canonical native Tool request.
        provider_contents: Optional validated continuation-aware contents.

    Returns:
        dict[str, Any]: Complete detached Gemini request body.

    Notes:
        System messages retain the established user-preamble projection for wire
        compatibility. Provider-native system instructions remain a later adapter
        migration and are not changed by the streaming cutover.
    """

    turns = (
        [dict(content) for content in provider_contents]
        if provider_contents is not None
        else _gemini_stable_contents(messages)
    )

    generation_config: dict[str, Any] = {"temperature": temperature, "topP": top_p}
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    if thinking_config:
        generation_config["thinkingConfig"] = dict(thinking_config)
    if structured_output_fields:
        generation_config.update(structured_output_fields.get("generationConfig") or {})
    elif output_format == "json_object":
        generation_config["responseMimeType"] = "application/json"
    elif output_format == "json_schema":
        if json_schema is None:
            raise ValueError("output_format='json_schema' requires json_schema")
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseJsonSchema"] = json_schema

    payload: dict[str, Any] = {
        "contents": turns,
        "generationConfig": generation_config,
    }
    if tool_request is not None:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": _gemini_function_parameters(tool.input_schema),
                    }
                    for tool in tool_request.tools
                ]
            }
        ]
        function_calling_config: dict[str, Any] = {
            "mode": {"auto": "AUTO", "required": "ANY", "none": "NONE"}[tool_request.choice],
        }
        if tool_request.choice == "required":
            function_calling_config["allowedFunctionNames"] = [
                tool.name for tool in tool_request.tools
            ]
        payload["toolConfig"] = {"functionCallingConfig": function_calling_config}
    return payload


def _gemini_usage(usage_metadata: Any) -> dict[str, int]:
    """Normalize one Gemini usage receipt into shared raw keys.

    Intro:
        Gemini token counters are retained without conflating missing fields with
        unrelated totals, including cache reads and reasoning tokens.

    Examples:
        Normalize a complete receipt:
            ```python
            usage = _gemini_usage({"promptTokenCount": 3, "candidatesTokenCount": 2})
            ```

        Normalize a missing receipt:
            ```python
            assert _gemini_usage(None) == {}
            ```

    Args:
        usage_metadata: Candidate Gemini `usageMetadata` mapping.

    Returns:
        dict[str, int]: Present canonical raw usage counters only.

    Notes:
        `reasoning_tokens` maps Gemini's billed `thoughtsTokenCount`; assistant
        output remains `candidatesTokenCount`.
    """

    if not isinstance(usage_metadata, dict) or not usage_metadata:
        return {}
    keys = {
        "input_tokens": "promptTokenCount",
        "output_tokens": "candidatesTokenCount",
        "cache_read_tokens": "cachedContentTokenCount",
        "reasoning_tokens": "thoughtsTokenCount",
    }
    return {
        target: int(usage_metadata.get(source, 0) or 0)
        for target, source in keys.items()
        if source in usage_metadata
    }


class GeminiGenerateContentAdapter:
    """Physical adapter for the Gemini GenerateContent Chat endpoint."""

    @staticmethod
    async def invoke(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        thinking_mode: str | None = None,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_request: ToolCallRequest | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Generate one unary Gemini response or native Tool selection.

        Intro:
            Projects the stable conversation through the shared GenerateContent
            builder, then normalizes text, usage, or ordered Tool items.

        Examples:
            Generate text:
                ```python
                result = await GeminiGenerateContentAdapter.invoke(
                    client,
                    messages,
                    model="gemini-test",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=True,
                )
                ```

            Generate a native Tool selection:
                ```python
                result = await GeminiGenerateContentAdapter.invoke(
                    client,
                    messages,
                    model="gemini-test",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=True,
                    tool_request=tool_request,
                )
                ```

        Args:
            host: Bound generic client owning the Gemini transport.
            messages: Provider-projected stable conversation messages.
            model: Exact configured Gemini model identity.
            reasoning_effort: Optional normalized reasoning-depth override.
            thinking_mode: Optional profile thinking-mode override.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text, JSON, schema, or raw mode.
            json_schema: Optional canonical JSON schema.
            fail_on_unsupported: Whether unsupported native features must fail.
            tools: Deprecated provider-neutral Tool catalog, which is rejected.
            tool_request: Optional canonical native Tool request.
            **kw: Additional bounded Gemini sampling and structured-output options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Normalized output, usage receipt, and transport metadata.

        Notes:
            Native Tool calling and structured output are mutually exclusive. The
            provider-neutral `tools` escape hatch is intentionally not projected.
        """

        await host._ensure_client()
        assert host._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)
        structured_output_fields = kw.pop("structured_output_fields", None)

        if tools is not None:
            raise LLMUnsupportedFeatureError(
                host.provider,
                model,
                "provider-neutral tools",
                "Gemini function declaration translation is not wired yet; refusing to pass tools through blindly.",
            )
        if tool_request is not None and (
            structured_output_fields or output_format in {"json_object", "json_schema"}
        ):
            raise ValueError("Native Tool calling cannot be combined with structured output")

        async def _call():
            """Issue the already-validated unary GenerateContent request.

            Intro:
                Resolves thinking configuration beside request construction and
                converts the checked provider response into the shared result.

            Examples:
                Await the internal call:
                    ```python
                    result = await _call()
                    ```

                Read the normalized pair:
                    ```python
                    value, usage = (await _call()).value
                    ```

            Args:
                This function accepts no arguments.

            Returns:
                ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                    Normalized output, usage receipt, and transport metadata.

            Notes:
                Retry ownership remains in the surrounding shared runtime.
            """

            thinking_cfg = host._gemini_thinking_config(
                model=model, reasoning_effort=reasoning_effort, thinking_mode=thinking_mode
            )
            if tool_request is None:
                provider_contents = _gemini_stable_contents(messages)
                replay_contents: tuple[dict[str, Any], ...] = ()
            else:
                provider_contents, replay_contents = _gemini_continuation_contents(
                    messages,
                    tool_request=tool_request,
                    model=model,
                )
            payload = _gemini_generate_content_payload(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                thinking_config=thinking_cfg,
                output_format=output_format,
                json_schema=json_schema,
                structured_output_fields=structured_output_fields,
                tool_request=tool_request,
                provider_contents=provider_contents,
            )

            r = await host._client.post(
                f"{host.base_url}/v1beta/models/{model}:generateContent?key={host.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            metadata = checked_response_metadata("google", model, "chat", r)

            data = r.json()
            usage = _gemini_usage(data.get("usageMetadata"))

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            cand = (data.get("candidates") or [{}])[0]
            if tool_request is not None:
                if str(cand.get("finishReason") or "").upper() == "MAX_TOKENS":
                    raise LLMToolCallResponseError(
                        code="truncated",
                        message=(
                            "Gemini stopped at maxOutputTokens before completing "
                            "native Tool selection."
                        ),
                    )
                return ProviderCallResult(
                    (
                        _gemini_tool_call_response(
                            cand,
                            tool_request=tool_request,
                            model=model,
                            stable_messages=messages,
                            replay_contents=replay_contents,
                        ),
                        usage,
                    ),
                    metadata,
                )
            txt = "".join(p.get("text", "") for p in (cand.get("content", {}).get("parts") or []))
            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    @staticmethod
    async def stream(
        host: Any,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        reasoning_summary: str | None = None,
        thinking_mode: str | None = None,
        max_output_tokens: int | None = None,
        on_delta: Any = None,
        on_thinking_delta: Any = None,
        on_usage_update: Any = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        """Stream one Gemini GenerateContent request over SSE.

        Intro:
            Uses `streamGenerateContent` with the same request builder as unary
            generation, separating thought-summary parts from assistant text.

        Examples:
            Stream assistant text:
                ```python
                result = await GeminiGenerateContentAdapter.stream(
                    client,
                    messages,
                    model="gemini-test",
                    on_delta=on_delta,
                )
                ```

            Stream thought summaries:
                ```python
                result = await GeminiGenerateContentAdapter.stream(
                    client,
                    messages,
                    model="gemini-test",
                    reasoning_summary="auto",
                    on_thinking_delta=on_thinking_delta,
                    on_usage_update=on_usage_update,
                )
                ```

        Args:
            host: Bound generic client owning the Gemini transport.
            messages: Provider-projected stable conversation messages.
            model: Exact configured Gemini model identity.
            reasoning_effort: Optional normalized reasoning-depth override.
            reasoning_summary: Optional request for displayable thought summaries.
            thinking_mode: Optional profile thinking-mode override.
            max_output_tokens: Optional maximum generated tokens.
            on_delta: Optional async assistant-text callback.
            on_thinking_delta: Optional async thought-summary callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded Gemini sampling options.

        Returns:
            ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated assistant
                text, latest cumulative usage, and transport metadata.

        Notes:
            Thought signatures are not exposed as reasoning text. This text-only
            stream does not support native Tool continuation, whose signature replay
            remains owned by the non-streaming Tool protocol.
        """

        await host._ensure_client()
        assert host._client is not None
        thinking_config = host._gemini_thinking_config(
            model=model,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
        )
        if reasoning_summary is not None:
            thinking_config = dict(thinking_config or {})
            thinking_config["includeThoughts"] = True
        payload = _gemini_generate_content_payload(
            messages,
            temperature=kw.get("temperature", 0.5),
            top_p=kw.get("top_p", 1.0),
            max_output_tokens=max_output_tokens,
            thinking_config=thinking_config,
            output_format="text",
            json_schema=None,
            structured_output_fields=None,
            tool_request=None,
        )
        url = (
            f"{host.base_url}/v1beta/models/{model}:streamGenerateContent"
            f"?alt=sse&key={host.api_key}"
        )
        text_chunks: list[str] = []
        usage: dict[str, int] = {}

        async with host._client.stream(
            "POST",
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
        ) as response:
            if response.is_error:
                await response.aread()
            metadata = checked_response_metadata("google", model, "chat_stream", response)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if not data_str:
                    continue
                try:
                    event = json.loads(data_str)
                except (TypeError, ValueError):
                    continue
                error = event.get("error")
                if isinstance(error, dict):
                    raise RuntimeError(
                        f"Gemini streaming error: {error.get('message') or 'Unknown error'}"
                    )
                event_usage = _gemini_usage(event.get("usageMetadata"))
                if event_usage:
                    usage = event_usage
                    if on_usage_update is not None:
                        await on_usage_update(dict(usage))
                candidate = (event.get("candidates") or [{}])[0]
                for part in (candidate.get("content") or {}).get("parts") or []:
                    if not isinstance(part, dict):
                        continue
                    delta = str(part.get("text") or "")
                    if not delta:
                        continue
                    if bool(part.get("thought")):
                        if on_thinking_delta is not None:
                            await on_thinking_delta(delta)
                    else:
                        text_chunks.append(delta)
                        if on_delta is not None:
                            await on_delta(delta)

        return ProviderCallResult(("".join(text_chunks), usage), metadata)

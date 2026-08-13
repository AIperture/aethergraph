"""Physical Google Gemini GenerateContent Chat adapter."""

from __future__ import annotations

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
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    LLMUnsupportedFeatureError,
)
from aethergraph.services.llm.utils import (
    _to_gemini_parts,
)


def _gemini_tool_call_response(candidate: dict[str, Any]) -> ToolCallResponse:
    """Normalize Gemini function-call parts without flattening part boundaries."""

    items: list[AssistantOutput | ToolCall] = []
    parts = list((candidate.get("content") or {}).get("parts") or [])
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
        metadata: dict[str, Any] = {"part_index": part_index}
        thought_signature = part.get("thoughtSignature") or part.get("thought_signature")
        if thought_signature is not None:
            metadata["thought_signature"] = thought_signature
        items.append(
            ToolCall(
                call_id=str(
                    function_call.get("id") or part.get("id") or f"gemini-call-{part_index}"
                ),
                name=str(function_call.get("name") or ""),
                arguments=dict(arguments),
                provider_metadata=metadata,
            )
        )
    return ToolCallResponse(
        items=tuple(items),
        finish_reason=str(candidate.get("finishReason") or ""),
        provider_metadata={
            "candidate_index": int(candidate.get("index") or 0),
            "part_count": len(parts),
        },
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

    Returns:
        dict[str, Any]: Complete detached Gemini request body.

    Notes:
        System messages retain the established user-preamble projection for wire
        compatibility. Provider-native system instructions remain a later adapter
        migration and are not changed by the streaming cutover.
    """

    system_parts: list[str] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            system_parts.append(content if isinstance(content, str) else str(content))
    system = "\n".join(system_parts)

    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            continue
        role = "user" if message.get("role") == "user" else "model"
        turns.append({"role": role, "parts": _to_gemini_parts(message.get("content"))})
    if system:
        turns.insert(0, {"role": "user", "parts": [{"text": f"System instructions: {system}"}]})

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
                        "parameters": tool.input_schema,
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
            )

            r = await host._client.post(
                f"{host.base_url}/v1/models/{model}:generateContent?key={host.api_key}",
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
                return ProviderCallResult((_gemini_tool_call_response(cand), usage), metadata)
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
        url = f"{host.base_url}/v1/models/{model}:streamGenerateContent?alt=sse&key={host.api_key}"
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

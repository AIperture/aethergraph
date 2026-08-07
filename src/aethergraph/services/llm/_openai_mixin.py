"""OpenAI Responses API methods (chat + stream + image generation)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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


def _openai_tool_call_response(data: dict[str, Any]) -> ToolCallResponse:
    """Normalize OpenAI response items without flattening Tool calls into text."""

    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for output_index, item in enumerate(list(data.get("output") or [])):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            arguments = _openai_tool_arguments(
                item.get("arguments"),
                call_name=str(item.get("name") or ""),
            )
            calls.append(
                ToolCall(
                    call_id=str(
                        item.get("call_id") or item.get("id") or f"openai-call-{output_index}"
                    ),
                    name=str(item.get("name") or ""),
                    arguments=arguments,
                    provider_metadata={
                        "item_id": str(item.get("id") or ""),
                        "status": str(item.get("status") or ""),
                        "output_index": output_index,
                    },
                )
            )
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
    return ToolCallResponse(
        items=tuple(calls),
        text="".join(text_parts),
        finish_reason=str(data.get("status") or ""),
        provider_metadata={
            "response_id": str(data.get("id") or ""),
            "output_item_count": len(list(data.get("output") or [])),
        },
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

        # Tools (Responses API style)
        if tool_request is not None:
            body["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                    "strict": False,
                }
                for tool in tool_request.tools
            ]
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
                return ProviderCallResult((_openai_tool_call_response(data), usage), metadata)
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

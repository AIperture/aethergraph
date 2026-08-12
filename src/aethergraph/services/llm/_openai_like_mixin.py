"""OpenAI-compatible chat completions (OpenRouter, LMStudio, Ollama)."""

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
) -> ToolCallResponse:
    """Normalize one OpenAI-compatible Chat Completions Tool response."""

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

    return ToolCallResponse(
        items=tuple(items),
        finish_reason=str(choice.get("finish_reason") or ""),
        provider_metadata={
            "response_id": response_id,
            "choice_index": choice_index,
            "tool_call_count": len(raw_calls),
        },
    )


class _OpenAILikeMixin:
    """Provider methods for OpenRouter, LMStudio, Ollama (OpenAI-compatible endpoints)."""

    async def _chat_openai_like_chat_completions(
        self,
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
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        response_format = None
        structured_output_fields = kw.pop("structured_output_fields", None)

        if structured_output_fields:
            response_format = structured_output_fields.get("response_format")
        elif output_format == "json_object":
            if self.provider == "lmstudio":
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
            if self.provider == "lmstudio":
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
                raise RuntimeError(f"provider {self.provider} does not support native json_schema")
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
            if reasoning_effort is not None and self.provider == "deepseek":
                body["reasoning_effort"] = self._map_deepseek_reasoning_effort(reasoning_effort)
            if self.provider == "deepseek":
                body.update(self._deepseek_thinking_body(**kw))
            if response_format is not None:
                body["response_format"] = response_format
            if tool_request is not None:
                body["tools"] = [
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
                body["tool_choice"] = tool_request.choice
                body["parallel_tool_calls"] = tool_request.max_calls > 1
            elif tools is not None:
                body["tools"] = tools
            if tool_request is None and tool_choice is not None:
                body["tool_choice"] = tool_choice

            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers_openai_like(),
                json=body,
            )
            metadata = checked_response_metadata(self.provider, model, "chat", r)

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            if tool_request is not None:
                response = _openai_like_tool_call_response(
                    data,
                    tool_request=tool_request,
                    provider=self.provider,
                )
                return ProviderCallResult((response, usage), metadata)

            txt, _ = _first_text(data.get("choices", []))
            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    async def _chat_openai_like_chat_completions_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        on_delta: Any = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        await self._ensure_client()
        assert self._client is not None

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
        if reasoning_effort is not None and self.provider == "deepseek":
            body["reasoning_effort"] = self._map_deepseek_reasoning_effort(reasoning_effort)
        if self.provider == "deepseek":
            body.update(self._deepseek_thinking_body(**kw))

        chunks: list[str] = []
        usage: dict[str, int] = {}

        async def _call():
            nonlocal usage
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers_openai_like(),
                json=body,
            ) as r:
                if r.is_error:
                    await r.aread()
                metadata = checked_response_metadata(
                    self.provider,
                    model,
                    "chat_stream",
                    r,
                )

                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if not data_str or data_str == "[DONE]":
                        break
                    try:
                        evt = json.loads(data_str)
                    except Exception:
                        continue
                    choices = evt.get("choices") or []
                    if choices:
                        delta = (choices[0].get("delta") or {}).get("content") or ""
                        if delta:
                            chunks.append(delta)
                            if on_delta is not None:
                                await on_delta(delta)
                    if evt.get("usage"):
                        usage = evt.get("usage") or usage

                return metadata

        metadata = await _call()
        return ProviderCallResult(("".join(chunks), usage), metadata)

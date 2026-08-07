"""Google Gemini methods (chat + image generation)."""

from __future__ import annotations

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
    LLMUnsupportedFeatureError,
)
from aethergraph.services.llm.utils import (
    _data_url_to_b64_and_mime,
    _is_data_url,
    _normalize_base_url_no_trailing_slash,
    _to_gemini_parts,
)


def _gemini_tool_call_response(candidate: dict[str, Any]) -> ToolCallResponse:
    """Normalize Gemini function-call parts without flattening part boundaries."""

    calls: list[ToolCall] = []
    text_parts: list[str] = []
    parts = list((candidate.get("content") or {}).get("parts") or [])
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        if "text" in part:
            text_parts.append(str(part.get("text") or ""))
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
        calls.append(
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
        items=tuple(calls),
        text="".join(text_parts),
        finish_reason=str(candidate.get("finishReason") or ""),
        provider_metadata={
            "candidate_index": int(candidate.get("index") or 0),
            "part_count": len(parts),
        },
    )


class _GeminiMixin:
    """Provider methods for Google Gemini."""

    async def _chat_gemini_generate_content(
        self,
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
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)
        structured_output_fields = kw.pop("structured_output_fields", None)

        if tools is not None:
            raise LLMUnsupportedFeatureError(
                self.provider,
                model,
                "provider-neutral tools",
                "Gemini function declaration translation is not wired yet; refusing to pass tools through blindly.",
            )
        if tool_request is not None and (
            structured_output_fields or output_format in {"json_object", "json_schema"}
        ):
            raise ValueError("Native Tool calling cannot be combined with structured output")

        # Merge system messages into preamble
        system_parts: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                system_parts.append(c if isinstance(c, str) else str(c))
        system = "\n".join(system_parts)

        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "user" if m.get("role") == "user" else "model"
            parts = _to_gemini_parts(m.get("content"))
            turns.append({"role": role, "parts": parts})

        if system:
            turns.insert(0, {"role": "user", "parts": [{"text": f"System instructions: {system}"}]})

        async def _call():
            gen_cfg: dict[str, Any] = {"temperature": temperature, "topP": top_p}
            if max_output_tokens is not None:
                gen_cfg["maxOutputTokens"] = max_output_tokens
            thinking_cfg = self._gemini_thinking_config(
                model=model, reasoning_effort=reasoning_effort, thinking_mode=thinking_mode
            )
            if thinking_cfg:
                gen_cfg["thinkingConfig"] = thinking_cfg

            # Gemini native structured outputs
            if structured_output_fields:
                gen_cfg.update(structured_output_fields.get("generationConfig") or {})
            elif output_format == "json_object":
                gen_cfg["responseMimeType"] = "application/json"
            elif output_format == "json_schema":
                if json_schema is None:
                    raise ValueError("output_format='json_schema' requires json_schema")
                gen_cfg["responseMimeType"] = "application/json"
                gen_cfg["responseJsonSchema"] = json_schema

            payload: dict[str, Any] = {
                "contents": turns,
                "generationConfig": gen_cfg,
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
                    "mode": {
                        "auto": "AUTO",
                        "required": "ANY",
                        "none": "NONE",
                    }[tool_request.choice],
                }
                if tool_request.choice == "required":
                    function_calling_config["allowedFunctionNames"] = [
                        tool.name for tool in tool_request.tools
                    ]
                payload["toolConfig"] = {
                    "functionCallingConfig": function_calling_config,
                }

            r = await self._client.post(
                f"{self.base_url}/v1/models/{model}:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            metadata = checked_response_metadata("google", model, "chat", r)

            data = r.json()
            um = data.get("usageMetadata") or {}
            usage = {
                "input_tokens": int(um.get("promptTokenCount", 0) or 0),
                "output_tokens": int(um.get("candidatesTokenCount", 0) or 0),
                "cache_read_tokens": int(um.get("cachedContentTokenCount", 0) or 0),
            }

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

    async def _image_gemini_generate(
        self,
        prompt: str,
        *,
        model: str,
        input_images: list[str] | None,
        **kw: Any,
    ) -> ProviderCallResult[ImageGenerationResult]:
        assert self._client is not None

        base = (
            _normalize_base_url_no_trailing_slash(self.base_url)
            or "https://generativelanguage.googleapis.com"
        )
        url = f"{base}/v1beta/models/{model}:generateContent"

        parts: list[dict[str, Any]] = []
        if input_images:
            for img in input_images:
                if not _is_data_url(img):
                    raise ValueError("Gemini input_images must be data: URLs (base64) for now.")
                b64, mime = _data_url_to_b64_and_mime(img)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        parts.append({"text": prompt})

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
        }

        async def _call():
            r = await self._client.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            metadata = checked_response_metadata("google", model, "image", r)

            data = r.json()
            cand = (data.get("candidates") or [{}])[0]
            content = cand.get("content") or {}
            out_parts = content.get("parts") or []

            imgs: list[GeneratedImage] = []
            for p in out_parts:
                inline = p.get("inlineData") or p.get("inline_data")
                if inline and inline.get("data"):
                    mime = inline.get("mimeType") or inline.get("mime_type")
                    imgs.append(GeneratedImage(b64=inline["data"], mime_type=mime))

            um = data.get("usageMetadata") or {}
            usage = {
                "input_tokens": int(um.get("promptTokenCount", 0) or 0),
                "output_tokens": int(um.get("candidatesTokenCount", 0) or 0),
            }

            return ProviderCallResult(
                ImageGenerationResult(images=imgs, usage=usage, raw=data),
                metadata,
            )

        return await _call()

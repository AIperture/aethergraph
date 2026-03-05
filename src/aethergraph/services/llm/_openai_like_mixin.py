"""OpenAI-compatible chat completions (OpenRouter, LMStudio, Ollama)."""

from __future__ import annotations

import json
from typing import Any

import httpx

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


class _OpenAILikeMixin:
    """Provider methods for OpenRouter, LMStudio, Ollama (OpenAI-compatible endpoints)."""

    async def _chat_openai_like_chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        response_format = None

        if output_format == "json_object":
            response_format = {"type": "json_object"}
            msg_for_provider = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            if fail_on_unsupported:
                raise RuntimeError(f"provider {self.provider} does not support native json_schema")
            msg_for_provider = _ensure_system_json_directive(messages, schema=json_schema)

        async def _call():
            body: dict[str, Any] = {
                "model": model,
                "messages": msg_for_provider,
                "temperature": temperature,
                "top_p": top_p,
            }
            if response_format is not None:
                body["response_format"] = response_format
            if tools is not None:
                body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice

            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers_openai_like(),
                json=body,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"OpenAI-like chat/completions error: {e.response.text}") from e

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return txt, usage

            txt, _ = _first_text(data.get("choices", []))
            return txt, usage

        return await self._retry.run(_call)

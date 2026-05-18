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
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
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
            if max_output_tokens is not None:
                body["max_tokens"] = max_output_tokens
            if reasoning_effort is not None and self.provider == "deepseek":
                body["reasoning_effort"] = self._map_deepseek_reasoning_effort(reasoning_effort)
            if self.provider == "deepseek":
                body.update(self._deepseek_thinking_body(**kw))
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

    async def _chat_openai_like_chat_completions_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        on_delta: Any = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
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
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    text = await r.aread()
                    raise RuntimeError(f"OpenAI-like streaming error: {text!r}") from e

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

        await self._retry.run(_call)
        return "".join(chunks), usage

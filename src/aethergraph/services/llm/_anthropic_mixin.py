"""Anthropic Messages API methods (chat + streaming with extended thinking)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any

import httpx

from aethergraph.services.llm.types import ChatOutputFormat
from aethergraph.services.llm.utils import _to_anthropic_blocks

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]


class _AnthropicMixin:
    """Provider methods for Anthropic Messages API."""

    # ------------------------------------------------------------------
    # Chat – non-streaming
    # ------------------------------------------------------------------
    async def _chat_anthropic_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        if tools is not None and fail_on_unsupported:
            raise RuntimeError("Anthropic tools/function calling not wired yet in this client")

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        # System text aggregation
        sys_msgs: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                sys_msgs.append(c if isinstance(c, str) else str(c))

        if output_format in ("json_object", "json_schema"):
            sys_msgs.insert(0, "Return ONLY valid JSON. No markdown, no commentary.")
            if output_format == "json_schema" and json_schema is not None:
                sys_msgs.insert(
                    1,
                    "JSON MUST conform to this schema:\n"
                    + json.dumps(json_schema, ensure_ascii=False),
                )

        # Convert messages to Anthropic format (blocks)
        conv: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            anthro_role = "assistant" if role == "assistant" else "user"
            content_blocks = _to_anthropic_blocks(m.get("content"))
            conv.append({"role": anthro_role, "content": content_blocks})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": kw.get("max_tokens", 1024),
            "messages": conv,
            "temperature": temperature,
            "top_p": top_p,
        }
        if sys_msgs:
            payload["system"] = "\n\n".join(sys_msgs)

        async def _call():
            r = await self._client.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                body = e.response.text or ""
                if e.response.status_code == 404:
                    hint = (
                        "Anthropic returned 404. Common causes:\n"
                        "1) base_url should be https://api.anthropic.com (no /v1 suffix)\n"
                        "2) model id is invalid / unavailable for your key\n"
                        f"Request URL: {e.request.url}\n"
                    )
                    raise RuntimeError(hint + "Response body:\n" + body) from e

                raise RuntimeError(f"Anthropic API error ({e.response.status_code}): {body}") from e

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return txt, usage

            blocks = data.get("content") or []
            txt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            return txt, usage

        return await self._retry.run(_call)

    # ------------------------------------------------------------------
    # Chat – streaming (with extended thinking support)
    # ------------------------------------------------------------------
    async def _chat_anthropic_messages_stream(
        self,
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
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        """
        Stream text using Anthropic Messages API with SSE.

        Handles ``text_delta`` for content and ``thinking_delta`` for
        extended thinking blocks when ``thinking_budget`` is set.
        """
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        # System text aggregation
        sys_msgs: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                sys_msgs.append(c if isinstance(c, str) else str(c))

        if output_format in ("json_object", "json_schema"):
            sys_msgs.insert(0, "Return ONLY valid JSON. No markdown, no commentary.")
            if output_format == "json_schema" and json_schema is not None:
                sys_msgs.insert(
                    1,
                    "JSON MUST conform to this schema:\n"
                    + json.dumps(json_schema, ensure_ascii=False),
                )

        # Convert messages to Anthropic format (blocks)
        conv: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            anthro_role = "assistant" if role == "assistant" else "user"
            content_blocks = _to_anthropic_blocks(m.get("content"))
            conv.append({"role": anthro_role, "content": content_blocks})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens or kw.get("max_tokens", 4096),
            "messages": conv,
            "stream": True,
        }

        # Extended thinking — requires no temperature
        if thinking_budget is not None:
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        else:
            payload["temperature"] = temperature
            payload["top_p"] = top_p

        if sys_msgs:
            payload["system"] = "\n\n".join(sys_msgs)

        headers: dict[str, str] = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if thinking_budget is not None:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        text_chunks: list[str] = []
        usage: dict[str, int] = {}

        async def _call():
            nonlocal usage

            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
            ) as r:
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = await r.aread()
                    raise RuntimeError(
                        f"Anthropic streaming error ({e.response.status_code}): {body!r}"
                    ) from e

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

        async def _handle_sse_event(event_type: str, data: dict[str, Any]):
            nonlocal usage

            if event_type == "message_start":
                msg = data.get("message", {})
                msg_usage = msg.get("usage", {})
                if msg_usage:
                    usage.update(msg_usage)

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

            # content_block_start, content_block_stop, message_stop, ping: no action needed

            elif event_type == "error":
                err = data.get("error", {})
                msg = err.get("message", "Unknown Anthropic streaming error")
                raise RuntimeError(f"Anthropic streaming error: {msg}")

        await self._retry.run(_call)

        return "".join(text_chunks), usage

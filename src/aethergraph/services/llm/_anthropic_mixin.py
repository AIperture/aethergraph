"""Anthropic Messages API methods (chat + streaming with extended thinking)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import json
from typing import Any

import httpx

from aethergraph.services.llm.types import ChatOutputFormat, LLMUnsupportedFeatureError
from aethergraph.services.llm.utils import _to_anthropic_blocks

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]


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
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
        thinking_mode: str | None = None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        if tools is not None:
            raise LLMUnsupportedFeatureError(
                self.provider,
                model,
                "provider-neutral tools",
                "Anthropic tool translation is not wired yet; refusing to drop tools silently.",
            )

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
            "max_tokens": max_output_tokens or kw.get("max_tokens", 1024),
            "messages": conv,
            "temperature": temperature,
            "top_p": top_p,
        }
        structured_output_fields = kw.pop("structured_output_fields", None)
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
        _validate_anthropic_cache_breakpoints(payload)

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

from __future__ import annotations

import asyncio
import json
from textwrap import fill
from typing import Any, Literal, Protocol

from aethergraph.observability.models import CaptureMode, LLMObservationRecord

PromptViewMode = Literal["off", "compact", "truncated", "full"]


class LLMObservationSink(Protocol):
    async def begin_llm_call(
        self, record: LLMObservationRecord, *, capture_mode: CaptureMode
    ) -> None: ...

    async def finish_llm_call(
        self, record: LLMObservationRecord, *, capture_mode: CaptureMode
    ) -> None: ...


def _usage_summary(usage: dict[str, Any]) -> tuple[int, int, int]:
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return prompt_tokens, completion_tokens, total_tokens


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                    continue
                item_type = item.get("type")
                if item_type:
                    parts.append(f"[{item_type}]")
                    continue
            parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2, default=str)
    return str(content)


def _clip_text(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _format_block(
    title: str,
    text: str,
    *,
    prompt_view: PromptViewMode,
    width: int,
    truncation_chars: int,
) -> str:
    body = text.strip()
    if not body:
        return ""
    if prompt_view == "truncated":
        body = _clip_text(body, limit=truncation_chars)
    if prompt_view in {"compact", "truncated"}:
        body = fill(body, width=width, replace_whitespace=False, drop_whitespace=False)
    return f"[{title}]\n{body}"


def render_console_observation(
    record: LLMObservationRecord,
    *,
    prompt_view: PromptViewMode = "compact",
    width: int = 88,
    truncation_chars: int = 600,
) -> str:
    prompt_tokens, completion_tokens, total_tokens = _usage_summary(record.usage or {})
    status = "ERROR" if record.error_type else "OK"
    lines = [
        "=" * 80,
        f"LLM CALL  [{record.call_name or '-'}] {record.provider}/{record.model}  profile={record.profile_name or 'default'}",
        f"call_id: {record.llm_call_id}",
        f"run_id:  {record.scope.run_id or '-'}",
        f"graph:   {record.scope.graph_id or '-'}",
        f"time:    {record.created_at}",
        f"latency: {record.latency_ms if record.latency_ms is not None else '-'} ms",
        f"tokens:  in={prompt_tokens}  out={completion_tokens}  total={total_tokens}",
        f"status:  {status}",
    ]
    if record.error_type:
        lines.append(f"error:   {record.error_type}: {record.error_message or ''}".rstrip())

    if prompt_view != "off":
        blocks: list[str] = []
        for message in record.messages or []:
            role = str(message.get("role") or "message").upper()
            content = _stringify_content(message.get("content"))
            block = _format_block(
                role,
                content,
                prompt_view=prompt_view,
                width=width,
                truncation_chars=truncation_chars,
            )
            if block:
                blocks.append(block)
        if record.raw_text:
            output_block = _format_block(
                "OUTPUT",
                record.raw_text,
                prompt_view=prompt_view,
                width=width,
                truncation_chars=truncation_chars,
            )
            if output_block:
                blocks.append(output_block)
        if blocks:
            lines.append("")
            lines.extend(blocks)

    lines.append("=" * 80)
    return "\n".join(lines)


class ConsoleLLMObservationSink:
    def __init__(
        self,
        *,
        prompt_view: PromptViewMode = "compact",
        width: int = 88,
        truncation_chars: int = 600,
    ) -> None:
        self.prompt_view = prompt_view
        self.width = width
        self.truncation_chars = truncation_chars
        self._lock = asyncio.Lock()

    async def begin_llm_call(
        self, record: LLMObservationRecord, *, capture_mode: CaptureMode
    ) -> None:
        return None

    async def finish_llm_call(
        self, record: LLMObservationRecord, *, capture_mode: CaptureMode
    ) -> None:
        rendered = render_console_observation(
            record,
            prompt_view=self.prompt_view,
            width=self.width,
            truncation_chars=self.truncation_chars,
        )
        async with self._lock:
            await asyncio.to_thread(print, rendered)

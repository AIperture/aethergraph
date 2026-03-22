from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from textwrap import fill
from typing import Any, Literal, Protocol
import uuid

CaptureMode = Literal["metadata", "full"]
PromptViewMode = Literal["off", "compact", "truncated", "full"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            return repr(value)
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            return repr(value)
    return repr(value)


def sanitize_observation_value(value: Any) -> Any:
    return _json_safe(value)


def summarize_text(value: str | None, *, preview_chars: int = 240) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "length": len(value),
        "sha256": sha256(value.encode("utf-8")).hexdigest(),
        "preview": value[:preview_chars],
    }


def summarize_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = sanitize_observation_value(messages)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
    return {
        "count": len(messages),
        "sha256": sha256(payload.encode("utf-8")).hexdigest(),
        "preview": normalized[:3],
    }


@dataclass
class LLMObservationRecord:
    call_id: str
    created_at: str
    call_type: str
    provider: str
    model: str
    run_id: str | None = None
    graph_id: str | None = None
    user_id: str | None = None
    org_id: str | None = None
    session_id: str | None = None
    app_id: str | None = None
    agent_id: str | None = None
    node_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    profile_name: str | None = None
    call_name: str | None = None
    messages: list[dict[str, Any]] | None = None
    messages_preview: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    output_format: str | None = None
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    strict_schema: bool | None = None
    validate_json: bool | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    trace_payload: dict[str, Any] | None = None
    trace_payload_preview: dict[str, Any] | None = None
    raw_text: str | None = None
    raw_text_preview: dict[str, Any] | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None

    @classmethod
    def new(
        cls,
        *,
        call_type: str,
        provider: str,
        model: str,
        dimensions: dict[str, Any],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: str,
        json_schema: dict[str, Any] | None,
        schema_name: str | None,
        strict_schema: bool | None,
        validate_json: bool | None,
        extra_params: dict[str, Any],
        trace_payload: dict[str, Any] | None,
        profile_name: str | None = None,
        call_name: str | None = None,
    ) -> LLMObservationRecord:
        return cls(
            call_id=str(uuid.uuid4()),
            created_at=utc_now_iso(),
            call_type=call_type,
            provider=provider,
            model=model,
            run_id=dimensions.get("run_id"),
            graph_id=dimensions.get("graph_id"),
            user_id=dimensions.get("user_id"),
            org_id=dimensions.get("org_id"),
            session_id=dimensions.get("session_id"),
            app_id=dimensions.get("app_id"),
            agent_id=dimensions.get("agent_id"),
            node_id=dimensions.get("node_id"),
            trace_id=dimensions.get("trace_id"),
            span_id=dimensions.get("span_id"),
            profile_name=profile_name,
            call_name=call_name,
            messages=sanitize_observation_value(messages),
            messages_preview=summarize_messages(messages),
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=sanitize_observation_value(json_schema),
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            extra_params=sanitize_observation_value(extra_params) or {},
            trace_payload=sanitize_observation_value(trace_payload),
            trace_payload_preview=sanitize_observation_value(trace_payload),
        )

    def for_capture_mode(self, capture_mode: CaptureMode) -> dict[str, Any]:
        payload = asdict(self)
        if capture_mode == "full":
            payload["trace_payload_preview"] = payload["trace_payload"]
            payload["raw_text_preview"] = summarize_text(self.raw_text)
            return payload
        payload["messages"] = None
        payload["trace_payload"] = None
        payload["raw_text"] = None
        payload["raw_text_preview"] = summarize_text(self.raw_text)
        return payload


class LLMObservationSink(Protocol):
    async def emit(self, record: LLMObservationRecord, *, capture_mode: CaptureMode) -> None: ...


class JsonlLLMObservationSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def emit(self, record: LLMObservationRecord, *, capture_mode: CaptureMode) -> None:
        line = (
            json.dumps(record.for_capture_mode(capture_mode), ensure_ascii=False, default=str)
            + "\n"
        )
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _append_line(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)


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
        f"call_id: {record.call_id}",
        f"run_id:  {record.run_id or '-'}",
        f"graph:   {record.graph_id or '-'}",
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

    async def emit(self, record: LLMObservationRecord, *, capture_mode: CaptureMode) -> None:
        rendered = render_console_observation(
            record,
            prompt_view=self.prompt_view,
            width=self.width,
            truncation_chars=self.truncation_chars,
        )
        async with self._lock:
            await asyncio.to_thread(print, rendered)

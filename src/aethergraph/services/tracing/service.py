from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

from aethergraph.core.runtime.runtime_metering import current_meter_context


def utc_now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
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
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            return repr(value)
    return repr(value)


def _truncate_text(text: str, *, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _preview_value(value: Any, *, max_items: int = 5, max_text: int = 240) -> Any:
    safe = _json_safe(value)
    if isinstance(safe, str):
        return _truncate_text(safe, limit=max_text)
    if isinstance(safe, list):
        return [
            _preview_value(item, max_items=max_items, max_text=max_text)
            for item in safe[:max_items]
        ]
    if isinstance(safe, dict):
        keys = list(safe.keys())[:max_items]
        return {
            key: _preview_value(safe[key], max_items=max_items, max_text=max_text) for key in keys
        }
    return safe


def summarize_payload(value: Any) -> dict[str, Any]:
    safe = _json_safe(value)
    try:
        encoded = json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        encoded = json.dumps(repr(safe), ensure_ascii=False)
    metadata: dict[str, Any] = {"type": type(value).__name__ if value is not None else "NoneType"}
    if isinstance(safe, str):
        metadata["length"] = len(safe)
    elif isinstance(safe, list):
        metadata["count"] = len(safe)
    elif isinstance(safe, dict):
        metadata["count"] = len(safe)
        metadata["keys"] = list(safe.keys())[:10]
    return {
        "metadata": metadata,
        "preview": _preview_value(safe),
        "hashes": {
            "sha256": sha256(encoded.encode("utf-8")).hexdigest(),
        },
    }


def extract_metrics(value: Any) -> dict[str, int | float]:
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        return {}
    metrics: dict[str, int | float] = {}
    for key in ("bytes", "size", "size_bytes", "latency_ms", "duration_ms"):
        raw = safe.get(key)
        if isinstance(raw, (int, float)):
            metrics["bytes" if key in {"size", "size_bytes"} else key] = raw
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    ):
        raw = safe.get(key)
        if isinstance(raw, (int, float)):
            metrics[key] = raw
    return metrics


def current_trace_dimensions() -> dict[str, Any]:
    ctx = current_meter_context.get()
    return {
        "trace_id": ctx.get("trace_id"),
        "parent_span_id": ctx.get("span_id"),
        "run_id": ctx.get("run_id"),
        "graph_id": ctx.get("graph_id"),
        "session_id": ctx.get("session_id"),
        "node_id": ctx.get("node_id"),
        "agent_id": ctx.get("agent_id"),
        "app_id": ctx.get("app_id"),
        "user_id": ctx.get("user_id"),
        "org_id": ctx.get("org_id"),
    }


def resolve_tracer(explicit: Any | None = None) -> BaseTracer:
    if explicit is not None:
        return explicit
    try:
        from aethergraph.core.runtime.runtime_services import current_services

        tracer = getattr(current_services(), "tracer", None)
        if tracer is not None:
            return tracer
    except Exception:
        pass
    return NoopTracer()


class TracerProtocol(Protocol):
    async def start_span(
        self,
        *,
        service: str,
        operation: str,
        request: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> TraceSpan: ...


@dataclass
class TraceSpan:
    tracer: BaseTracer
    service: str
    operation: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    dims: dict[str, Any]
    started_at: float = field(default_factory=time.perf_counter)
    token: Token | None = None
    tags: list[str] = field(default_factory=list)
    finished: bool = False

    async def emit(
        self,
        *,
        phase: str,
        status: str,
        request: Any | None = None,
        response: Any | None = None,
        error: BaseException | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = int((time.perf_counter() - self.started_at) * 1000)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "phase": phase,
            "service": self.service,
            "operation": self.operation,
            "status": status,
            "duration_ms": duration_ms,
            "tags": self.tags,
            "request": summarize_payload(request) if request is not None else None,
            "response": summarize_payload(response) if response is not None else None,
            "error": (
                {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                if error is not None
                else None
            ),
            "metrics": extract_metrics(metrics),
        }
        payload.update({k: v for k, v in self.dims.items() if v is not None})
        if metadata:
            payload.update({k: _json_safe(v) for k, v in metadata.items() if v is not None})
        await self.tracer._append_trace_event(self.trace_id, payload)

    async def wait(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        request: Any | None = None,
    ) -> None:
        await self.emit(phase="wait", status="pending", metadata=metadata, request=request)

    async def resume(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        response: Any | None = None,
    ) -> None:
        await self.emit(phase="resume", status="resumed", metadata=metadata, response=response)

    async def finish(
        self,
        *,
        response: Any | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        await self.emit(
            phase="end",
            status="ok",
            response=response,
            metadata=metadata,
            metrics=metrics,
        )
        self._reset_context()

    async def fail(
        self,
        error: BaseException,
        *,
        metadata: dict[str, Any] | None = None,
        response: Any | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if self.finished:
            return
        self.finished = True
        await self.emit(
            phase="error",
            status="error",
            response=response,
            error=error,
            metadata=metadata,
            metrics=metrics,
        )
        self._reset_context()

    def _reset_context(self) -> None:
        if self.token is not None:
            current_meter_context.reset(self.token)
            self.token = None


class BaseTracer:
    async def start_span(
        self,
        *,
        service: str,
        operation: str,
        request: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> TraceSpan:
        dims = current_trace_dimensions()
        trace_id = str(dims.get("trace_id") or f"tr_{uuid.uuid4().hex}")
        parent_span_id = dims.get("parent_span_id")
        span_id = f"sp_{uuid.uuid4().hex}"
        next_ctx = dict(current_meter_context.get() or {})
        next_ctx["trace_id"] = trace_id
        next_ctx["parent_span_id"] = parent_span_id
        next_ctx["span_id"] = span_id
        token = current_meter_context.set(next_ctx)
        span = TraceSpan(
            tracer=self,
            service=service,
            operation=operation,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            dims=dims,
            token=token,
            tags=list(tags or []),
        )
        await span.emit(
            phase="start",
            status="ok",
            request=request,
            metadata=metadata,
            metrics=metrics,
        )
        return span

    @asynccontextmanager
    async def span(
        self,
        *,
        service: str,
        operation: str,
        request: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> AsyncIterator[TraceSpan]:
        span = await self.start_span(
            service=service,
            operation=operation,
            request=request,
            tags=tags,
            metadata=metadata,
            metrics=metrics,
        )
        try:
            yield span
        except Exception as exc:
            await span.fail(exc)
            raise
        else:
            await span.finish()

    async def _append_trace_event(self, trace_id: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError


class NoopTracer(BaseTracer):
    async def start_span(
        self,
        *,
        service: str,
        operation: str,
        request: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> TraceSpan:
        dims = current_trace_dimensions()
        trace_id = str(dims.get("trace_id") or f"tr_{uuid.uuid4().hex}")
        span_id = f"sp_{uuid.uuid4().hex}"
        next_ctx = dict(current_meter_context.get() or {})
        next_ctx["trace_id"] = trace_id
        next_ctx["parent_span_id"] = dims.get("parent_span_id")
        next_ctx["span_id"] = span_id
        token = current_meter_context.set(next_ctx)
        return TraceSpan(
            tracer=self,
            service=service,
            operation=operation,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=dims.get("parent_span_id"),
            dims=dims,
            token=token,
            tags=list(tags or []),
        )

    async def _append_trace_event(self, trace_id: str, payload: dict[str, Any]) -> None:
        return


class EventLogTracer(BaseTracer):
    def __init__(self, *, event_log: Any, event_hub: Any | None = None) -> None:
        self.event_log = event_log
        self.event_hub = event_hub

    async def _append_trace_event(self, trace_id: str, payload: dict[str, Any]) -> None:
        run_id = payload.get("run_id") or "unknown"
        row = {
            "id": f"evt_{uuid.uuid4().hex}",
            "ts": utc_now_ts(),
            "scope_id": f"trace:run/{run_id}",
            "kind": "trace",
            "payload": payload,
        }
        await self.event_log.append(row)
        if self.event_hub is not None:
            await self.event_hub.broadcast(row)

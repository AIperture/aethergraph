from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import Token
from dataclasses import dataclass, field
from hashlib import sha256
import json
import logging
from pathlib import Path
import time
from typing import Any, Protocol
import uuid

from aethergraph.core.runtime.runtime_metering import current_meter_context

from .models import ObservationRecord, ObservationScope


class ObservationSink(Protocol):
    async def append_observation(self, record: ObservationRecord, **kwargs: Any) -> str: ...


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
    """Build a bounded, hash-addressed preview for observation persistence."""
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
        "hashes": {"sha256": sha256(encoded.encode("utf-8")).hexdigest()},
    }


def extract_metrics(value: Any) -> dict[str, int | float]:
    """Extract the bounded numeric metrics supported by operation observations."""
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
        "cache_read_tokens",
        "cache_write_tokens",
        "uncached_input_tokens",
    ):
        raw = safe.get(key)
        if isinstance(raw, (int, float)):
            metrics[key] = raw
    return metrics


def _current_operation_dimensions() -> dict[str, Any]:
    context = current_meter_context.get() or {}
    return {
        "trace_id": context.get("trace_id"),
        "parent_span_id": context.get("span_id"),
        "run_id": context.get("run_id"),
        "graph_id": context.get("graph_id"),
        "session_id": context.get("session_id"),
        "node_id": context.get("node_id"),
        "agent_id": context.get("agent_id"),
        "app_id": context.get("app_id"),
        "user_id": context.get("user_id"),
        "org_id": context.get("org_id"),
    }


@dataclass
class OperationSpan:
    observer: OperationObserver
    service: str
    operation: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    dimensions: dict[str, Any]
    request: Any | None = None
    initial_metadata: dict[str, Any] = field(default_factory=dict)
    initial_metrics: dict[str, Any] = field(default_factory=dict)
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
        request_value = request if request is not None else self.request
        merged_metadata = {**self.initial_metadata, **dict(metadata or {})}
        merged_metrics = {**self.initial_metrics, **dict(metrics or {})}
        attributes: dict[str, Any] = {
            "schema_version": 1,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "phase": phase,
            "service": self.service,
            "operation": self.operation,
            "duration_ms": duration_ms,
            "tags": self.tags,
            "request": summarize_payload(request_value) if request_value is not None else None,
            "response": summarize_payload(response) if response is not None else None,
            "error": (
                {"type": type(error).__name__, "message": str(error)} if error is not None else None
            ),
            "metrics": extract_metrics(merged_metrics),
        }
        if merged_metadata:
            attributes.update(
                {
                    key: _json_safe(value)
                    for key, value in merged_metadata.items()
                    if value is not None
                }
            )
        record_status = "error" if status == "error" else "pending" if status == "pending" else "ok"
        observation_id = self.span_id if phase in {"end", "error"} else str(uuid.uuid4())
        await self.observer.append(
            ObservationRecord(
                observation_id=observation_id,
                category="service_operation",
                name=self.operation,
                summary=f"{self.service}/{self.operation} {phase}",
                scope=ObservationScope.from_dimensions(
                    {**self.dimensions, "trace_id": self.trace_id}
                ),
                status=record_status,
                severity="error" if error is not None else "info",
                attributes=attributes,
                parent_observation_id=self.parent_span_id,
            )
        )

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
        await self.emit(phase="resume", status="ok", metadata=metadata, response=response)

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
        try:
            await self.emit(
                phase="end",
                status="ok",
                response=response,
                metadata=metadata,
                metrics=metrics,
            )
        finally:
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
        try:
            await self.emit(
                phase="error",
                status="error",
                response=response,
                error=error,
                metadata=metadata,
                metrics=metrics,
            )
        finally:
            self._reset_context()

    def _reset_context(self) -> None:
        if self.token is not None:
            current_meter_context.reset(self.token)
            self.token = None


class OperationObserver:
    """Persist bounded service-operation observations through one canonical sink."""

    def __init__(self, sink: ObservationSink | None = None) -> None:
        self.sink = sink

    async def append(self, record: ObservationRecord) -> None:
        if self.sink is None:
            return
        try:
            await self.sink.append_observation(record)
        except Exception:
            logging.getLogger("aethergraph.observability.operations").warning(
                "Operation observation persistence failed",
                exc_info=True,
                extra={"observation_skip": True},
            )

    async def start_span(
        self,
        *,
        service: str,
        operation: str,
        request: Any | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> OperationSpan:
        """Start one operation observation span.

        Intro:
            Captures runtime dimensions in a context variable and returns a span
            whose terminal or wait lifecycle events persist through the configured
            observation sink.

        Examples:
            Record a successful operation:
            ```python
            span = await observer.start_span(service="runner", operation="submit")
            await span.finish(response={"run_id": "run-1"})
            ```

            Record an operation failure:
            ```python
            span = await observer.start_span(service="artifacts", operation="save")
            await span.fail(RuntimeError("write failed"))
            ```

        Args:
            service: Stable producer service name.
            operation: Stable operation name within the producer service.
            request: Optional request value summarized only when an event is emitted.
            tags: Optional bounded classification tags.
            metadata: Optional initial operation metadata.
            metrics: Optional initial numeric metrics.

        Returns:
            OperationSpan: Active span bound to the current runtime dimensions.

        Notes:
            Initial request, metadata, and metrics are merged into lifecycle events;
            persistence occurs on `wait`, `resume`, `finish`, or `fail`.
        """
        dimensions = _current_operation_dimensions()
        trace_id = str(dimensions.get("trace_id") or f"tr_{uuid.uuid4().hex}")
        parent_span_id = dimensions.get("parent_span_id")
        span_id = f"sp_{uuid.uuid4().hex}"
        next_context = dict(current_meter_context.get() or {})
        next_context["trace_id"] = trace_id
        next_context["parent_span_id"] = parent_span_id
        next_context["span_id"] = span_id
        token = current_meter_context.set(next_context)
        return OperationSpan(
            observer=self,
            service=service,
            operation=operation,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            dimensions=dimensions,
            request=request,
            initial_metadata=dict(metadata or {}),
            initial_metrics=dict(metrics or {}),
            token=token,
            tags=list(tags or []),
        )

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
    ) -> AsyncIterator[OperationSpan]:
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


_UNBOUND_OBSERVER = OperationObserver()


def resolve_operation_observer(explicit: OperationObserver | None = None) -> OperationObserver:
    """Resolve the active canonical operation observer for the current runtime."""
    if explicit is not None:
        return explicit
    try:
        from aethergraph.core.runtime.runtime_services import current_services

        sink = getattr(current_services(), "observation_sink", None)
        if sink is not None:
            return OperationObserver(sink)
    except Exception:
        pass
    return _UNBOUND_OBSERVER

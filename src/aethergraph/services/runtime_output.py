from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import sys
from typing import Any, TextIO
from uuid import uuid4

from aethergraph.contracts.services.runtime_output import (
    RuntimeOutputFrame,
    RuntimeOutputSink,
)

_capture_state: ContextVar[_CaptureState | None] = ContextVar(
    "aethergraph_runtime_output_capture",
    default=None,
)


@dataclass
class _RunBounds:
    rows: int = 0
    size_bytes: int = 0
    marker_emitted: bool = False


@dataclass
class _Barrier:
    future: asyncio.Future[None]


class EventLogRuntimeOutputSink:
    """Asynchronously persist bounded runtime output frames to the event log."""

    def __init__(
        self,
        *,
        event_log: Any,
        logger: Any = None,
        tags: tuple[str, ...] = (),
        max_line_bytes: int = 16 * 1024,
        max_run_bytes: int = 256 * 1024,
        max_rows_per_run: int = 1_000,
    ):
        self.event_log = event_log
        self.logger = logger
        self.tags = tuple(tags)
        self.max_line_bytes = max_line_bytes
        self.max_run_bytes = max_run_bytes
        self.max_rows_per_run = max_rows_per_run
        self._bounds: dict[str, _RunBounds] = {}
        self._queue: asyncio.Queue[RuntimeOutputFrame | _Barrier] | None = None
        self._worker_task: asyncio.Task[None] | None = None

    @staticmethod
    def _clip_utf8(text: str, limit: int) -> tuple[str, bool]:
        encoded = text.encode("utf-8")
        if len(encoded) <= limit:
            return text, False
        return encoded[:limit].decode("utf-8", errors="ignore"), True

    def _ensure_worker(self) -> asyncio.Queue[RuntimeOutputFrame | _Barrier]:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._run_worker())
        return self._queue

    def _bounded(self, frame: RuntimeOutputFrame) -> RuntimeOutputFrame | None:
        text = frame.text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        text, line_clipped = self._clip_utf8(text, self.max_line_bytes)
        if not text and not frame.truncated:
            return None

        bounds = self._bounds.setdefault(frame.run_id, _RunBounds())
        marks_truncation = (frame.truncated or line_clipped) and not bounds.marker_emitted
        if marks_truncation:
            bounds.marker_emitted = True
        frame = replace(frame, text=text, truncated=marks_truncation)
        size = len(frame.text.encode("utf-8"))
        if bounds.rows >= self.max_rows_per_run or bounds.size_bytes + size > self.max_run_bytes:
            if bounds.marker_emitted:
                return None
            bounds.marker_emitted = True
            return replace(
                frame,
                text="[runtime output truncated]",
                partial=False,
                truncated=True,
            )

        bounds.rows += 1
        bounds.size_bytes += size
        return frame

    def emit(self, frame: RuntimeOutputFrame) -> None:
        bounded = self._bounded(frame)
        if bounded is not None:
            self._ensure_worker().put_nowait(bounded)

    async def _run_worker(self) -> None:
        token = _capture_state.set(None)
        try:
            assert self._queue is not None
            while True:
                item = await self._queue.get()
                try:
                    if isinstance(item, _Barrier):
                        if not item.future.done():
                            item.future.set_result(None)
                        continue
                    await self.event_log.append(self._event_row(item))
                except Exception as exc:
                    if isinstance(item, _Barrier):
                        if not item.future.done():
                            item.future.set_result(None)
                    else:
                        self._report_persistence_failure(exc)
                finally:
                    self._queue.task_done()
        finally:
            _capture_state.reset(token)

    def _report_persistence_failure(self, exc: Exception) -> None:
        if self.logger is None:
            return
        warning = getattr(self.logger, "warning", None)
        if warning is not None:
            warning("Runtime output persistence failed: %s", exc, exc_info=True)

    def _event_row(self, frame: RuntimeOutputFrame) -> dict[str, Any]:
        scope_id = frame.session_id or frame.run_id
        return {
            "id": str(uuid4()),
            "ts": datetime.now(UTC).timestamp(),
            "scope_id": scope_id,
            "kind": "runtime_console",
            "session_id": frame.session_id,
            "run_id": frame.run_id,
            "graph_id": frame.graph_id,
            "node_id": frame.node_id,
            "tool": frame.tool_name,
            "tags": ["runtime-console", *self.tags],
            "payload": {
                "type": "runtime.console.output",
                "schema_version": "ag.runtime-console-output/v1",
                "execution_id": frame.execution_id,
                "stream": frame.stream,
                "text": frame.text,
                "sequence": frame.sequence,
                "partial": frame.partial,
                "truncated": frame.truncated,
                "meta": {
                    "source": frame.source,
                    "run_id": frame.run_id,
                    "session_id": frame.session_id,
                    "graph_id": frame.graph_id,
                    "node_id": frame.node_id,
                    "tool_name": frame.tool_name,
                    "eof": frame.eof,
                },
            },
        }

    async def _flush(self) -> None:
        if self._queue is None or self._worker_task is None:
            return
        future = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Barrier(future=future))
        await future

    async def flush_execution(self, execution_id: str) -> None:
        await self._flush()

    async def flush_run(self, run_id: str) -> None:
        await self._flush()

    async def close(self) -> None:
        await self._flush()
        if self._worker_task is not None:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None


@dataclass
class _CaptureState:
    sink: RuntimeOutputSink
    execution_id: str
    run_id: str
    session_id: str | None
    graph_id: str | None
    node_id: str
    tool_name: str | None
    sequence: int = 0
    stdout_buffer: str = ""
    stderr_buffer: str = ""

    def _emit(self, stream: str, text: str, *, partial: bool, eof: bool = False) -> None:
        if not text:
            return
        self.sequence += 1
        self.sink.emit(
            RuntimeOutputFrame(
                execution_id=self.execution_id,
                run_id=self.run_id,
                session_id=self.session_id,
                graph_id=self.graph_id,
                node_id=self.node_id,
                tool_name=self.tool_name,
                stream=stream,
                sequence=self.sequence,
                text=text,
                eof=eof,
                partial=partial,
            )
        )

    def write(self, stream: str, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        attr = "stdout_buffer" if stream == "stdout" else "stderr_buffer"
        buffer = getattr(self, attr) + normalized
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            self._emit(stream, line, partial=False)
        setattr(self, attr, buffer)

    def flush_stream(self, stream: str, *, eof: bool = False) -> None:
        attr = "stdout_buffer" if stream == "stdout" else "stderr_buffer"
        buffer = getattr(self, attr)
        if buffer:
            self._emit(stream, buffer, partial=True, eof=eof)
            setattr(self, attr, "")

    def finish(self) -> None:
        self.flush_stream("stdout", eof=True)
        self.flush_stream("stderr", eof=True)


class _RuntimeStreamProxy:
    def __init__(self, original: TextIO, stream: str):
        self.original = original
        self.stream = stream

    def write(self, value: str) -> int:
        state = _capture_state.get()
        if state is None:
            return self.original.write(value)
        state.write(self.stream, value)
        return len(value)

    def flush(self) -> None:
        state = _capture_state.get()
        if state is None:
            self.original.flush()
        else:
            state.flush_stream(self.stream)

    def isatty(self) -> bool:
        return self.original.isatty()

    def fileno(self) -> int:
        return self.original.fileno()

    @property
    def encoding(self) -> str | None:
        return self.original.encoding

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


_proxy_refcount = 0
_original_stdout: TextIO | None = None
_original_stderr: TextIO | None = None
_stdout_proxy: _RuntimeStreamProxy | None = None
_stderr_proxy: _RuntimeStreamProxy | None = None


class RuntimeStreamCaptureHandle:
    def __init__(self) -> None:
        self._closed = False

    def close(self) -> None:
        global _proxy_refcount, _original_stdout, _original_stderr
        global _stdout_proxy, _stderr_proxy
        if self._closed:
            return
        self._closed = True
        _proxy_refcount = max(0, _proxy_refcount - 1)
        if _proxy_refcount:
            return
        if sys.stdout is _stdout_proxy and _original_stdout is not None:
            sys.stdout = _original_stdout
        if sys.stderr is _stderr_proxy and _original_stderr is not None:
            sys.stderr = _original_stderr
        _original_stdout = None
        _original_stderr = None
        _stdout_proxy = None
        _stderr_proxy = None


def install_runtime_stream_capture() -> RuntimeStreamCaptureHandle:
    global _proxy_refcount, _original_stdout, _original_stderr
    global _stdout_proxy, _stderr_proxy
    if _proxy_refcount == 0:
        _original_stdout = sys.stdout
        _original_stderr = sys.stderr
        _stdout_proxy = _RuntimeStreamProxy(sys.stdout, "stdout")
        _stderr_proxy = _RuntimeStreamProxy(sys.stderr, "stderr")
        sys.stdout = _stdout_proxy
        sys.stderr = _stderr_proxy
    _proxy_refcount += 1
    return RuntimeStreamCaptureHandle()


@asynccontextmanager
async def capture_runtime_output(
    *,
    sink: RuntimeOutputSink | None,
    execution_id: str,
    run_id: str,
    session_id: str | None,
    graph_id: str | None,
    node_id: str,
    tool_name: str | None,
) -> AsyncIterator[None]:
    if sink is None:
        yield
        return

    state = _CaptureState(
        sink=sink,
        execution_id=execution_id,
        run_id=run_id,
        session_id=session_id,
        graph_id=graph_id,
        node_id=node_id,
        tool_name=tool_name,
    )
    token = _capture_state.set(state)
    try:
        yield
    finally:
        state.finish()
        _capture_state.reset(token)
        await sink.flush_execution(execution_id)


@dataclass
class RuntimeOutputCaptureHost:
    container: Any
    sink: EventLogRuntimeOutputSink
    streams: RuntimeStreamCaptureHandle

    async def flush_run(self, run_id: str) -> None:
        await self.sink.flush_run(run_id)

    async def close(self) -> None:
        if getattr(self.container, "runtime_output_sink", None) is self.sink:
            self.container.runtime_output_sink = None
        await self.sink.close()
        self.streams.close()


def enable_runtime_output_capture(
    container: Any,
    *,
    tags: tuple[str, ...] = (),
) -> RuntimeOutputCaptureHost:
    """Opt one host into Python stdout/stderr capture for active tools."""
    logger = getattr(container, "logger", None)
    if logger is not None and hasattr(logger, "for_service"):
        logger = logger.for_service(ns="runtime_output")
    sink = EventLogRuntimeOutputSink(
        event_log=container.eventlog,
        logger=logger,
        tags=tags,
    )
    container.runtime_output_sink = sink
    return RuntimeOutputCaptureHost(
        container=container,
        sink=sink,
        streams=install_runtime_stream_capture(),
    )

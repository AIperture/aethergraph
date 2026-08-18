from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import sys
from typing import Any, TextIO

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


class _RuntimeOutputBounds:
    def __init__(
        self,
        *,
        max_line_bytes: int,
        max_run_bytes: int,
        max_rows_per_run: int,
    ) -> None:
        self.max_line_bytes = max_line_bytes
        self.max_run_bytes = max_run_bytes
        self.max_rows_per_run = max_rows_per_run
        self._bounds: dict[str, _RunBounds] = {}

    def checkpoint(self, run_id: str) -> _RunBounds | None:
        current = self._bounds.get(run_id)
        return replace(current) if current is not None else None

    def restore(self, run_id: str, checkpoint: _RunBounds | None) -> None:
        if checkpoint is None:
            self._bounds.pop(run_id, None)
        else:
            self._bounds[run_id] = checkpoint

    @staticmethod
    def _clip_utf8(text: str, limit: int) -> tuple[str, bool]:
        encoded = text.encode("utf-8")
        if len(encoded) <= limit:
            return text, False
        return encoded[:limit].decode("utf-8", errors="ignore"), True

    def bounded(self, frame: RuntimeOutputFrame) -> RuntimeOutputFrame | None:
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
    sink: RuntimeOutputSink
    streams: RuntimeStreamCaptureHandle

    async def flush_run(self, run_id: str) -> None:
        await self.sink.flush_run(run_id)

    async def close(self) -> None:
        if getattr(self.container, "runtime_output_sink", None) is self.sink:
            self.container.runtime_output_sink = None
        self.streams.close()


def enable_runtime_output_capture(
    container: Any,
    *,
    tags: tuple[str, ...] = (),
) -> RuntimeOutputCaptureHost:
    """Opt one host into Python stdout/stderr capture for active tools."""
    sink = container.storage_services.runtime_output.with_tags(tags)
    container.runtime_output_sink = sink
    return RuntimeOutputCaptureHost(
        container=container,
        sink=sink,
        streams=install_runtime_stream_capture(),
    )

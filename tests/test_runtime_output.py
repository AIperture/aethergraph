from __future__ import annotations

import asyncio
from contextlib import contextmanager
from io import StringIO
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from aethergraph.contracts.services.runtime_output import RuntimeOutputFrame
from aethergraph.core.execution.retry_policy import RetryPolicy
from aethergraph.core.execution.step_forward import step_forward
from aethergraph.core.graph.graph_fn import GRAPH_FN_ROOT_NODE_ID, GraphFunction
from aethergraph.services.runtime_output import (
    EventLogRuntimeOutputSink,
    capture_runtime_output,
    enable_runtime_output_capture,
    install_runtime_stream_capture,
)


class _EventLog:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    async def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)


class _FailingEventLog:
    async def append(self, row: dict[str, Any]) -> None:
        raise OSError("event log unavailable")


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str, exc: Exception, **kwargs) -> None:
        self.warnings.append(message % exc)


@contextmanager
def _installed_stream_capture():
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    passthrough_stdout = StringIO()
    passthrough_stderr = StringIO()
    sys.stdout = passthrough_stdout
    sys.stderr = passthrough_stderr
    handle = install_runtime_stream_capture()
    try:
        yield passthrough_stdout, passthrough_stderr
    finally:
        handle.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


@pytest.mark.asyncio
async def test_capture_records_stdout_stderr_and_partial_line():
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(event_log=event_log)

    with _installed_stream_capture() as (passthrough_stdout, _):
        print("outside")
        async with capture_runtime_output(
            sink=sink,
            execution_id="exec-1",
            run_id="run-1",
            session_id="session-1",
            graph_id="graph-1",
            node_id="node-1",
            tool_name="demo",
        ):
            print("hello")
            print("problem", file=sys.stderr)
            print("partial", end="")

        assert passthrough_stdout.getvalue() == "outside\n"
    payloads = [row["payload"] for row in event_log.rows]
    assert [(payload["stream"], payload["text"]) for payload in payloads] == [
        ("stdout", "hello"),
        ("stderr", "problem"),
        ("stdout", "partial"),
    ]
    assert [payload["sequence"] for payload in payloads] == [1, 2, 3]
    assert payloads[-1]["partial"] is True
    assert event_log.rows[0]["scope_id"] == "session-1"
    assert event_log.rows[0]["node_id"] == "node-1"
    await sink.close()


@pytest.mark.asyncio
async def test_concurrent_captures_do_not_swap_attribution():
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(event_log=event_log)

    async def _worker(run_id: str, node_id: str) -> None:
        async with capture_runtime_output(
            sink=sink,
            execution_id=f"exec-{node_id}",
            run_id=run_id,
            session_id="session-1",
            graph_id="graph-1",
            node_id=node_id,
            tool_name=f"tool-{node_id}",
        ):
            print(f"{node_id}-first")
            await asyncio.sleep(0)
            print(f"{node_id}-second")

    with _installed_stream_capture():
        await asyncio.gather(_worker("run-a", "node-a"), _worker("run-b", "node-b"))

    by_node: dict[str, list[str]] = {}
    for row in event_log.rows:
        by_node.setdefault(row["node_id"], []).append(row["payload"]["text"])
        assert row["run_id"] == row["payload"]["meta"]["run_id"]
    assert by_node == {
        "node-a": ["node-a-first", "node-a-second"],
        "node-b": ["node-b-first", "node-b-second"],
    }
    await sink.close()


@pytest.mark.asyncio
async def test_fake_sandbox_frames_keep_execution_identity_and_order():
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(event_log=event_log)
    for execution_id, run_id, sequence, text in [
        ("sandbox-a", "run-a", 1, "a-1"),
        ("sandbox-b", "run-b", 1, "b-1"),
        ("sandbox-a", "run-a", 2, "a-2"),
        ("sandbox-b", "run-b", 2, "b-2"),
    ]:
        sink.emit(
            RuntimeOutputFrame(
                execution_id=execution_id,
                run_id=run_id,
                session_id="session-1",
                graph_id="graph-1",
                node_id=f"node-{run_id}",
                tool_name="sandboxed",
                stream="stdout",
                sequence=sequence,
                text=text,
                source="sandbox.stream",
            )
        )
    await sink.flush_run("run-a")

    by_execution: dict[str, list[tuple[int, str]]] = {}
    for row in event_log.rows:
        payload = row["payload"]
        by_execution.setdefault(payload["execution_id"], []).append(
            (payload["sequence"], payload["text"])
        )
        assert payload["meta"]["source"] == "sandbox.stream"
    assert by_execution == {
        "sandbox-a": [(1, "a-1"), (2, "a-2")],
        "sandbox-b": [(1, "b-1"), (2, "b-2")],
    }
    await sink.close()


@pytest.mark.asyncio
async def test_output_storm_emits_one_bounded_truncation_marker():
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(
        event_log=event_log,
        max_rows_per_run=2,
        max_run_bytes=1_000,
    )
    for sequence in range(1, 8):
        sink.emit(
            RuntimeOutputFrame(
                execution_id="exec-1",
                run_id="run-1",
                session_id=None,
                graph_id="graph-1",
                node_id="node-1",
                tool_name="noisy",
                stream="stdout",
                sequence=sequence,
                text=f"line-{sequence}",
            )
        )
    await sink.flush_run("run-1")

    assert len(event_log.rows) == 3
    markers = [row for row in event_log.rows if row["payload"]["truncated"]]
    assert len(markers) == 1
    assert markers[0]["payload"]["text"] == "[runtime output truncated]"
    await sink.close()


@pytest.mark.asyncio
async def test_long_lines_are_utf8_clipped_with_one_truncation_flag():
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(
        event_log=event_log,
        max_line_bytes=8,
    )
    for sequence in (1, 2):
        sink.emit(
            RuntimeOutputFrame(
                execution_id="exec-1",
                run_id="run-1",
                session_id=None,
                graph_id="graph-1",
                node_id="node-1",
                tool_name="wide",
                stream="stdout",
                sequence=sequence,
                text="éééééé",
            )
        )
    await sink.flush_run("run-1")

    assert [row["payload"]["text"] for row in event_log.rows] == ["éééé", "éééé"]
    assert [row["payload"]["truncated"] for row in event_log.rows] == [True, False]
    await sink.close()


@pytest.mark.asyncio
async def test_persistence_failure_does_not_escape_flush():
    logger = _Logger()
    sink = EventLogRuntimeOutputSink(event_log=_FailingEventLog(), logger=logger)
    sink.emit(
        RuntimeOutputFrame(
            execution_id="exec-1",
            run_id="run-1",
            session_id=None,
            graph_id="graph-1",
            node_id="node-1",
            tool_name="demo",
            stream="stdout",
            sequence=1,
            text="still successful",
        )
    )

    await sink.flush_run("run-1")
    assert logger.warnings == ["Runtime output persistence failed: event log unavailable"]
    await sink.close()


@pytest.mark.asyncio
async def test_host_capture_is_explicit_and_flushable_by_run():
    event_log = _EventLog()
    container = SimpleNamespace(
        eventlog=event_log,
        logger=None,
        runtime_output_sink=None,
    )
    host = enable_runtime_output_capture(container, tags=("studio-test",))
    try:
        assert container.runtime_output_sink is host.sink
        async with capture_runtime_output(
            sink=container.runtime_output_sink,
            execution_id="exec-host",
            run_id="run-host",
            session_id="session-host",
            graph_id="graph-host",
            node_id="node-host",
            tool_name="hosted",
        ):
            print("host output")
        await host.flush_run("run-host")
    finally:
        await host.close()

    assert container.runtime_output_sink is None
    assert event_log.rows[0]["tags"] == ["runtime-console", "studio-test"]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_step_forward_captures_sync_and_async_tool_prints(
    is_async: bool,
):
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(event_log=event_log)

    if is_async:

        async def _logic():
            print("async tool output")
            return {"value": "async"}

    else:

        def _logic():
            print("sync tool output")
            return {"value": "sync"}

    class _Context:
        run_id = "run-1"
        session_id = "session-1"
        graph_id = "graph-1"
        logger_factory = None
        runtime_output_sink = sink

        async def resolve_inputs(self, node):
            return {}

        def get_logic(self, logic):
            return logic

        def create_node_context(self, node):
            return SimpleNamespace()

    node = SimpleNamespace(
        logic=_logic,
        node_id="node-1",
        tool_name="demo",
        attempts=0,
    )
    with _installed_stream_capture():
        result = await step_forward(
            node=node,
            ctx=_Context(),
            retry_policy=RetryPolicy(),
        )

    assert result.outputs == {"value": "async" if is_async else "sync"}
    assert event_log.rows[0]["payload"]["text"] == (
        "async tool output" if is_async else "sync tool output"
    )
    assert event_log.rows[0]["tool"] == "demo"
    await sink.close()


@pytest.mark.asyncio
async def test_graph_function_captures_nested_async_tool_prints():
    event_log = _EventLog()
    sink = EventLogRuntimeOutputSink(event_log=event_log)

    async def _printing_tool() -> str:
        print("graph function tool output")
        return "ok"

    async def _graph_body() -> dict[str, str]:
        return {"result": await _printing_tool()}

    graph = GraphFunction(
        name="graph_fn.runtime.output",
        fn=_graph_body,
        outputs=["result"],
    )
    runtime_ctx = SimpleNamespace(
        run_id="run-graph-fn",
        session_id="session-graph-fn",
        graph_id=graph.name,
        runtime_output_sink=sink,
        create_node_context=lambda node: SimpleNamespace(),
    )
    env = SimpleNamespace(
        resume_payload=None,
        make_ctx=lambda **kwargs: runtime_ctx,
    )

    with _installed_stream_capture():
        result = await graph.run(env=env)

    assert result == {"result": "ok"}
    assert event_log.rows[0]["payload"]["text"] == "graph function tool output"
    assert event_log.rows[0]["node_id"] == GRAPH_FN_ROOT_NODE_ID
    assert event_log.rows[0]["tool"] == graph.name
    await sink.close()

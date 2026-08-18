from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.observability import (
    OperationObserver,
    resolve_operation_observer,
    summarize_payload,
)
from aethergraph.observability.models import ObservationRecord


def test_trace_payload_summary_hashes_and_preview() -> None:
    summary = summarize_payload({"prompt": "x" * 400, "count": 3})

    assert summary["metadata"]["type"] == "dict"
    assert summary["metadata"]["count"] == 2
    assert "sha256" in summary["hashes"]
    assert summary["preview"]["prompt"].endswith("...")


@pytest.mark.asyncio
async def test_operation_observer_persists_bounded_terminal_record() -> None:
    class Sink:
        def __init__(self) -> None:
            self.records: list[ObservationRecord] = []

        async def append_observation(
            self,
            record: ObservationRecord,
            **_: Any,
        ) -> str:
            self.records.append(record)
            return record.observation_id

    sink = Sink()
    observer = OperationObserver(sink)
    token = current_meter_context.set(
        {
            "run_id": "run-1",
            "session_id": "session-1",
            "graph_id": "graph-1",
            "node_id": "node-1",
        }
    )
    try:
        span = await observer.start_span(
            service="runner",
            operation="submit",
            request={"prompt": "x" * 400},
            metadata={"target_run_id": "run-2"},
        )
        await span.finish(
            response={"run_id": "run-2"},
            metrics={"latency_ms": 12, "ignored": "value"},
        )
    finally:
        current_meter_context.reset(token)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.category == "service_operation"
    assert record.name == "submit"
    assert record.scope.run_id == "run-1"
    assert record.scope.trace_id == span.trace_id
    assert record.status == "ok"
    assert record.attributes["phase"] == "end"
    assert record.attributes["target_run_id"] == "run-2"
    assert record.attributes["request"]["preview"]["prompt"].endswith("...")
    assert record.attributes["metrics"] == {"latency_ms": 12}


@pytest.mark.asyncio
async def test_runtime_operation_observer_resolves_canonical_write_sink() -> None:
    class Sink:
        def __init__(self) -> None:
            self.records: list[ObservationRecord] = []

        async def append_observation(self, record: ObservationRecord) -> str:
            self.records.append(record)
            return record.observation_id

    sink = Sink()

    with use_services(SimpleNamespace(observation_sink=sink)):
        span = await resolve_operation_observer().start_span(
            service="artifacts",
            operation="save",
            request={"image": "data:image/png;base64,c2Vuc2l0aXZl"},
        )
        await span.fail(RuntimeError("write failed"))

    assert len(sink.records) == 1
    assert sink.records[0].category == "service_operation"
    assert sink.records[0].status == "error"
    assert sink.records[0].attributes["request"]["preview"]["image"].startswith("data:image/png")


@pytest.mark.asyncio
async def test_operation_persistence_failure_does_not_fail_runtime_operation() -> None:
    class BrokenSink:
        async def append_observation(self, *_: Any, **__: Any) -> str:
            raise RuntimeError("store unavailable")

    original = current_meter_context.get()
    span = await OperationObserver(BrokenSink()).start_span(
        service="runner",
        operation="submit",
    )

    await span.finish(response={"run_id": "run-1"})

    assert span.finished is True
    assert current_meter_context.get() == original

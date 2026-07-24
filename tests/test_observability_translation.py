from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.services.observability import (
    LLMObservationRecord,
    ObservabilityFacade,
    ObservationPolicy,
    SQLiteObservationStore,
)
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.runs.sqlite_run_store import SQLiteRunStore


@pytest.mark.asyncio
async def test_v2_presenter_projects_semantic_events_and_metadata_context(
    tmp_path,
) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events" / "events.db"))
    engine_event_log = SqliteEventLog(str(tmp_path / "memory_events" / "events.db"))
    run_store = SQLiteRunStore(str(tmp_path / "runs" / "runs.db"))
    store = SQLiteObservationStore(
        tmp_path / "events" / "observability.db",
        policy=ObservationPolicy(capture_mode="metadata"),
    )
    await run_store.create(
        RunRecord(
            run_id="run-1",
            graph_id="graph-1",
            kind="graphfn",
            status=RunStatus.succeeded,
            started_at=datetime(2026, 7, 17, tzinfo=UTC),
            session_id="session-1",
            agent_id="planner",
        )
    )
    llm = LLMObservationRecord.new(
        call_type="chat",
        provider="openai",
        model="gpt-test",
        dimensions={"run_id": "run-1", "session_id": "session-1"},
        messages=[{"role": "user", "content": "choose a tool"}],
        reasoning_effort=None,
        max_output_tokens=None,
        output_format="text",
        json_schema=None,
        schema_name=None,
        strict_schema=None,
        validate_json=None,
        extra_params={},
        request_args={},
        provider_request_args={},
        compatibility_notes=[],
        trace_payload=None,
    )
    await store.append_llm_call(llm)

    async def append(
        event_id: str,
        kind: str,
        data: dict,
        *,
        text: str,
        ts: str,
        tool: str | None = None,
    ) -> None:
        resource_tags = [
            tag
            for link in data.get("resource_links") or []
            for tag in (
                f"resource:{link['resource_key']}",
                f"resource_relation:{link['relation']}",
            )
        ]
        await engine_event_log.append(
            {
                "event_id": event_id,
                "ts": ts,
                "run_id": "run-1",
                "session_id": "session-1",
                "graph_id": "graph-1",
                "kind": kind,
                "text": text,
                "tags": ["agent_engine", *resource_tags],
                "tool": tool,
                "data": {"event_kind": kind, "agent_instance_id": "planner", **data},
            }
        )

    await append(
        "request-1",
        "agent_engine.user_request",
        {"turn_id": "turn-1"},
        text="Analyze this",
        ts="2026-07-17T00:00:01+00:00",
    )
    await append(
        "decision-1",
        "agent_engine.decision",
        {
            "step_index": 1,
            "selected_action": {"tool_name": "search", "args": {"q": "AG"}},
            "prompt_manifest_id": llm.prompt_manifest_id,
            "llm_call_id": llm.llm_call_id,
            "new_context_entry_ids": ["request-1"],
            "dynamic_context_summary": {"entries": 1},
            "status": "selected",
        },
        text="Selected search",
        ts="2026-07-17T00:00:02+00:00",
        tool="search",
    )
    await append(
        "call-1",
        "agent_engine.tool_call",
        {
            "caused_by_event_id": "decision-1",
            "args": {"q": "AG"},
            "status": "started",
        },
        text="search started",
        ts="2026-07-17T00:00:03+00:00",
        tool="search",
    )
    await append(
        "result-1",
        "agent_engine.tool_result",
        {
            "caused_by_event_id": "call-1",
            "result": {"summary": "found"},
            "status": "completed",
            "resource_links": [
                {
                    "resource_key": "artifact:report-1",
                    "relation": "output",
                    "artifact_id": "report-1",
                }
            ],
        },
        text="search completed",
        ts="2026-07-17T00:00:04+00:00",
        tool="search",
    )
    await append(
        "plan-1",
        "agent_engine.plan_created",
        {
            "plan": {
                "version": 1,
                "goal": "Investigate",
                "status": "active",
                "steps": [],
            }
        },
        text="Plan created",
        ts="2026-07-17T00:00:05+00:00",
    )
    await append(
        "dispatch-1",
        "agent_engine.dispatch_entered",
        {
            "dispatch_token": "dispatch-1",
            "source_agent_instance_id": "planner",
            "target_agent_instance_id": "researcher",
            "status": "dispatched",
        },
        text="Dispatched researcher",
        ts="2026-07-17T00:00:06+00:00",
    )
    await append(
        "return-1",
        "agent_engine.return_intent",
        {
            "dispatch_token": "dispatch-1",
            "status": "completed",
        },
        text="Research completed",
        ts="2026-07-17T00:00:06.500000+00:00",
    )
    await event_log.append(
        {
            "ts": "2026-07-17T00:00:07+00:00",
            "run_id": "run-1",
            "kind": "meter.llm",
            "tags": ["meter"],
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "cache_read_tokens": 6,
            "uncached_input_tokens": 4,
        }
    )

    facade = ObservabilityFacade(
        store,
        event_log=event_log,
        engine_event_log=engine_event_log,
        run_store=run_store,
    )
    sessions = await facade.list_trace_sessions()
    assert sessions["items"][0]["latest_trace_id"] == "run-1"
    spans = (await facade.get_trace_spans("run-1"))["items"]
    assert {span["kind"] for span in spans} >= {
        "graph_turn",
        "react_cycle",
        "context_composition",
        "tool_call",
        "agent_dispatch",
    }
    tool = next(span for span in spans if span["kind"] == "tool_call")
    assert tool["payload"]["resource_links"][0]["resource_key"] == "artifact:report-1"
    dispatch = next(span for span in spans if span["span_id"] == "dispatch-1")
    assert dispatch["status"] == "completed"
    assert dispatch["payload"]["result_summary"] == "Research completed"
    assert (await facade.get_trace_plans("run-1"))["items"][0]["plan"]["goal"] == "Investigate"
    graph = await facade.get_trace_graph("run-1")
    assert graph["edges"]["dispatch-1"]["target_node_id"] == "researcher"
    resource_events = await facade.list_resource_events("artifact:report-1", relation="output")
    assert resource_events["engine_events"][0]["event_id"] == "result-1"
    usage = await facade.get_usage(run_id="run-1")
    assert usage["llm_calls"] == 1
    assert usage["tool_calls"] == 1
    assert usage["cache_read_tokens"] == 6
    snapshot = await facade.get_trace_context_snapshot("run-1", str(llm.prompt_manifest_id))
    assert snapshot["capture_mode"] == "metadata"
    assert snapshot["body"]["sections"] == []
    await facade.close()
    await event_log.close()
    await engine_event_log.close()


@pytest.mark.asyncio
async def test_v2_presenter_omits_context_span_when_capture_is_off(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events" / "events.db"))
    engine_event_log = SqliteEventLog(str(tmp_path / "memory_events" / "events.db"))
    run_store = SQLiteRunStore(str(tmp_path / "runs" / "runs.db"))
    store = SQLiteObservationStore(
        tmp_path / "events" / "observability.db",
        policy=ObservationPolicy(capture_mode="off"),
    )
    await run_store.create(
        RunRecord(
            run_id="run-off",
            graph_id="graph-1",
            kind="graphfn",
            status=RunStatus.succeeded,
            started_at=datetime(2026, 7, 17, tzinfo=UTC),
        )
    )
    await engine_event_log.append(
        {
            "event_id": "decision-off",
            "ts": "2026-07-17T00:00:00+00:00",
            "run_id": "run-off",
            "kind": "agent_engine.decision",
            "text": "Selected answer",
            "tags": ["agent_engine"],
            "data": {
                "event_kind": "agent_engine.decision",
                "step_index": 1,
                "prompt_manifest_id": "",
            },
        }
    )
    facade = ObservabilityFacade(
        store,
        event_log=event_log,
        engine_event_log=engine_event_log,
        run_store=run_store,
    )
    spans = (await facade.get_trace_spans("run-off"))["items"]
    assert [span["kind"] for span in spans] == ["react_cycle"]
    await facade.close()
    await event_log.close()
    await engine_event_log.close()

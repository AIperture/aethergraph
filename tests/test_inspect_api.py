from __future__ import annotations

from datetime import UTC, datetime
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import observability as observability_api
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.services.inspect import (
    AgentEventTypeRegistry,
    emit_agent_event,
    register_default_agent_event_types,
)
from aethergraph.services.observability import ObservabilityFacade, PurgeResult
from aethergraph.services.observability.logging import ObservationLogHandler


class FakeEventLog:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    async def append(self, evt: dict) -> None:
        self.rows.append(evt)

    async def query(
        self,
        *,
        scope_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        offset: int = 0,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        order_dir: str = "desc",
    ) -> list[dict]:
        out: list[dict] = []
        min_ts = since.timestamp() if since else None
        max_ts = until.timestamp() if until else None
        for row in self.rows:
            ts = float(row.get("ts") or 0.0)
            if scope_id is not None and row.get("scope_id") != scope_id:
                continue
            if kinds is not None and row.get("kind") not in kinds:
                continue
            if min_ts is not None and ts < min_ts:
                continue
            if max_ts is not None and ts > max_ts:
                continue
            if user_id is not None and row.get("user_id") != user_id:
                continue
            if org_id is not None and row.get("org_id") != org_id:
                continue
            for key, value in (
                ("run_id", run_id),
                ("session_id", session_id),
                ("agent_id", agent_id),
                ("graph_id", graph_id),
                ("node_id", node_id),
            ):
                if value is not None and row.get(key) != value:
                    break
            else:
                row_tags = set(row.get("tags") or [])
                if tags is not None and not row_tags.issuperset(tags):
                    continue
                out.append(row)
                continue
            continue
        out.sort(key=lambda row: float(row.get("ts") or 0), reverse=order_dir != "asc")
        out = out[offset:]
        if limit is not None:
            out = out[:limit]
        return out

    async def get_many(self, scope_id: str, event_ids: list[str]) -> list[dict]:
        return [
            row
            for row in self.rows
            if row.get("scope_id") == scope_id and row.get("id") in event_ids
        ]


class FakeRunManager:
    def __init__(self) -> None:
        self.status = RunStatus.failed

    async def get_record(self, run_id: str) -> RunRecord | None:
        if run_id != "run-1":
            return None
        return RunRecord(
            run_id="run-1",
            graph_id="graph-1",
            kind="graphfn",
            status=self.status,
            started_at=datetime(2026, 3, 11, tzinfo=UTC),
            user_id="u1",
            org_id="o1",
            session_id="sess-1",
            agent_id="agent-1",
            app_id="app-1",
        )

    async def get(self, run_id: str) -> RunRecord | None:
        return await self.get_record(run_id)

    async def list(self, **kwargs) -> list[RunRecord]:
        del kwargs
        record = await self.get_record("run-1")
        return [record] if record is not None else []


class FakeObservationStore:
    def __init__(self, llm_rows: list[dict], observation_rows: list[dict]) -> None:
        self.llm_rows = llm_rows
        self.observation_rows = observation_rows
        self.appended = []

    async def query_llm_calls(self, **kwargs):
        run_id = kwargs.get("run_id")
        session_id = kwargs.get("session_id")
        agent_id = kwargs.get("agent_id")
        app_id = kwargs.get("app_id")
        graph_id = kwargs.get("graph_id")
        node_id = kwargs.get("node_id")
        user_id = kwargs.get("user_id")
        org_id = kwargs.get("org_id")
        since = kwargs.get("since")
        until = kwargs.get("until")
        out = []
        for row in self.llm_rows:
            created_at = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
            if run_id and row["run_id"] != run_id:
                continue
            if session_id and row["session_id"] != session_id:
                continue
            if agent_id and row["agent_id"] != agent_id:
                continue
            if app_id and row["app_id"] != app_id:
                continue
            if graph_id and row["graph_id"] != graph_id:
                continue
            if node_id and row["node_id"] != node_id:
                continue
            if user_id and row["user_id"] != user_id:
                continue
            if org_id and row["org_id"] != org_id:
                continue
            if since and created_at < since:
                continue
            if until and created_at > until:
                continue
            sanitized = dict(row)
            sanitized.pop("messages", None)
            sanitized.pop("raw_text", None)
            sanitized.pop("trace_payload", None)
            out.append(sanitized)
        return out

    async def get_llm_call(self, call_id: str):
        for row in self.llm_rows:
            if call_id == row["call_id"]:
                return row
        return None

    async def list_observations(self, filters):
        return [
            row
            for row in self.observation_rows
            if (filters.category is None or row["category"] == filters.category)
            and (filters.run_id is None or row["run_id"] == filters.run_id)
            and (filters.session_id is None or row["session_id"] == filters.session_id)
            and (filters.user_id is None or row["user_id"] == filters.user_id)
            and (filters.org_id is None or row["org_id"] == filters.org_id)
        ]

    async def append_observation(self, record):
        self.appended.append(record)
        return record.observation_id

    async def list_suppressed_scopes(self):
        return {"session_id": set(), "run_id": set(), "trace_id": set()}

    async def delete_session_observations(self, session_id: str, *, dry_run: bool = False):
        del session_id
        return PurgeResult(dry_run, 0, 0, 0, 0, 0, 0)


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    eventlog = FakeEventLog()
    eventlog.rows.extend(
        [
            {
                "id": "trace-evt-1",
                "ts": 1.0,
                "scope_id": "trace:run/run-1",
                "kind": "trace",
                "user_id": "u1",
                "org_id": "o1",
                "payload": {
                    "trace_id": "tr_1",
                    "span_id": "sp_1",
                    "parent_span_id": None,
                    "service": "runner",
                    "operation": "submit",
                    "phase": "start",
                    "status": "ok",
                    "duration_ms": 10,
                    "request": {"preview": "req"},
                    "response": {"preview": "res"},
                    "metrics": {"duration_ms": 10},
                    "run_id": "run-1",
                    "session_id": "sess-1",
                    "graph_id": "graph-1",
                    "agent_id": "agent-1",
                    "app_id": "app-1",
                    "user_id": "u1",
                    "org_id": "o1",
                    "tags": ["submit"],
                },
            },
            {
                "id": "trace-evt-err",
                "ts": 2.0,
                "scope_id": "trace:run/run-1",
                "kind": "trace",
                "user_id": "u1",
                "org_id": "o1",
                "payload": {
                    "trace_id": "tr_1",
                    "span_id": "sp_2",
                    "parent_span_id": "sp_1",
                    "service": "runner",
                    "operation": "submit",
                    "phase": "error",
                    "status": "error",
                    "duration_ms": 12,
                    "error": {"type": "RuntimeError", "message": "boom"},
                    "run_id": "run-1",
                    "session_id": "sess-1",
                    "graph_id": "graph-1",
                    "agent_id": "agent-1",
                    "app_id": "app-1",
                    "user_id": "u1",
                    "org_id": "o1",
                    "tags": ["submit"],
                },
            },
            {
                "id": "log-evt-1",
                "ts": 3.0,
                "scope_id": "run-1",
                "kind": "inspect_log",
                "user_id": "u1",
                "org_id": "o1",
                "payload": {
                    "id": "log-evt-1",
                    "ts": 3.0,
                    "summary": "run failed",
                    "severity": "error",
                    "status": "error",
                    "producer": {"family": "logger", "name": "aethergraph.runtime"},
                    "scope": {
                        "run_id": "run-1",
                        "session_id": "sess-1",
                        "graph_id": "graph-1",
                        "agent_id": "agent-1",
                        "app_id": "app-1",
                        "trace_id": "tr_1",
                        "user_id": "u1",
                        "org_id": "o1",
                    },
                    "tags": ["error"],
                    "payload": {
                        "logger": "aethergraph.runtime",
                        "level": "error",
                        "message": "run failed",
                        "error": {"type": "RuntimeError", "message": "boom", "detail": "traceback"},
                        "extra": {"code": "E_RUN"},
                    },
                },
            },
            {
                "id": "log-evt-2",
                "ts": 4.0,
                "scope_id": "run-1",
                "kind": "inspect_log",
                "user_id": "u1",
                "org_id": "o1",
                "payload": {
                    "id": "log-evt-2",
                    "ts": 4.0,
                    "summary": "runner heartbeat",
                    "severity": "info",
                    "status": "info",
                    "producer": {"family": "logger", "name": "aethergraph.runner"},
                    "scope": {
                        "run_id": "run-1",
                        "session_id": "sess-1",
                        "graph_id": "graph-1",
                        "agent_id": "agent-1",
                        "app_id": "app-1",
                        "trace_id": "tr_1",
                        "user_id": "u1",
                        "org_id": "o1",
                    },
                    "tags": ["info"],
                    "payload": {
                        "logger": "aethergraph.runner",
                        "level": "info",
                        "message": "runner heartbeat",
                        "extra": {"code": "I_RUN"},
                    },
                },
            },
        ]
    )
    llm_rows = [
        {
            "call_id": "call-1",
            "created_at": "2026-03-11T00:00:00+00:00",
            "call_type": "chat",
            "provider": "openai",
            "model": "gpt-test",
            "run_id": "run-1",
            "session_id": "sess-1",
            "agent_id": "agent-1",
            "app_id": "app-1",
            "graph_id": "graph-1",
            "node_id": "node-1",
            "trace_id": "tr_1",
            "span_id": "sp_1",
            "user_id": "u1",
            "org_id": "o1",
            "messages_preview": {"count": 1},
            "messages": [{"role": "user", "content": "hello"}],
            "trace_payload_preview": {"step": "test"},
            "trace_payload": {"step": "test", "node": "node-1"},
            "raw_text_preview": {"length": 5},
            "raw_text": "world",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "latency_ms": 55,
            "reasoning_effort": "low",
            "output_format": "text",
            "error_type": None,
            "error_message": None,
        },
        {
            "call_id": "call-2",
            "created_at": "2026-03-11T00:05:00+00:00",
            "call_type": "chat",
            "provider": "anthropic",
            "model": "claude-test",
            "run_id": "run-1",
            "session_id": "sess-1",
            "agent_id": "agent-1",
            "app_id": "app-1",
            "graph_id": "graph-1",
            "node_id": "node-2",
            "trace_id": "tr_1",
            "span_id": "sp_2",
            "user_id": "u1",
            "org_id": "o1",
            "messages_preview": {"count": 2},
            "trace_payload_preview": {"step": "followup"},
            "raw_text_preview": {"length": 9},
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
            "latency_ms": 42,
            "reasoning_effort": "medium",
            "output_format": "json",
            "error_type": "RateLimitError",
            "error_message": "too many requests",
        },
    ]
    observation_rows = []
    for event_row in eventlog.rows:
        if event_row.get("kind") == "trace":
            payload = event_row["payload"]
            observation_rows.append(
                {
                    "observation_id": event_row["id"],
                    "category": "service_operation",
                    "name": payload["operation"],
                    "occurred_at": datetime.fromtimestamp(event_row["ts"], tz=UTC).isoformat(),
                    "summary": f"{payload['service']}/{payload['operation']}",
                    "severity": "error" if payload.get("error") else "info",
                    "status": payload["status"],
                    **{
                        key: payload.get(key)
                        for key in (
                            "run_id",
                            "session_id",
                            "graph_id",
                            "agent_id",
                            "app_id",
                            "trace_id",
                            "user_id",
                            "org_id",
                        )
                    },
                    "attributes": payload,
                }
            )
            continue
        if event_row.get("kind") != "inspect_log":
            continue
        payload = event_row["payload"]
        observation_rows.append(
            {
                "observation_id": payload["id"],
                "category": "log",
                "name": payload["producer"]["name"],
                "occurred_at": datetime.fromtimestamp(payload["ts"], tz=UTC).isoformat(),
                "summary": payload["summary"],
                "severity": payload["severity"],
                "status": "error" if payload["severity"] == "error" else "ok",
                **payload["scope"],
                "attributes": payload["payload"],
            }
        )

    class FakeContainer:
        pass

    run_manager = FakeRunManager()
    FakeContainer.run_manager = run_manager
    FakeContainer.eventlog = eventlog
    FakeContainer.observability = ObservabilityFacade(
        FakeObservationStore(llm_rows, observation_rows),
        event_log=eventlog,
        engine_event_log=eventlog,
        run_store=FakeContainer.run_manager,
        owns_store=False,
    )
    FakeContainer.agent_event_registry = register_default_agent_event_types(
        AgentEventTypeRegistry()
    )

    monkeypatch.setattr(
        "aethergraph.api.v1.observability.current_services", lambda: FakeContainer()
    )
    app = FastAPI()
    app.include_router(observability_api.router, prefix="/api/v1")
    app.include_router(observability_api.trace_router)

    async def fake_get_identity():
        return RequestIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[observability_api.get_identity] = fake_get_identity

    token = current_meter_context.set(
        {
            "run_id": "run-1",
            "session_id": "sess-1",
            "graph_id": "graph-1",
            "agent_id": "agent-1",
            "app_id": "app-1",
            "user_id": "u1",
            "org_id": "o1",
            "trace_id": "tr_1",
            "span_id": "sp_1",
        }
    )
    try:
        import asyncio

        asyncio.run(
            emit_agent_event(
                event_type="planning.started",
                summary="plan started",
                payload={"stage": 1},
                producer_name="deeplens",
                event_log=eventlog,
            )
        )
    finally:
        current_meter_context.reset(token)

    tc = TestClient(app)
    tc.fake_eventlog = eventlog
    tc.fake_run_manager = run_manager
    return tc


def test_get_run_trace(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/runs/run-1/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 2
    assert data["items"][0]["trace_id"] == "tr_1"


def test_connected_trace_router_uses_observability_facade(client: TestClient) -> None:
    resp = client.get("/api/trace/sessions")
    assert resp.status_code == 200
    assert resp.json()["items"][0]["latest_trace_id"] == "run-1"


def test_connected_trace_router_deletes_completed_session_observations(
    client: TestClient,
) -> None:
    deleted = client.delete("/api/trace/sessions/sess-1")
    batch = client.post(
        "/api/trace/sessions/delete",
        json={"session_ids": ["sess-1", "sess-1"]},
    )

    assert deleted.status_code == 204
    assert batch.status_code == 200
    assert batch.json() == {"deleted": 1}


def test_connected_trace_router_refuses_active_session_deletion(
    client: TestClient,
) -> None:
    client.fake_run_manager.status = RunStatus.waiting

    response = client.delete("/api/trace/sessions/sess-1")

    assert response.status_code == 409
    assert "active runs: run-1" in response.json()["detail"]


def test_get_run_trace_summary(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/runs/run-1/trace/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["span_count"] == 2
    assert data["error_count"] == 1
    assert data["top_failing_services"]["runner"] == 1


def test_get_run_llm_calls(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/runs/run-1/llm-calls")
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["provider"] == "anthropic"
    assert item["messages_preview"]["count"] == 2
    assert item["messages"] is None
    assert item["raw_text"] is None
    assert item["trace_payload"] is None


def test_get_llm_call_detail_includes_full_payload(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/llm-calls/call-1")
    assert resp.status_code == 200
    item = resp.json()
    assert item["messages"][0]["content"] == "hello"
    assert item["raw_text"] == "world"
    assert item["trace_payload"]["node"] == "node-1"


def test_get_run_logs_and_errors(client: TestClient) -> None:
    run_logs = client.get("/api/v1/inspect/runs/run-1/logs")
    assert run_logs.status_code == 200
    log_item = next(item for item in run_logs.json()["items"] if item["level"] == "error")
    assert log_item["message"] == "run failed"
    assert log_item["trace_status"] == "error"

    errors = client.get("/api/v1/inspect/errors")
    assert errors.status_code == 200
    error_item = errors.json()["items"][0]
    assert error_item["run_status"] == "failed"
    assert error_item["level"] == "error"


def test_get_run_agent_events(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/runs/run-1/agent-events")
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["event_type"] == "planning.started"
    assert item["producer"]["name"] == "deeplens"


def test_list_agent_event_types(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/agent-event-types")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(item["event_type"] == "planning.started" for item in items)


def test_list_global_traces_filters_and_ordering(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/traces?service=runner&service=memory&status=error")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["span_id"] == "trace-evt-err"


def test_list_global_llm_calls_filters_and_ordering(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/llm-calls?status=error")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["call_id"] == "call-2"

    all_calls = client.get("/api/v1/inspect/llm-calls")
    assert all_calls.status_code == 200
    assert all_calls.json()["items"][0]["call_id"] == "call-2"


def test_list_global_logs_filters_and_ordering(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/logs?level=info&logger=aethergraph.runner")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["message"] == "runner heartbeat"

    all_logs = client.get("/api/v1/inspect/logs")
    assert all_logs.status_code == 200
    assert all_logs.json()["items"][0]["id"] == "log-evt-2"


def test_list_agent_events_supports_time_window(client: TestClient) -> None:
    resp = client.get("/api/v1/inspect/agent-events?from=2100-01-01T00:00:00Z")
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 0


def test_observation_log_handler_emits_one_scoped_record() -> None:
    store = FakeObservationStore([], [])
    handler = ObservationLogHandler(store, level=logging.INFO)
    logger = logging.getLogger("aethergraph.test.inspect")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    token = current_meter_context.set(
        {
            "run_id": "run-1",
            "session_id": "sess-1",
            "graph_id": "graph-1",
            "agent_id": "agent-1",
            "app_id": "app-1",
            "user_id": "u1",
            "org_id": "o1",
            "trace_id": "tr_1",
            "span_id": "sp_1",
        }
    )
    try:
        logger.error("structured failure")
    finally:
        current_meter_context.reset(token)

    assert len(store.appended) == 1
    row = store.appended[0]
    assert row.category == "log"
    assert row.scope.run_id == "run-1"
    assert row.attributes["message"] == "structured failure"

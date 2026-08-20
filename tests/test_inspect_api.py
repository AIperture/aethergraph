from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from aethergraph.api.v1 import observability as observability_api
from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.observability import (
    AgentEventTypeRegistry,
    emit_agent_event,
    register_default_agent_event_types,
)
from aethergraph.observability.canonical_inspection import CanonicalInspectionReader
from aethergraph.observability.canonical_service import ProviderObservationService
from aethergraph.observability.contracts import InspectScope
from aethergraph.observability.logging import ObservationLogHandler
from aethergraph.observability.models import (
    LLMObservationRecord,
    ObservationFilter,
    ObservationRecord,
    ObservationScope,
)
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.storage.contracts import StorageOpenMode, StorageScope
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 3, 11, tzinfo=UTC)
SCOPE = ObservationScope(
    org_id="o1",
    user_id="u1",
    run_id="run-1",
    session_id="sess-1",
    agent_id="agent-1",
    app_id="app-1",
    graph_id="graph-1",
    node_id="node-1",
    trace_id="tr_1",
)
DIMENSIONS = {
    field.name: getattr(SCOPE, field.name)
    for field in fields(ObservationScope)
    if getattr(SCOPE, field.name) is not None
}


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
            started_at=NOW,
            user_id="u1",
            org_id="o1",
            session_id="sess-1",
            agent_id="agent-1",
            app_id="app-1",
        )


class CaptureObservationSink:
    def __init__(self) -> None:
        self.appended: list[ObservationRecord] = []

    async def append_observation(self, record: ObservationRecord) -> str:
        self.appended.append(record)
        return record.observation_id


def _observation(
    observation_id: str,
    *,
    occurred_at: datetime,
    category: str,
    name: str,
    summary: str,
    severity: str,
    status: str,
    attributes: dict,
) -> ObservationRecord:
    return ObservationRecord(
        observation_id=observation_id,
        occurred_at=occurred_at.isoformat(),
        category=category,
        name=name,
        summary=summary,
        severity=severity,
        status=status,
        scope=SCOPE,
        attributes=attributes,
    )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    service = ProviderObservationService(
        repository=LocalObservationRepository(database=database),
        owner_scope=StorageScope(project_id="project-1"),
        policy=ObservationPolicy(capture_mode="manifest"),
    )
    run_manager = FakeRunManager()

    async def resolve_statuses(run_ids: set[str]) -> dict[str, str]:
        return {run_id: "failed" for run_id in run_ids if run_id == "run-1"}

    async def seed() -> None:
        await service.append_observation(
            _observation(
                "trace-evt-1",
                occurred_at=NOW,
                category="service_operation",
                name="submit",
                summary="runner/submit",
                severity="info",
                status="ok",
                attributes={
                    "service": "runner",
                    "operation": "submit",
                    "phase": "start",
                    "duration_ms": 10,
                    "request": {"preview": "req"},
                    "response": {"preview": "res"},
                    "metrics": {"duration_ms": 10},
                },
            )
        )
        await service.append_observation(
            _observation(
                "trace-evt-err",
                occurred_at=NOW + timedelta(seconds=1),
                category="service_operation",
                name="submit",
                summary="runner/submit",
                severity="error",
                status="error",
                attributes={
                    "service": "runner",
                    "operation": "submit",
                    "phase": "error",
                    "duration_ms": 12,
                    "error": {"type": "RuntimeError", "message": "boom"},
                },
            )
        )
        await service.append_observation(
            _observation(
                "log-evt-1",
                occurred_at=NOW + timedelta(seconds=2),
                category="log",
                name="aethergraph.runtime",
                summary="run failed",
                severity="error",
                status="error",
                attributes={
                    "logger": "aethergraph.runtime",
                    "level": "error",
                    "message": "run failed",
                    "error": {"type": "RuntimeError", "message": "boom"},
                    "extra": {"code": "E_RUN"},
                },
            )
        )
        await service.append_observation(
            _observation(
                "log-evt-2",
                occurred_at=NOW + timedelta(seconds=3),
                category="log",
                name="aethergraph.runner",
                summary="runner heartbeat",
                severity="info",
                status="ok",
                attributes={
                    "logger": "aethergraph.runner",
                    "level": "info",
                    "message": "runner heartbeat",
                    "extra": {"code": "I_RUN"},
                },
            )
        )
        calls = (
            LLMObservationRecord(
                llm_call_id="call-1",
                created_at=NOW.isoformat(),
                call_type="chat",
                provider="openai",
                model="gpt-test",
                scope=SCOPE,
                messages=[{"role": "user", "content": "hello"}],
                reasoning_effort="low",
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
                trace_payload={"step": "test", "node": "node-1"},
                raw_text="world",
                usage={"input_tokens": 3, "output_tokens": 4},
                latency_ms=55,
            ),
            LLMObservationRecord(
                llm_call_id="call-2",
                created_at=(NOW + timedelta(minutes=5)).isoformat(),
                call_type="chat",
                provider="anthropic",
                model="claude-test",
                scope=SCOPE,
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                reasoning_effort="medium",
                max_output_tokens=None,
                output_format="json",
                json_schema=None,
                schema_name=None,
                strict_schema=None,
                validate_json=None,
                extra_params={},
                request_args={},
                provider_request_args={},
                compatibility_notes=[],
                trace_payload={"step": "followup"},
                raw_text="throttled",
                usage={"input_tokens": 6, "output_tokens": 2},
                latency_ms=42,
                error_type="RateLimitError",
                error_message="too many requests",
            ),
        )
        for call in calls:
            raw_text = call.raw_text
            usage = call.usage
            latency_ms = call.latency_ms
            error_type = call.error_type
            error_message = call.error_message
            call.raw_text = None
            call.usage = {}
            call.latency_ms = None
            call.error_type = None
            call.error_message = None
            await service.begin_llm_call(call, capture_mode="manifest")
            call.raw_text = raw_text
            call.usage = usage
            call.latency_ms = latency_ms
            call.error_type = error_type
            call.error_message = error_message
            call.lifecycle_status = "failed" if error_type else "completed"
            await service.finish_llm_call(call, capture_mode="manifest")

        token = current_meter_context.set(DIMENSIONS)
        try:
            await emit_agent_event(
                event_type="planning.started",
                summary="plan started",
                payload={"stage": 1},
                producer_name="deeplens",
                observation_sink=service,
            )
        finally:
            current_meter_context.reset(token)

    asyncio.run(seed())

    class StorageServices:
        def inspection(self, *, identity):
            return CanonicalInspectionReader(
                service,
                identity=identity,
                run_status_resolver=resolve_statuses,
            )

    container = SimpleNamespace(
        run_manager=run_manager,
        storage_services=StorageServices(),
        agent_event_registry=register_default_agent_event_types(AgentEventTypeRegistry()),
    )
    monkeypatch.setattr(observability_api, "current_services", lambda: container)
    app = FastAPI()
    app.include_router(observability_api.router, prefix="/api/v1")

    async def fake_get_identity() -> RequestIdentity:
        return RequestIdentity(user_id="u1", org_id="o1", mode="cloud")

    app.dependency_overrides[observability_api.get_identity] = fake_get_identity
    test_client = TestClient(app)
    test_client.fake_run_manager = run_manager
    yield test_client
    test_client.close()
    asyncio.run(database.close())


def test_get_run_trace(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/runs/run-1/trace")
    assert response.status_code == 200
    assert len(response.json()["items"]) == 2
    assert response.json()["items"][0]["trace_id"] == "tr_1"


def test_superseded_trace_routes_are_removed(client: TestClient) -> None:
    paths = {
        path for route in client.app.routes if (path := getattr(route, "path", None)) is not None
    }
    assert not any(path.startswith("/api/trace") for path in paths)


def test_get_run_trace_summary(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/runs/run-1/trace/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["span_count"] == 2
    assert data["error_count"] == 1
    assert data["top_failing_services"]["runner"] == 1
    assert data["trace_id_count"] == 1
    assert data["trace_ids_truncated"] is False


def test_get_run_llm_calls(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/runs/run-1/llm-calls")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["provider"] == "anthropic"
    assert item["messages_preview"]["message_count"] == 2
    assert item["messages"] is None
    assert item["raw_text"] is None
    assert item["trace_payload"] is None


def test_get_run_llm_summary_reports_truthful_breakdown_metadata(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/runs/run-1/llm-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_calls"] == 2
    assert data["total_prompt_tokens"] == 9
    assert data["total_completion_tokens"] == 6
    assert data["total_tokens"] == 15
    assert data["error_count"] == 1
    assert data["model_count"] == 2
    assert data["by_model_truncated"] is False


def test_get_llm_call_detail_includes_full_payload(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/llm-calls/call-1")
    assert response.status_code == 200
    item = response.json()
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
    assert errors.json()["items"][0]["run_status"] == "failed"
    assert errors.json()["items"][0]["level"] == "error"


def test_get_run_agent_events(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/runs/run-1/agent-events")
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["event_type"] == "planning.started"
    assert item["producer"]["name"] == "deeplens"


def test_list_agent_event_types(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/agent-event-types")
    assert response.status_code == 200
    assert any(item["event_type"] == "planning.started" for item in response.json()["items"])


def test_list_global_traces_filters_and_ordering(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/traces?service=runner&service=memory&status=error")
    assert response.status_code == 200
    assert [item["span_id"] for item in response.json()["items"]] == ["trace-evt-err"]


def test_list_global_llm_calls_filters_and_ordering(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/llm-calls?status=error")
    assert response.status_code == 200
    assert [item["call_id"] for item in response.json()["items"]] == ["call-2"]
    all_calls = client.get("/api/v1/inspect/llm-calls")
    assert all_calls.json()["items"][0]["call_id"] == "call-2"


def test_list_global_logs_filters_and_ordering(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/logs?level=info&logger=aethergraph.runner")
    assert response.status_code == 200
    assert [item["message"] for item in response.json()["items"]] == ["runner heartbeat"]
    all_logs = client.get("/api/v1/inspect/logs")
    assert all_logs.json()["items"][0]["id"] == "log-evt-2"


def test_list_agent_events_supports_time_window(client: TestClient) -> None:
    response = client.get("/api/v1/inspect/agent-events?from=2100-01-01T00:00:00Z")
    assert response.status_code == 200
    assert response.json()["items"] == []


def test_inspect_app_id_is_explicit_deprecated_compatibility_metadata(
    client: TestClient,
) -> None:
    for record_type in (ObservationScope, ObservationFilter):
        app_field = next(item for item in fields(record_type) if item.name == "app_id")
        assert app_field.metadata["deprecated"] is True
        assert app_field.metadata["compatibility_only"] is True
    assert InspectScope.model_fields["app_id"].deprecated is True
    schema = client.app.openapi()
    for path in (
        "/api/v1/inspect/traces",
        "/api/v1/inspect/llm-calls",
        "/api/v1/inspect/logs",
        "/api/v1/inspect/errors",
        "/api/v1/inspect/agent-events",
    ):
        parameters = schema["paths"][path]["get"]["parameters"]
        app_parameter = next(item for item in parameters if item["name"] == "app_id")
        assert app_parameter["deprecated"] is True
        assert "compatibility metadata" in app_parameter["description"]


def test_observation_log_handler_emits_one_scoped_record() -> None:
    sink = CaptureObservationSink()
    handler = ObservationLogHandler(sink, level=logging.INFO)
    logger = logging.getLogger("aethergraph.test.inspect")
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    token = current_meter_context.set(DIMENSIONS)
    try:
        logger.error("structured failure")
    finally:
        current_meter_context.reset(token)
    assert len(sink.appended) == 1
    row = sink.appended[0]
    assert row.category == "log"
    assert row.scope.run_id == "run-1"
    assert row.attributes["message"] == "structured failure"


def test_presentation_formatter_does_not_invent_persisted_scope() -> None:
    from aethergraph.observability.logger.formatters import SafeFormatter

    record = logging.LogRecord(
        name="aethergraph.test.inspect",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="scope-free",
        args=(),
        exc_info=None,
    )
    assert SafeFormatter("%(message)s app=%(app_id)s client=%(client_id)s").format(record) == (
        "scope-free app=- client=-"
    )

    observation = ObservationLogHandler._to_observation(record)

    assert observation.scope.app_id is None
    assert not hasattr(record, "app_id")
    assert not hasattr(record, "client_id")

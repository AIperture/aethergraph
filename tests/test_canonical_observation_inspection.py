from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

import aethergraph.observability.canonical_inspection as canonical_inspection
from aethergraph.observability.canonical_inspection import CanonicalInspectionReader
from aethergraph.observability.canonical_service import ProviderObservationService
from aethergraph.observability.inspection import (
    ObservabilityIdentity,
    ObservabilityNotFoundError,
)
from aethergraph.observability.models import (
    LLMObservationRecord,
    ObservationRecord,
    ObservationScope,
)
from aethergraph.observability.policy import ObservationPolicy
from aethergraph.storage.contracts import (
    StorageConfigurationError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalObservationRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 16, tzinfo=UTC)
OWNER = StorageScope(project_id="project-1")
SCOPE = ObservationScope(
    project_id="project-1",
    org_id="org-1",
    user_id="user-1",
    app_id="legacy-app",
    session_id="session-1",
    run_id="run-1",
    graph_id="graph-1",
    node_id="node-1",
    agent_id="agent-1",
    trace_id="trace-1",
    turn_id="turn-1",
)


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _service(repository: LocalObservationRepository) -> ProviderObservationService:
    return ProviderObservationService(
        repository=repository,
        owner_scope=OWNER,
        policy=ObservationPolicy(capture_mode="manifest"),
    )


def _observation(
    observation_id: str,
    *,
    category: str,
    occurred_at: datetime,
    attributes: dict,
    status: str = "ok",
    severity: str = "info",
) -> ObservationRecord:
    return ObservationRecord(
        observation_id=observation_id,
        category=category,
        name=str(attributes.get("operation") or observation_id),
        summary=observation_id,
        occurred_at=occurred_at.isoformat(),
        scope=SCOPE,
        status=status,
        severity=severity,
        attributes=attributes,
    )


@pytest.mark.asyncio
async def test_canonical_trace_reader_uses_promoted_filters_and_provider_cursor(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    service = _service(LocalObservationRepository(database=database))
    await service.append_observation(
        _observation(
            "trace-runner",
            category="service_operation",
            occurred_at=NOW,
            attributes={
                "service": "runner",
                "operation": "submit",
                "phase": "end",
                "duration_ms": 7,
            },
        )
    )
    await service.append_observation(
        _observation(
            "trace-memory",
            category="trace",
            occurred_at=NOW + timedelta(seconds=1),
            status="error",
            severity="error",
            attributes={
                "service": "memory",
                "operation": "search",
                "phase": "end",
                "duration_ms": 5,
                "error": {"type": "RuntimeError", "message": "failed"},
            },
        )
    )
    reader = CanonicalInspectionReader(service)

    first = await reader.list_traces(limit=1)
    second = await reader.list_traces(limit=1, cursor=first.next_cursor)
    assert [item.service for item in (*first.items, *second.items)] == ["memory", "runner"]
    assert first.next_cursor is not None and second.next_cursor is None
    filtered = await reader.list_traces(service=["runner"], app_id="legacy-app")
    assert [item.operation for item in filtered.items] == ["submit"]
    assert filtered.items[0].scope.model_dump()["app_id"] == "legacy-app"
    assert "compatibility_metadata" not in filtered.items[0].payload
    summary = await reader.summarize_traces("run-1")
    assert summary.span_count == 2
    assert summary.error_count == 1
    assert summary.total_duration_ms == 12
    assert summary.trace_ids == ["trace-1"]
    assert summary.trace_id_count == 1
    assert not summary.trace_ids_truncated
    assert summary.top_failing_services == {"memory": 1}
    assert summary.latest_error_ts == (NOW + timedelta(seconds=1)).timestamp()
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await reader.list_traces(
            service=["runner"],
            limit=1,
            cursor=first.next_cursor,
        )
    await database.close()


@pytest.mark.asyncio
async def test_canonical_llm_reader_separates_bounded_list_and_exact_detail(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    service = _service(LocalObservationRepository(database=database))
    call = LLMObservationRecord(
        llm_call_id="call-1",
        created_at=NOW.isoformat(),
        call_type="chat",
        provider="openai",
        model="gpt-test",
        scope=SCOPE,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        max_output_tokens=100,
        output_format="text",
        json_schema=None,
        schema_name=None,
        strict_schema=None,
        validate_json=None,
        extra_params={},
        request_args={"temperature": 0},
        provider_request_args={"temperature": 0},
        compatibility_notes=["normalized"],
        trace_payload={"step": "done"},
        raw_text="hello back",
        usage={"input_tokens": 2, "output_tokens": 2},
        latency_ms=10,
        tool_surface={
            "schema_version": "aethergraph.llm-tool-surface/v1",
            "discovery_mode": "native_client",
            "catalog_fingerprint": "catalog-1",
            "surface_fingerprint": "surface-1",
            "active_count": 1,
            "tools": [
                {
                    "ordinal": 0,
                    "name": "read",
                    "exposure": "deferred",
                    "active": True,
                    "input_schema_digest": "schema-1",
                }
            ],
        },
        request_items=[{"ordinal": 0, "kind": "tool_output", "call_id": "call-a"}],
        response_items=[{"ordinal": 0, "kind": "tool_call", "tool_name": "read"}],
        tool_definitions=[
            {
                "name": "read",
                "description": "Read.",
                "input_schema": {"type": "object"},
                "exposure": "immediate",
                "path": None,
            }
        ],
    )
    call.raw_text = None
    call.response_items = []
    await service.begin_llm_call(call, capture_mode="manifest")
    call.raw_text = "hello back"
    call.response_items = [{"ordinal": 0, "kind": "tool_call", "tool_name": "read"}]
    call.lifecycle_status = "completed"
    await service.finish_llm_call(call, capture_mode="manifest")
    reader = CanonicalInspectionReader(service)

    page = await reader.list_llm_calls(
        provider="openai",
        model="gpt-test",
        call_type="chat",
        app_id="legacy-app",
    )
    assert len(page.items) == 1
    assert page.items[0].messages is None
    assert page.items[0].raw_text is None
    assert page.items[0].tool_surface is not None
    assert page.items[0].tool_surface["schema_version"] == ("aethergraph.llm-tool-surface/v2")
    assert page.items[0].tool_surface["callable_count"] == 1
    assert page.items[0].tool_surface["activated_deferred_count"] == 1
    assert page.items[0].tool_surface["searchable_count"] == 0
    assert page.items[0].tool_surface["tools"][0]["callable"] is True
    assert "active" not in page.items[0].tool_surface["tools"][0]
    assert page.items[0].response_items[0]["tool_name"] == "read"
    detail = await reader.get_llm_call("call-1", required_run_id="run-1")
    assert detail.messages == [{"role": "user", "content": "hello"}]
    assert detail.raw_text == "hello back"
    assert detail.trace_payload == {"step": "done"}
    assert detail.reasoning_effort == "low"
    assert detail.tools is not None
    assert detail.tools[0]["name"] == "read"
    summary = await reader.summarize_llm_calls("run-1")
    assert summary.total_calls == 1
    assert summary.total_prompt_tokens == 2
    assert summary.total_completion_tokens == 2
    assert summary.total_tokens == 4
    assert summary.by_model == {"gpt-test": 1}
    assert summary.model_count == 1
    assert not summary.by_model_truncated
    with pytest.raises(ObservabilityNotFoundError, match="not found"):
        await reader.get_llm_call("call-1", required_run_id="other")
    await database.close()


@pytest.mark.asyncio
async def test_canonical_log_reader_bounds_enrichment_and_identity(tmp_path: Path) -> None:
    database = _database(tmp_path)
    service = _service(LocalObservationRepository(database=database))
    await service.append_observation(
        _observation(
            "log-error",
            category="log",
            occurred_at=NOW,
            status="error",
            severity="error",
            attributes={
                "logger": "aethergraph.runner",
                "level": "error",
                "message": "failed",
                "error": {"type": "RuntimeError", "message": "failed"},
            },
        )
    )
    await service.append_observation(
        _observation(
            "log-info",
            category="log",
            occurred_at=NOW + timedelta(seconds=1),
            attributes={
                "logger": "aethergraph.memory",
                "level": "info",
                "message": "ready",
            },
        )
    )

    async def resolve_statuses(run_ids: set[str]) -> dict[str, str]:
        assert run_ids == {"run-1"}
        return {"run-1": "failed"}

    reader = CanonicalInspectionReader(service, run_status_resolver=resolve_statuses)
    page = await reader.list_logs(
        level="error",
        logger="aethergraph.runner",
        run_status="failed",
        trace_status="error",
    )
    assert [item.message for item in page.items] == ["failed"]
    assert page.items[0].run_status == "failed"
    assert page.items[0].trace_status == "error"
    errors = await reader.list_logs(levels=("warning", "error", "critical"))
    assert [item.message for item in errors.items] == ["failed"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        await reader.list_logs(level="error", levels=("critical",))

    hidden = CanonicalInspectionReader(
        service,
        identity=ObservabilityIdentity(mode="cloud", user_id="other", org_id="org-1"),
    )
    assert not (await hidden.list_logs()).items
    await database.close()


def test_canonical_inspection_public_docstrings_are_strict() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    methods = (
        CanonicalInspectionReader.__init__,
        CanonicalInspectionReader.list_traces,
        CanonicalInspectionReader.list_llm_calls,
        CanonicalInspectionReader.get_llm_call,
        CanonicalInspectionReader.summarize_traces,
        CanonicalInspectionReader.summarize_llm_calls,
        CanonicalInspectionReader.list_logs,
        CanonicalInspectionReader.list_agent_events,
    )
    for method in methods:
        docstring = inspect.getdoc(method) or ""
        positions = tuple(docstring.find(section) for section in required)
        assert all(position >= 0 for position in positions), method.__name__
        assert positions == tuple(sorted(positions)), method.__name__
        assert docstring.count("```python") >= 2, method.__name__

    source = inspect.getsource(canonical_inspection)
    for forbidden in (
        "limit=None",
        "decode_cursor",
        "_paginate_rows",
        "SQLiteObservationStore",
        "hydrate_prompt_manifest",
    ):
        assert forbidden not in source

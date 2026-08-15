from __future__ import annotations

from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from typing import get_type_hints

import pytest

from aethergraph.storage.contracts import (
    RunQuery,
    RunRecord,
    RunRepository,
    RunResultRecord,
    RunResultRepository,
    RunStatus,
    SessionKind,
    SessionQuery,
    SessionRecord,
    SessionRepository,
    StorageBundle,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
RUN_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    session_id="session-1",
    run_id="run-1",
    graph_id="graph-1",
)
SESSION_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    session_id="session-1",
)


def _running_run() -> RunRecord:
    return RunRecord(
        run_id="run-1",
        graph_id="graph-1",
        kind="taskgraph",
        status=RunStatus.RUNNING,
        scope=RUN_SCOPE,
        revision=1,
        started_at=NOW,
        tags=("workflow",),
        metadata={"source": "test"},
    )


def test_run_records_enforce_scope_lifecycle_and_counter_consistency() -> None:
    running = _running_run()
    completed = replace(
        running,
        status=RunStatus.SUCCEEDED,
        revision=2,
        finished_at=NOW + timedelta(minutes=1),
        result_available=True,
        result_updated_at=NOW + timedelta(minutes=1),
    )

    assert completed.status is RunStatus.SUCCEEDED
    assert completed.metadata["source"] == "test"
    assert "app_id" not in {item.name for item in fields(RunRecord)}
    with pytest.raises(ValueError, match="terminal"):
        replace(running, status=RunStatus.FAILED)
    with pytest.raises(ValueError, match="artifact timestamps"):
        replace(running, artifact_count=1)


def test_run_result_requires_success_and_deep_freezes_outputs() -> None:
    outputs = {"items": [1, 2]}
    result = RunResultRecord(
        run_id="run-1",
        graph_id="graph-1",
        scope=RUN_SCOPE,
        status=RunStatus.SUCCEEDED,
        outputs=outputs,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        source="direct",
    )
    outputs["items"].append(3)

    assert result.outputs["items"] == (1, 2)
    with pytest.raises(ValueError, match="succeeded"):
        replace(result, status=RunStatus.FAILED)


def test_session_records_enforce_scope_time_and_artifact_consistency() -> None:
    session = SessionRecord(
        session_id="session-1",
        kind=SessionKind.CHAT,
        scope=SESSION_SCOPE,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        title="Migration",
        metadata={"channel": "ui"},
    )

    assert session.metadata["channel"] == "ui"
    assert "app_id" not in {item.name for item in fields(SessionRecord)}
    with pytest.raises(ValueError, match="agree"):
        replace(session, artifact_count=1)
    with pytest.raises(ValueError, match="match canonical scope"):
        replace(session, session_id="other")


def test_control_queries_are_bounded_and_deduplicated() -> None:
    assert RunQuery(scope=RUN_SCOPE, statuses=(RunStatus.RUNNING,)).statuses == (RunStatus.RUNNING,)
    assert SessionQuery(scope=SESSION_SCOPE, kinds=(SessionKind.CHAT,)).kinds == (SessionKind.CHAT,)
    with pytest.raises(ValueError, match="duplicates"):
        RunQuery(scope=RUN_SCOPE, statuses=(RunStatus.RUNNING, RunStatus.RUNNING))
    with pytest.raises(ValueError, match="duplicates"):
        SessionQuery(scope=SESSION_SCOPE, kinds=(SessionKind.CHAT, SessionKind.CHAT))


def test_bundle_exposes_control_repositories_by_exact_protocol() -> None:
    hints = get_type_hints(StorageBundle)

    assert hints["runs"] is RunRepository
    assert hints["run_results"] is RunResultRepository
    assert hints["sessions"] is SessionRepository


def test_control_protocol_docstrings_follow_required_section_order() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for protocol in (RunRepository, RunResultRepository, SessionRepository):
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)

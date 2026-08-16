from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from aethergraph.observability.legacy_cleanup import (
    LegacyObservabilityContainmentError,
    LegacyObservabilityWorkspaceActiveError,
    cleanup_legacy_observability,
    scan_legacy_observability,
)
from aethergraph.server.server_state import workspace_lock


def _legacy_trace_event(event_id: str = "legacy-trace") -> dict:
    return {
        "id": event_id,
        "ts": 1.0,
        "scope_id": "trace:run/run-1",
        "kind": "trace",
        "tags": ["service"],
        "payload": {
            "schema_version": 1,
            "trace_id": "trace-1",
            "span_id": "span-1",
            "parent_span_id": None,
            "phase": "start",
            "service": "llm",
            "operation": "chat",
            "status": "ok",
        },
    }


def _seed_workspace(workspace: Path) -> None:
    trace_dir = workspace / "trace"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace.sqlite3").write_bytes(b"trace-main")
    (trace_dir / "trace.sqlite3-wal").write_bytes(b"trace-wal")
    (trace_dir / "trace.sqlite3-shm").write_bytes(b"trace-shm")
    llm_dir = workspace / "events" / "llm"
    llm_dir.mkdir(parents=True)
    (llm_dir / "llm_calls.jsonl").write_text('{"legacy":true}\n', encoding="utf-8")

    event_db = workspace / "events" / "events.db"
    with sqlite3.connect(event_db) as connection:
        connection.execute(
            "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, kind TEXT, payload TEXT NOT NULL)"
        )
        rows = [
            _legacy_trace_event(),
            {
                "id": "custom-trace",
                "ts": 2.0,
                "scope_id": "custom",
                "kind": "trace",
                "payload": {"schema_version": 2, "purpose": "user-defined"},
            },
            {
                "id": "canonical",
                "ts": 3.0,
                "scope_id": "run-1",
                "kind": "agent_engine.decision",
                "payload": {"selected_action": "tool"},
            },
        ]
        connection.executemany(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            [(float(row["ts"]), str(row["kind"]), json.dumps(row)) for row in rows],
        )


def test_scan_reports_only_fixed_legacy_files_and_exact_generic_trace_shape(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _seed_workspace(workspace)
    unrelated = workspace / "other" / "llm_calls.jsonl"
    unrelated.parent.mkdir()
    unrelated.write_text("keep", encoding="utf-8")

    report = scan_legacy_observability(workspace)

    assert {candidate.relative_path for candidate in report.files} == {
        "trace/trace.sqlite3",
        "trace/trace.sqlite3-wal",
        "trace/trace.sqlite3-shm",
        "events/llm/llm_calls.jsonl",
    }
    assert report.file_bytes == sum(candidate.physical_bytes for candidate in report.files)
    assert report.event_rows[0].row_count == 1
    assert report.event_rows[0].logical_bytes > 0
    assert unrelated.is_file()


def test_dry_run_is_read_only_and_apply_archives_exact_candidates(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    archive = tmp_path / "archive"
    _seed_workspace(workspace)

    preview = cleanup_legacy_observability(workspace)
    assert preview.dry_run is True
    assert (workspace / "trace" / "trace.sqlite3").is_file()

    result = cleanup_legacy_observability(workspace, apply=True, archive_dir=archive)

    assert result.dry_run is False
    assert result.archived is True
    assert result.deleted_event_rows == 1
    assert result.deleted_event_row_bytes == preview.report.event_row_bytes
    assert set(result.removed_files) == {
        "trace/trace.sqlite3",
        "trace/trace.sqlite3-wal",
        "trace/trace.sqlite3-shm",
        "events/llm/llm_calls.jsonl",
    }
    for relative_path in result.removed_files:
        assert not (workspace / Path(relative_path)).exists()
        assert (archive / Path(relative_path)).is_file()
    archived_rows = (archive / "events" / "events.db.legacy-trace-rows.jsonl").read_text(
        encoding="utf-8"
    )
    assert json.loads(archived_rows)["event_row_id"] > 0
    assert (archive / "legacy_observability_cleanup_manifest.json").is_file()

    with sqlite3.connect(workspace / "events" / "events.db") as connection:
        remaining = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM events")]
    assert {row["id"] for row in remaining} == {"custom-trace", "canonical"}


def test_apply_without_archive_records_unarchived_removal(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _seed_workspace(workspace)

    result = cleanup_legacy_observability(workspace, apply=True)

    assert result.archived is False
    assert result.archive_dir is None
    assert result.archive_manifest is None
    assert result.deleted_event_rows == 1
    assert scan_legacy_observability(workspace).candidate_bytes == 0


def test_cleanup_rejects_archive_inside_workspace_before_deletion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _seed_workspace(workspace)

    with pytest.raises(LegacyObservabilityContainmentError, match="outside the workspace"):
        cleanup_legacy_observability(
            workspace,
            apply=True,
            archive_dir=workspace / "archive",
        )

    assert (workspace / "trace" / "trace.sqlite3").is_file()


def test_cleanup_refuses_workspace_held_by_runtime_lock(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with (
        workspace_lock(workspace),
        pytest.raises(LegacyObservabilityWorkspaceActiveError, match="Workspace is active"),
    ):
        cleanup_legacy_observability(workspace, apply=True)

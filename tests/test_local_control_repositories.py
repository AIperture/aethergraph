from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    PageRequest,
    RunQuery,
    RunRecord,
    RunResultRecord,
    RunStatus,
    SessionKind,
    SessionQuery,
    SessionRecord,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalRunRepository,
    LocalRunResultRepository,
    LocalSessionRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 20, tzinfo=UTC)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


def _run(
    run_id: str,
    *,
    started_at: datetime = NOW,
    project_id: str = "project-1",
    status: RunStatus = RunStatus.RUNNING,
) -> RunRecord:
    graph_id = "graph-1"
    return RunRecord(
        run_id=run_id,
        graph_id=graph_id,
        kind="taskgraph",
        status=status,
        scope=StorageScope(
            tenant_id="tenant-1",
            project_id=project_id,
            session_id="session-1",
            run_id=run_id,
            graph_id=graph_id,
        ),
        revision=1,
        started_at=started_at,
    )


def _session(
    session_id: str,
    *,
    updated_at: datetime = NOW,
    project_id: str = "project-1",
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        kind=SessionKind.CHAT,
        scope=StorageScope(
            tenant_id="tenant-1",
            project_id=project_id,
            user_id="user-1",
            session_id=session_id,
        ),
        revision=1,
        created_at=NOW,
        updated_at=updated_at,
        source="runtime",
    )


@pytest.mark.asyncio
async def test_run_create_cas_scope_and_recent_cursor_query(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runs = LocalRunRepository(database=database)
    records = tuple(
        _run(f"run-{index}", started_at=NOW + timedelta(seconds=index)) for index in range(3)
    )
    for record in records:
        assert await runs.create(record) == record
    assert await runs.create(records[0]) == records[0]
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await runs.create(replace(records[0], error="different"))
    assert await runs.get(StorageScope(project_id="other"), records[0].run_id) is None
    assert await runs.get(StorageScope(), records[0].run_id) is None

    query = RunQuery(
        scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        statuses=(RunStatus.RUNNING,),
        page=PageRequest(limit=2),
    )
    page_one = await runs.query(query)
    page_two = await runs.query(
        replace(query, page=PageRequest(limit=2, cursor=page_one.next_cursor))
    )
    assert (*page_one.items, *page_two.items) == tuple(reversed(records))
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await runs.query(
            replace(
                query,
                statuses=(RunStatus.FAILED,),
                page=PageRequest(limit=2, cursor=page_one.next_cursor),
            )
        )

    updated = replace(records[0], revision=2, status=RunStatus.WAITING)
    assert await runs.compare_and_set(updated, 1) == updated
    with pytest.raises(StorageConflictError):
        await runs.compare_and_set(updated, 1)
    with pytest.raises(StorageIntegrityError, match="immutable"):
        await runs.compare_and_set(replace(updated, revision=3, kind="other"), 2)
    await database.close()


@pytest.mark.asyncio
async def test_run_artifact_receipts_are_atomic_idempotent_and_concurrent(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runs = LocalRunRepository(database=database)
    initial = _run("run-1")
    await runs.create(initial)

    updates = await asyncio.gather(
        *(
            runs.record_artifact(
                initial.scope,
                initial.run_id,
                f"content-{index}",
                f"occurrence-{index}",
                NOW + timedelta(seconds=index),
            )
            for index in range(20)
        )
    )
    current = await runs.get(initial.scope, initial.run_id)
    assert current is not None
    assert current.artifact_count == 20
    assert current.revision == 21
    assert current.first_artifact_at == NOW
    assert current.last_artifact_at == NOW + timedelta(seconds=19)
    assert len(current.recent_artifact_ids) == 10
    assert set(current.recent_artifact_ids) <= {f"content-{index}" for index in range(20)}
    assert (
        await runs.record_artifact(
            initial.scope,
            initial.run_id,
            "content-0",
            "occurrence-0",
            NOW,
        )
        == current
    )
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await runs.record_artifact(
            initial.scope,
            initial.run_id,
            "different-content",
            "occurrence-0",
            NOW,
        )
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await runs.record_artifact(
            initial.scope,
            initial.run_id,
            "content-0",
            "occurrence-0",
            NOW + timedelta(days=1),
        )
    with pytest.raises(StorageNotFoundError):
        await runs.record_artifact(
            StorageScope(project_id="other", run_id=initial.run_id),
            initial.run_id,
            "foreign-content",
            "foreign",
            NOW,
        )
    assert len({record.revision for record in updates}) == 20
    await database.close()

    reopened = _database(tmp_path, StorageOpenMode.READ_WRITE)
    reopened_runs = LocalRunRepository(database=reopened)
    assert await reopened_runs.get(initial.scope, initial.run_id) == current
    assert (
        await reopened_runs.record_artifact(
            initial.scope,
            initial.run_id,
            "content-0",
            "occurrence-0",
            NOW,
        )
        == current
    )
    await reopened.close()


@pytest.mark.asyncio
async def test_result_and_run_availability_commit_together(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runs = LocalRunRepository(database=database)
    results = LocalRunResultRepository(database=database)
    running = _run("run-1")
    await runs.create(running)
    completed = replace(
        running,
        revision=2,
        status=RunStatus.SUCCEEDED,
        finished_at=NOW + timedelta(seconds=1),
    )
    await runs.compare_and_set(completed, 1)
    result = RunResultRecord(
        run_id=completed.run_id,
        graph_id=completed.graph_id,
        scope=completed.scope,
        status=RunStatus.SUCCEEDED,
        outputs={"answer": 42},
        revision=1,
        created_at=NOW + timedelta(seconds=2),
        updated_at=NOW + timedelta(seconds=2),
        source="runtime",
    )

    assert await results.compare_and_set(result, 0) == result
    marked = await runs.get(completed.scope, completed.run_id)
    assert marked is not None
    assert marked.revision == 3
    assert marked.result_available is True
    assert marked.result_updated_at == result.updated_at
    assert await results.get(StorageScope(project_id="project-1"), result.run_id) == result
    assert await results.get(StorageScope(project_id="other"), result.run_id) is None

    next_result = replace(
        result,
        revision=2,
        outputs={"answer": 43},
        updated_at=NOW + timedelta(seconds=3),
    )
    assert await results.compare_and_set(next_result, 1) == next_result
    remarked = await runs.get(completed.scope, completed.run_id)
    assert remarked is not None and remarked.revision == 4
    with pytest.raises(StorageConflictError):
        await results.compare_and_set(next_result, 1)
    await database.close()


@pytest.mark.asyncio
async def test_result_rejects_non_successful_or_cross_scope_run(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runs = LocalRunRepository(database=database)
    results = LocalRunResultRepository(database=database)
    running = _run("run-1")
    await runs.create(running)
    result = RunResultRecord(
        run_id=running.run_id,
        graph_id=running.graph_id,
        scope=running.scope,
        status=RunStatus.SUCCEEDED,
        outputs={},
        revision=1,
        created_at=NOW,
        updated_at=NOW,
        source="runtime",
    )
    with pytest.raises(StorageIntegrityError, match="successful"):
        await results.compare_and_set(result, 0)
    with pytest.raises(StorageNotFoundError):
        await results.compare_and_set(
            replace(result, scope=replace(result.scope, project_id="other")),
            0,
        )
    await database.close()


@pytest.mark.asyncio
async def test_result_delete_is_scoped_revisioned_atomic_and_restart_durable(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runs = LocalRunRepository(database=database)
    results = LocalRunResultRepository(database=database)
    running = _run("run-delete")
    await runs.create(running)
    completed = replace(
        running,
        revision=2,
        status=RunStatus.SUCCEEDED,
        finished_at=NOW + timedelta(seconds=1),
    )
    await runs.compare_and_set(completed, 1)
    result = RunResultRecord(
        run_id=completed.run_id,
        graph_id=completed.graph_id,
        scope=completed.scope,
        status=RunStatus.SUCCEEDED,
        outputs={"answer": 42},
        revision=1,
        created_at=NOW + timedelta(seconds=2),
        updated_at=NOW + timedelta(seconds=2),
        source="runtime",
    )
    await results.compare_and_set(result, 0)

    assert not await results.delete(StorageScope(project_id="other"), result.run_id, 1)
    with pytest.raises(StorageConflictError):
        await results.delete(result.scope, result.run_id, 2)
    assert await results.delete(result.scope, result.run_id, 1)
    assert not await results.delete(result.scope, result.run_id, 1)
    assert await results.get(result.scope, result.run_id) is None
    cleared = await runs.get(result.scope, result.run_id)
    assert cleared is not None
    assert cleared.revision == 4
    assert cleared.result_available is False
    assert cleared.result_updated_at is None

    recreated = replace(
        result,
        outputs={"answer": 43},
        updated_at=NOW + timedelta(seconds=3),
    )
    await results.compare_and_set(recreated, 0)
    await database.close()

    reopened = _database(tmp_path, StorageOpenMode.READ_WRITE)
    reopened_runs = LocalRunRepository(database=reopened)
    reopened_results = LocalRunResultRepository(database=reopened)
    assert await reopened_results.get(result.scope, result.run_id) == recreated
    remarked = await reopened_runs.get(result.scope, result.run_id)
    assert remarked is not None
    assert remarked.revision == 5
    assert remarked.result_available is True
    assert remarked.result_updated_at == recreated.updated_at
    await reopened.close()


@pytest.mark.asyncio
async def test_session_cas_query_and_artifact_receipts(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    sessions = LocalSessionRepository(database=database)
    records = tuple(
        _session(f"session-{index}", updated_at=NOW + timedelta(seconds=index))
        for index in range(3)
    )
    for record in records:
        await sessions.create(record)
    query = SessionQuery(
        scope=StorageScope(project_id="project-1"),
        kinds=(SessionKind.CHAT,),
        page=PageRequest(limit=2),
    )
    page_one = await sessions.query(query)
    page_two = await sessions.query(
        replace(query, page=PageRequest(limit=2, cursor=page_one.next_cursor))
    )
    assert (*page_one.items, *page_two.items) == tuple(reversed(records))
    renamed = replace(
        records[0],
        revision=2,
        updated_at=NOW + timedelta(minutes=1),
        title="Renamed",
    )
    assert await sessions.compare_and_set(renamed, 1) == renamed

    counted = await sessions.record_artifact(
        renamed.scope,
        renamed.session_id,
        "occurrence-1",
        NOW + timedelta(minutes=2),
    )
    assert counted.revision == 3
    assert counted.artifact_count == 1
    assert counted.updated_at == NOW + timedelta(minutes=2)
    assert (
        await sessions.record_artifact(
            renamed.scope,
            renamed.session_id,
            "occurrence-1",
            NOW + timedelta(minutes=2),
        )
        == counted
    )
    with pytest.raises(StorageIntegrityError, match="provider-owned"):
        await sessions.compare_and_set(replace(counted, revision=4, artifact_count=2), 3)
    cleared = replace(
        counted,
        revision=4,
        updated_at=NOW + timedelta(minutes=3),
        title="",
        external_reference="",
    )
    assert await sessions.compare_and_set(cleared, 3) == cleared
    with pytest.raises(StorageConflictError):
        await sessions.compare_and_set(cleared, 3)
    assert await sessions.get(StorageScope(project_id="other"), renamed.session_id) is None
    await database.close()

    reopened = _database(tmp_path, StorageOpenMode.READ_WRITE)
    reopened_sessions = LocalSessionRepository(database=reopened)
    assert await reopened_sessions.get(cleared.scope, cleared.session_id) == cleared
    await reopened.close()


@pytest.mark.asyncio
async def test_session_delete_is_scoped_revisioned_and_cascades_only_receipts(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runs = LocalRunRepository(database=database)
    sessions = LocalSessionRepository(database=database)
    session = _session("session-delete")
    await sessions.create(session)
    counted = await sessions.record_artifact(
        session.scope,
        session.session_id,
        "occurrence-1",
        NOW + timedelta(seconds=1),
    )
    related_run = _run("run-preserved")
    related_run = replace(
        related_run,
        scope=replace(related_run.scope, session_id=session.session_id),
    )
    await runs.create(related_run)

    assert not await sessions.delete(
        StorageScope(project_id="other"), counted.session_id, counted.revision
    )
    with pytest.raises(StorageConflictError):
        await sessions.delete(counted.scope, counted.session_id, 1)
    assert await sessions.delete(counted.scope, counted.session_id, counted.revision)
    assert not await sessions.delete(counted.scope, counted.session_id, counted.revision)
    assert await sessions.get(counted.scope, counted.session_id) is None
    receipts = await database.fetch_all(
        "SELECT occurrence_id FROM local_session_artifact_occurrences WHERE session_id = ?",
        (counted.session_id,),
    )
    assert receipts == ()
    assert await runs.get(related_run.scope, related_run.run_id) == related_run
    await database.close()

    reopened = _database(tmp_path, StorageOpenMode.READ_WRITE)
    reopened_runs = LocalRunRepository(database=reopened)
    reopened_sessions = LocalSessionRepository(database=reopened)
    assert await reopened_sessions.get(counted.scope, counted.session_id) is None
    assert await reopened_runs.get(related_run.scope, related_run.run_id) == related_run
    await reopened.close()


@pytest.mark.asyncio
async def test_control_schema_is_canonical_and_query_paths_are_indexed(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalRunRepository(database=database)
    LocalRunResultRepository(database=database)
    LocalSessionRepository(database=database)
    for table in ("local_runs", "local_run_results", "local_sessions"):
        columns = {
            str(row["name"]) for row in await database.fetch_all(f"PRAGMA table_info({table})")
        }
        assert not {"app_id", "application_id", "client_id", "path"} & columns
    run_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_runs
            WHERE tenant_id = ? AND project_id = ?
            ORDER BY started_at DESC, run_id DESC LIMIT ?
            """,
            ("tenant-1", "project-1", 20),
        )
    )
    session_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_sessions
            WHERE tenant_id = ? AND project_id = ?
            ORDER BY updated_at DESC, session_id DESC LIMIT ?
            """,
            ("tenant-1", "project-1", 20),
        )
    )
    assert "ix_local_runs_project_started" in run_plan
    assert "ix_local_sessions_project_updated" in session_plan
    assert "SCAN local_runs" not in run_plan
    assert "SCAN local_sessions" not in session_plan
    await database.close()


@pytest.mark.asyncio
async def test_read_only_control_repositories_read_and_reject_mutation(
    tmp_path: Path,
) -> None:
    running = _run("run-1")
    session = _session("session-1")
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable_runs = LocalRunRepository(database=writable_database)
    writable_sessions = LocalSessionRepository(database=writable_database)
    await writable_runs.create(running)
    await writable_sessions.create(session)
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly_runs = LocalRunRepository(database=readonly_database)
    readonly_results = LocalRunResultRepository(database=readonly_database)
    readonly_sessions = LocalSessionRepository(database=readonly_database)
    assert await readonly_runs.get(running.scope, running.run_id) == running
    assert await readonly_sessions.get(session.scope, session.session_id) == session
    assert await readonly_results.get(running.scope, running.run_id) is None
    with pytest.raises(StorageReadOnlyError):
        await readonly_runs.create(_run("new"))
    with pytest.raises(StorageReadOnlyError):
        await readonly_runs.record_artifact(
            running.scope,
            running.run_id,
            "content",
            "occ",
            NOW,
        )
    with pytest.raises(StorageReadOnlyError):
        await readonly_results.compare_and_set(
            RunResultRecord(
                run_id=running.run_id,
                graph_id=running.graph_id,
                scope=running.scope,
                status=RunStatus.SUCCEEDED,
                outputs={},
                revision=1,
                created_at=NOW,
                updated_at=NOW,
                source="runtime",
            ),
            0,
        )
    with pytest.raises(StorageReadOnlyError):
        await readonly_results.delete(running.scope, running.run_id, 1)
    with pytest.raises(StorageReadOnlyError):
        await readonly_sessions.create(_session("new"))
    with pytest.raises(StorageReadOnlyError):
        await readonly_sessions.delete(session.scope, session.session_id, session.revision)
    await readonly_database.close()


def test_local_control_repository_docstrings_follow_required_section_order() -> None:
    required = ("Examples:", "Args:", "Returns:", "Notes:")
    for repository in (
        LocalRunRepository,
        LocalRunResultRepository,
        LocalSessionRepository,
    ):
        for name, member in inspect.getmembers(repository, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (repository.__name__, name)
            assert positions == tuple(sorted(positions)), (repository.__name__, name)

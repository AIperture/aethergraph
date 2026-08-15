from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    InboundEventDraft,
    PageRequest,
    RuntimeOutputFrame,
    RuntimeOutputStream,
    SemanticEventDraft,
    SemanticEventKind,
    SemanticEventQuery,
    StorageCapacityError,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalInboundEventRepository,
    LocalRuntimeOutputSink,
    LocalSemanticEventRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 16, 3, tzinfo=UTC)
SESSION_SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    session_id="session-1",
)
RUN_SCOPE = replace(
    SESSION_SCOPE,
    run_id="run-1",
    graph_id="graph-1",
    node_id="node-1",
)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.EVENTS,
        mode=mode,
    )


def _inbound(
    event_id: str = "inbound-1",
    *,
    external_event_id: str = "external-1",
    received_at: datetime = NOW,
) -> InboundEventDraft:
    return InboundEventDraft(
        event_id=event_id,
        deployment_id="deployment-1",
        route_id="route-1",
        integration_id="slack-main",
        external_event_id=external_event_id,
        received_at=received_at,
        scope=SESSION_SCOPE,
        payload={"text": "hello", "nested": [1, 2]},
        resource_keys=("artifact:1",),
    )


def _semantic(
    event_id: str,
    sequence: int,
    *,
    kind: SemanticEventKind = SemanticEventKind.MESSAGE_DELTA,
    turn_id: str = "turn-1",
) -> SemanticEventDraft:
    return SemanticEventDraft(
        event_id=event_id,
        deployment_id="deployment-1",
        turn_id=turn_id,
        sequence=sequence,
        producer="agent.support",
        occurred_at=NOW + timedelta(seconds=sequence),
        kind=kind,
        scope=SESSION_SCOPE,
        payload={"text": event_id},
    )


def _frame(
    output_id: str,
    execution_id: str,
    sequence: int,
    *,
    scope: StorageScope = RUN_SCOPE,
    text: str | None = None,
) -> RuntimeOutputFrame:
    return RuntimeOutputFrame(
        output_id=output_id,
        execution_id=execution_id,
        scope=scope,
        stream=RuntimeOutputStream.STDOUT,
        sequence=sequence,
        text=text or output_id,
        source="python",
        tags=("runtime",),
    )


@pytest.mark.asyncio
async def test_inbound_append_is_ordered_idempotent_scoped_and_conflict_safe(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalInboundEventRepository(database=database)
    first = _inbound()
    second = _inbound(
        "inbound-2",
        external_event_id="external-2",
        received_at=NOW + timedelta(seconds=1),
    )
    stored_first = await repository.append(first)
    stored_second = await repository.append(second)
    assert stored_first.cursor != stored_second.cursor
    assert await repository.append(first) == stored_first
    assert await repository.get(SESSION_SCOPE, first.event_id) == stored_first
    assert await repository.get(StorageScope(project_id="other"), first.event_id) is None
    with pytest.raises(StorageIntegrityError, match="event identity"):
        await repository.append(replace(first, route_id="other"))
    with pytest.raises(StorageIntegrityError, match="external event"):
        await repository.append(_inbound("other", external_event_id=first.external_event_id))
    await database.close()


@pytest.mark.asyncio
async def test_concurrent_inbound_exact_retry_creates_one_row(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalInboundEventRepository(database=database)
    records = await asyncio.gather(*(repository.append(_inbound()) for _ in range(20)))
    assert all(record == records[0] for record in records)
    rows = await database.fetch_all("SELECT COUNT(*) FROM local_inbound_events")
    assert int(rows[0][0]) == 1
    await database.close()


@pytest.mark.asyncio
async def test_semantic_events_reject_identity_and_sequence_and_page_ascending(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalSemanticEventRepository(database=database)
    events = tuple(_semantic(f"semantic-{index}", index) for index in range(3))
    records = tuple([await repository.append(event) for event in events])
    with pytest.raises(StorageIntegrityError, match="identity"):
        await repository.append(events[0])
    with pytest.raises(StorageIntegrityError, match="sequence"):
        await repository.append(_semantic("different", 0))

    query = SemanticEventQuery(
        deployment_id="deployment-1",
        scope=SESSION_SCOPE,
        page=PageRequest(limit=2),
    )
    first = await repository.query(query)
    second = await repository.query(
        replace(query, page=PageRequest(limit=2, cursor=first.next_cursor))
    )
    assert (*first.items, *second.items) == records
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.query(
            replace(
                query,
                kinds=(SemanticEventKind.ERROR,),
                page=PageRequest(limit=2, cursor=first.next_cursor),
            )
        )
    filtered = await repository.query(replace(query, turn_id="turn-1"))
    assert len(filtered.items) == 2
    await database.close()


@pytest.mark.asyncio
async def test_runtime_output_capacity_selective_barriers_and_exact_retry(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    sink = LocalRuntimeOutputSink(database=database, max_pending_frames=2)
    first = _frame("output-1", "execution-1", 1)
    second = _frame("output-2", "execution-2", 1)
    sink.emit(first)
    sink.emit(first)
    sink.emit(second)
    with pytest.raises(StorageCapacityError, match="full"):
        sink.emit(_frame("output-3", "execution-3", 1))

    await sink.flush_execution("execution-1")
    rows = await database.fetch_all("SELECT output_id FROM local_runtime_output ORDER BY cursor")
    assert [str(row[0]) for row in rows] == ["output-1"]
    sink.emit(_frame("output-3", "execution-3", 1))
    await sink.flush_run("run-1")
    rows = await database.fetch_all("SELECT output_id FROM local_runtime_output ORDER BY cursor")
    assert [str(row[0]) for row in rows] == ["output-1", "output-2", "output-3"]

    sink.emit(first)
    await sink.flush_execution(first.execution_id)
    assert int((await database.fetch_all("SELECT COUNT(*) FROM local_runtime_output"))[0][0]) == 3
    await database.close()


@pytest.mark.asyncio
async def test_runtime_output_identity_failure_is_atomic_and_remains_pending(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    sink = LocalRuntimeOutputSink(database=database)
    first = _frame("output-1", "execution-1", 1)
    sink.emit(first)
    with pytest.raises(StorageIntegrityError, match="identity"):
        sink.emit(replace(first, text="different"))
    with pytest.raises(StorageIntegrityError, match="sequence"):
        sink.emit(_frame("output-2", "execution-1", 1))
    await sink.flush_execution("execution-1")

    conflicting = replace(first, text="persisted conflict")
    sink.emit(conflicting)
    with pytest.raises(StorageIntegrityError, match="identity"):
        await sink.flush_execution("execution-1")
    assert int((await database.fetch_all("SELECT COUNT(*) FROM local_runtime_output"))[0][0]) == 1
    await database.close()


@pytest.mark.asyncio
async def test_stream_repositories_read_only_and_typed_corruption(tmp_path: Path) -> None:
    writer_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    inbound_writer = LocalInboundEventRepository(database=writer_database)
    semantic_writer = LocalSemanticEventRepository(database=writer_database)
    await inbound_writer.append(_inbound())
    await semantic_writer.append(_semantic("semantic-1", 0))
    await writer_database.close()

    database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    inbound = LocalInboundEventRepository(database=database)
    semantic = LocalSemanticEventRepository(database=database)
    sink = LocalRuntimeOutputSink(database=database)
    assert await inbound.get(SESSION_SCOPE, "inbound-1") is not None
    assert (
        await semantic.query(SemanticEventQuery(deployment_id="deployment-1", scope=SESSION_SCOPE))
    ).items
    with pytest.raises(StorageReadOnlyError):
        await inbound.append(_inbound("new", external_event_id="new"))
    with pytest.raises(StorageReadOnlyError):
        await semantic.append(_semantic("new", 1))
    with pytest.raises(StorageReadOnlyError):
        sink.emit(_frame("new", "execution", 1))
    await database.close()

    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    inbound = LocalInboundEventRepository(database=database)
    semantic = LocalSemanticEventRepository(database=database)
    await database.transaction(
        lambda connection: connection.execute("UPDATE local_inbound_events SET payload_json = '[]'")
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await inbound.get(SESSION_SCOPE, "inbound-1")
    await database.transaction(
        lambda connection: connection.execute("UPDATE local_semantic_events SET kind = 'invalid'")
    )
    with pytest.raises(StorageIntegrityError, match="malformed"):
        await semantic.query(SemanticEventQuery(deployment_id="deployment-1", scope=SESSION_SCOPE))
    await database.close()


@pytest.mark.asyncio
async def test_stream_schema_is_clean_and_query_shapes_are_indexed(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalInboundEventRepository(database=database)
    LocalSemanticEventRepository(database=database)
    LocalRuntimeOutputSink(database=database)
    forbidden = {"app_id", "application_id", "client_id", "path"}
    for table in (
        "local_inbound_events",
        "local_semantic_events",
        "local_runtime_output",
    ):
        columns = await database.fetch_all(f"PRAGMA table_info({table})")
        assert forbidden.isdisjoint({str(column["name"]) for column in columns})
    plans = (
        await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_semantic_events "
            "WHERE deployment_id = ? AND session_id = ? ORDER BY cursor LIMIT 10",
            ("deployment-1", "session-1"),
        ),
        await database.fetch_all(
            "EXPLAIN QUERY PLAN SELECT * FROM local_runtime_output "
            "WHERE execution_id = ? ORDER BY execution_sequence",
            ("execution-1",),
        ),
    )
    details = tuple(" ".join(str(row[3]) for row in plan) for plan in plans)
    assert "ix_local_semantic_session_cursor" in details[0]
    assert "ix_local_runtime_output_execution" in details[1]
    await database.close()


def test_local_stream_public_docstrings_follow_repository_format() -> None:
    for repository in (
        LocalInboundEventRepository,
        LocalSemanticEventRepository,
        LocalRuntimeOutputSink,
    ):
        for name, method in inspect.getmembers(repository, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(method) or ""
            assert docstring.index("Examples:") < docstring.index("Args:")
            assert docstring.index("Args:") < docstring.index("Returns:")
            assert docstring.index("Returns:") < docstring.index("Notes:")
            assert docstring.count("```python") >= 2

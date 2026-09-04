from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from storage_conformance.suite import check_event_store_conformance, event

from aethergraph.storage.contracts import (
    EventDraft,
    EventQuery,
    PageRequest,
    SortDirection,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalEventStore,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 18, tzinfo=UTC)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.EVENTS,
        mode=mode,
    )


def _event(identifier: str, scope: StorageScope, **changes) -> EventDraft:
    values = {
        "event_id": identifier,
        "occurred_at": NOW,
        "scope": scope,
        "kind": "runtime.event",
        "stage": "execute",
        "topic": "runtime",
        "text": identifier,
        "tags": ("runtime", "important"),
        "payload": {"identifier": identifier, "nested": [1, True]},
        "metrics": {"latency_ms": 2.5},
        "severity": 20,
        "signal": 0.75,
    }
    values.update(changes)
    return EventDraft(**values)


@pytest.mark.asyncio
async def test_local_event_store_passes_shared_provider_conformance(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalEventStore(database=database, stream="runtime")

    await check_event_store_conformance(store)

    await database.close()


@pytest.mark.asyncio
async def test_logical_streams_are_isolated_on_one_events_database(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    runtime = LocalEventStore(database=database, stream="runtime")
    memory = LocalEventStore(database=database, stream="memory")
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    draft = _event("shared-id", scope)

    runtime_record = await runtime.append(draft)
    memory_record = await memory.append(draft)

    assert runtime_record.event_id == memory_record.event_id
    assert runtime_record.cursor != memory_record.cursor
    assert await runtime.get(scope, draft.event_id) == runtime_record
    assert await memory.get(scope, draft.event_id) == memory_record
    assert await runtime.get_many(scope, ("missing", draft.event_id)) == (runtime_record,)
    assert await memory.get_many(scope, (draft.event_id, "missing")) == (memory_record,)
    await database.close()


@pytest.mark.asyncio
async def test_event_append_is_idempotent_and_batch_conflicts_roll_back(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalEventStore(database=database, stream="runtime")
    scope = StorageScope(project_id="project-1")
    first = _event("event-1", scope)

    committed = await store.append(first)
    assert await store.append(first) == committed
    assert await store.get_many(scope, ("missing", first.event_id)) == (committed,)
    assert await store.get_many(scope, ()) == ()
    with pytest.raises(ValueError, match="duplicates"):
        await store.get_many(scope, (first.event_id, first.event_id))
    retries = await asyncio.gather(*(store.append(first) for _ in range(20)))
    assert {record.cursor for record in retries} == {committed.cursor}
    with pytest.raises(StorageIntegrityError, match="conflicting content"):
        await store.append(replace(first, text="different"))

    with pytest.raises(StorageIntegrityError):
        await store.append_many(
            (
                _event("event-2", scope),
                replace(first, text="conflict in batch"),
            )
        )
    assert await store.get(scope, "event-2") is None
    with pytest.raises(StorageConfigurationError, match="exceeds"):
        await store.append_many(tuple(first for _ in range(1001)))

    concurrent = await asyncio.gather(
        *(store.append(_event(f"concurrent-{index}", scope)) for index in range(40))
    )
    numeric_cursors = [int(record.cursor.removeprefix("event:")) for record in concurrent]
    assert len(set(numeric_cursors)) == 40
    ordered = await store.query(
        EventQuery(scope=scope, order=SortDirection.ASCENDING, page=PageRequest(limit=1000))
    )
    ordered_concurrent = [
        int(record.cursor.removeprefix("event:"))
        for record in ordered.items
        if record.event_id.startswith("concurrent-")
    ]
    assert ordered_concurrent == sorted(numeric_cursors)
    await database.close()


@pytest.mark.asyncio
async def test_event_filters_order_and_cursor_context_are_exact(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalEventStore(database=database, stream="runtime")
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    drafts = tuple(
        _event(
            f"event-{index}",
            scope,
            occurred_at=NOW + timedelta(seconds=index),
            kind="runtime.error" if index == 2 else "runtime.event",
            topic="errors" if index == 2 else "runtime",
            tags=("runtime", "important") if index != 1 else ("runtime",),
        )
        for index in range(4)
    )
    await store.append_many(drafts)

    query = EventQuery(
        scope=scope,
        kinds=("runtime.event",),
        stage="execute",
        topic="runtime",
        tags=("important",),
        occurred_at_min=NOW,
        occurred_at_max=NOW + timedelta(seconds=3),
        order=SortDirection.ASCENDING,
        page=PageRequest(limit=1),
    )
    first = await store.query(query)
    second = await store.query(replace(query, page=PageRequest(limit=1, cursor=first.next_cursor)))
    assert [row.event_id for row in (*first.items, *second.items)] == ["event-0", "event-3"]

    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await store.query(
            replace(
                query,
                topic="other",
                page=PageRequest(limit=1, cursor=first.next_cursor),
            )
        )
    await database.close()


@pytest.mark.asyncio
async def test_read_only_event_store_reads_and_rejects_appends(tmp_path: Path) -> None:
    scope = StorageScope(project_id="project-1")
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable = LocalEventStore(database=writable_database, stream="runtime")
    stored = await writable.append(event("event-1", scope))
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly = LocalEventStore(database=readonly_database, stream="runtime")
    assert await readonly.get(scope, "event-1") == stored
    with pytest.raises(StorageReadOnlyError):
        await readonly.append(event("event-2", scope))
    await readonly_database.close()

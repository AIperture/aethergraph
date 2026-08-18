from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from storage_conformance.suite import check_state_store_conformance

from aethergraph.storage.contracts import (
    PageRequest,
    SortDirection,
    StateHistoryQuery,
    StorageConfigurationError,
    StorageConflictError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalSQLiteDatabase,
    LocalStateStore,
)


class _Clock:
    def __init__(self) -> None:
        self._next = datetime(2026, 8, 15, 20, tzinfo=UTC)

    def now(self) -> datetime:
        value = self._next
        self._next += timedelta(microseconds=1)
        return value


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


@pytest.mark.asyncio
async def test_local_state_store_passes_shared_provider_conformance(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalStateStore(database=database, clock=_Clock())

    await check_state_store_conformance(store)

    await database.close()


@pytest.mark.asyncio
async def test_concurrent_create_has_exactly_one_cas_winner(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalStateStore(database=database, clock=_Clock())
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1", run_id="run-1")

    attempts = await asyncio.gather(
        *(
            store.compare_and_set(scope, "agent", "writer", 0, {"winner": index}, {})
            for index in range(8)
        ),
        return_exceptions=True,
    )

    winners = [result for result in attempts if not isinstance(result, BaseException)]
    conflicts = [result for result in attempts if isinstance(result, StorageConflictError)]
    assert len(winners) == 1
    assert len(conflicts) == 7
    assert (await store.get(scope, "agent", "writer")) == winners[0]
    history = await store.history(StateHistoryQuery(scope=scope, namespace="agent", key="writer"))
    assert history.items == (winners[0],)
    outbox = await database.fetch_all("SELECT operation, revision FROM local_state_outbox")
    assert [tuple(row) for row in outbox] == [("updated", 1)]
    await database.close()


@pytest.mark.asyncio
async def test_history_pagination_is_stable_across_delete_and_recreate(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalStateStore(database=database, clock=_Clock())
    scope = StorageScope(project_id="project-1")
    first = await store.compare_and_set(scope, "graph", "node", 0, {"value": 1}, {})
    second = await store.compare_and_set(scope, "graph", "node", 1, {"value": 2}, {})
    assert await store.delete(scope, "graph", "node", 2) is True
    recreated = await store.compare_and_set(scope, "graph", "node", 0, {"value": 3}, {})

    query = StateHistoryQuery(
        scope=scope,
        namespace="graph",
        key="node",
        order=SortDirection.ASCENDING,
        page=PageRequest(limit=2),
    )
    page_one = await store.history(query)
    page_two = await store.history(
        replace(query, page=PageRequest(limit=2, cursor=page_one.next_cursor))
    )
    assert (*page_one.items, *page_two.items) == (first, second, recreated)
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await store.history(
            replace(
                query,
                key="other",
                page=PageRequest(limit=2, cursor=page_one.next_cursor),
            )
        )
    outbox = await database.fetch_all(
        "SELECT operation, revision FROM local_state_outbox ORDER BY outbox_id"
    )
    assert [tuple(row) for row in outbox] == [
        ("updated", 1),
        ("updated", 2),
        ("deleted", 2),
        ("updated", 1),
    ]
    await database.close()


@pytest.mark.asyncio
async def test_get_many_preserves_duplicates_missing_slots_and_bound(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalStateStore(database=database, clock=_Clock())
    scope = StorageScope(project_id="project-1")
    stored = await store.compare_and_set(scope, "graph", "a", 0, {"value": 1}, {})

    assert await store.get_many(scope, "graph", ("a", "missing", "a")) == (
        stored,
        None,
        stored,
    )
    with pytest.raises(StorageConfigurationError, match="exceeds"):
        await store.get_many(scope, "graph", tuple(str(index) for index in range(1001)))
    await database.close()


@pytest.mark.asyncio
async def test_read_only_state_store_reads_and_rejects_mutation(tmp_path: Path) -> None:
    scope = StorageScope(project_id="project-1")
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable = LocalStateStore(database=writable_database, clock=_Clock())
    stored = await writable.compare_and_set(scope, "agent", "writer", 0, {}, {})
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly = LocalStateStore(database=readonly_database, clock=_Clock())
    assert await readonly.get(scope, "agent", "writer") == stored
    with pytest.raises(StorageReadOnlyError):
        await readonly.compare_and_set(scope, "agent", "writer", 1, {}, {})
    with pytest.raises(StorageReadOnlyError):
        await readonly.delete(scope, "agent", "writer", 1)
    await readonly_database.close()

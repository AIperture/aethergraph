from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    DocumentQuery,
    KeyValueQuery,
    PageRequest,
    StorageConfigurationError,
    StorageConflictError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalDocumentStore,
    LocalKeyValueStore,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 20, tzinfo=UTC)


class _Clock:
    def __init__(self) -> None:
        self.value = NOW

    def now(self) -> datetime:
        return self.value


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


@pytest.mark.asyncio
async def test_key_value_cas_ttl_and_stable_prefix_pagination(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    clock = _Clock()
    store = LocalKeyValueStore(database=database, clock=clock)
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    expiring = await store.compare_and_set(
        scope,
        "auth.grants",
        "grant-a",
        0,
        {"role": "reader"},
        NOW + timedelta(seconds=1),
    )
    second = await store.compare_and_set(
        scope,
        "auth.grants",
        "grant-b",
        0,
        {"role": "writer"},
    )
    third = await store.compare_and_set(
        scope,
        "auth.grants",
        "other",
        0,
        {"role": "owner"},
    )
    assert await store.get(scope, "auth.grants", "grant-a") == expiring
    with pytest.raises(StorageConflictError):
        await store.compare_and_set(scope, "auth.grants", "grant-a", 0, {})

    query = KeyValueQuery(
        scope=scope,
        namespace="auth.grants",
        key_prefix="grant-",
        page=PageRequest(limit=1),
    )
    page_one = await store.scan(query)
    page_two = await store.scan(
        replace(query, page=PageRequest(limit=1, cursor=page_one.next_cursor))
    )
    assert (*page_one.items, *page_two.items) == (expiring, second)
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await store.scan(
            replace(
                query,
                key_prefix="other",
                page=PageRequest(limit=1, cursor=page_one.next_cursor),
            )
        )

    clock.value = NOW + timedelta(seconds=2)
    assert await store.get(scope, "auth.grants", "grant-a") is None
    assert [row.key for row in (await store.scan(query)).items] == ["grant-b"]
    recreated = await store.compare_and_set(
        scope,
        "auth.grants",
        "grant-a",
        0,
        {"role": "renewed"},
    )
    assert recreated.revision == 1
    assert await store.delete(scope, "auth.grants", "grant-a", 1) is True
    assert await store.delete(scope, "auth.grants", "grant-a", 0) is False
    assert await store.get(scope, "auth.grants", "other") == third
    await database.close()


@pytest.mark.asyncio
async def test_key_value_purges_expired_rows_in_bounded_exact_scope_batches(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    clock = _Clock()
    store = LocalKeyValueStore(database=database, clock=clock)
    scope = StorageScope(project_id="project-1")
    other_scope = StorageScope(project_id="project-2")
    expiry = NOW + timedelta(seconds=1)
    for key in ("a", "b"):
        await store.compare_and_set(scope, "runtime.kv", key, 0, {"key": key}, expiry)
    await store.compare_and_set(scope, "other", "c", 0, {}, expiry)
    await store.compare_and_set(other_scope, "runtime.kv", "d", 0, {}, expiry)
    await store.compare_and_set(scope, "runtime.kv", "live", 0, {})
    clock.value = NOW + timedelta(seconds=2)

    assert await store.purge_expired(scope, "runtime.kv", 1) == 1
    assert await store.purge_expired(scope, "runtime.kv", 1) == 1
    assert await store.purge_expired(scope, "runtime.kv", 1) == 0
    remaining = await database.fetch_all(
        "SELECT scope_identity, namespace, key FROM local_key_values ORDER BY key"
    )

    assert {str(row["key"]) for row in remaining} == {"c", "d", "live"}
    with pytest.raises(ValueError, match="between 1 and 1000"):
        await store.purge_expired(scope, "runtime.kv", 0)
    await database.close()


@pytest.mark.asyncio
async def test_key_value_concurrent_create_has_one_winner(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalKeyValueStore(database=database, clock=_Clock())
    scope = StorageScope(project_id="project-1")

    attempts = await asyncio.gather(
        *(
            store.compare_and_set(scope, "runtime", "key", 0, {"writer": index})
            for index in range(8)
        ),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in attempts) == 1
    assert sum(isinstance(result, StorageConflictError) for result in attempts) == 7
    await database.close()


@pytest.mark.asyncio
async def test_document_cas_metadata_filter_pagination_and_delete(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalDocumentStore(database=database, clock=_Clock())
    scope = StorageScope(project_id="project-1")
    other_scope = StorageScope(project_id="project-2")
    first = await store.compare_and_set(
        scope,
        "registry",
        "agent-a",
        0,
        {"kind": "agent", "enabled": True},
        1,
    )
    second = await store.compare_and_set(
        scope,
        "registry",
        "agent-b",
        0,
        {"kind": "agent", "enabled": False},
        1,
    )
    await store.compare_and_set(
        scope,
        "registry",
        "graph-a",
        0,
        {"kind": "graph"},
        1,
    )
    await store.compare_and_set(
        other_scope,
        "registry",
        "agent-hidden",
        0,
        {"kind": "agent"},
        1,
    )
    query = DocumentQuery(
        scope=scope,
        namespace="registry",
        id_prefix="agent-",
        metadata={"kind": "agent"},
        page=PageRequest(limit=1),
    )
    page_one = await store.query(query)
    page_two = await store.query(
        replace(query, page=PageRequest(limit=1, cursor=page_one.next_cursor))
    )
    assert (*page_one.items, *page_two.items) == (first, second)
    assert await store.get(other_scope, "registry", "agent-a") is None

    updated = await store.compare_and_set(
        scope,
        "registry",
        "agent-a",
        1,
        {"kind": "tool", "enabled": True},
        2,
    )
    assert updated.revision == 2 and updated.schema_version == 2
    assert [row.document_id for row in (await store.query(query)).items] == ["agent-b"]
    with pytest.raises(StorageConflictError):
        await store.delete(scope, "registry", "agent-a", 1)
    assert await store.delete(scope, "registry", "agent-a", 2) is True
    assert await store.delete(scope, "registry", "agent-a", 0) is False
    await database.close()


@pytest.mark.asyncio
async def test_supporting_schema_uses_canonical_identity_and_indexed_queries(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalKeyValueStore(database=database, clock=_Clock())
    LocalDocumentStore(database=database, clock=_Clock())
    for table in ("local_key_values", "local_documents", "local_document_metadata"):
        columns = {
            str(row["name"]) for row in await database.fetch_all(f"PRAGMA table_info({table})")
        }
        assert not {"app_id", "application_id", "client_id", "path"} & columns
    kv_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_key_values
            WHERE scope_identity = ? AND namespace = ? AND key >= ? AND key < ?
            ORDER BY key LIMIT ?
            """,
            ('{"project_id":"project-1"}', "auth.grants", "a", "b", 10),
        )
    )
    document_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT * FROM local_document_metadata
            WHERE scope_identity = ? AND namespace = ? AND key = ? AND value_json = ?
            """,
            ('{"project_id":"project-1"}', "registry", "kind", '"agent"'),
        )
    )
    expiry_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN SELECT key FROM local_key_values
            INDEXED BY ix_local_key_values_expiry
            WHERE scope_identity = ? AND namespace = ?
              AND expires_at IS NOT NULL AND expires_at <= ?
            ORDER BY expires_at, key LIMIT ?
            """,
            ('{"project_id":"project-1"}', "runtime.kv", NOW.isoformat(), 10),
        )
    )
    assert "sqlite_autoindex_local_key_values_1" in kv_plan
    assert "ix_local_key_values_expiry" in expiry_plan
    assert "ix_local_document_metadata_value" in document_plan
    assert "SCAN local_key_values" not in kv_plan
    assert "SCAN local_key_values" not in expiry_plan
    assert "SCAN local_document_metadata" not in document_plan
    await database.close()


@pytest.mark.asyncio
async def test_read_only_supporting_stores_read_and_reject_mutation(tmp_path: Path) -> None:
    clock = _Clock()
    scope = StorageScope(project_id="project-1")
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable_kv = LocalKeyValueStore(database=writable_database, clock=clock)
    writable_documents = LocalDocumentStore(database=writable_database, clock=clock)
    kv = await writable_kv.compare_and_set(scope, "runtime", "key", 0, {"value": 1})
    document = await writable_documents.compare_and_set(
        scope, "registry", "agent", 0, {"kind": "agent"}, 1
    )
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly_kv = LocalKeyValueStore(database=readonly_database, clock=clock)
    readonly_documents = LocalDocumentStore(database=readonly_database, clock=clock)
    assert await readonly_kv.get(scope, "runtime", "key") == kv
    assert await readonly_documents.get(scope, "registry", "agent") == document
    with pytest.raises(StorageReadOnlyError):
        await readonly_kv.compare_and_set(scope, "runtime", "other", 0, {})
    with pytest.raises(StorageReadOnlyError):
        await readonly_kv.delete(scope, "runtime", "key", 1)
    with pytest.raises(StorageReadOnlyError):
        await readonly_kv.purge_expired(scope, "runtime", 10)
    with pytest.raises(StorageReadOnlyError):
        await readonly_documents.compare_and_set(scope, "registry", "other", 0, {}, 1)
    with pytest.raises(StorageReadOnlyError):
        await readonly_documents.delete(scope, "registry", "agent", 1)
    await readonly_database.close()

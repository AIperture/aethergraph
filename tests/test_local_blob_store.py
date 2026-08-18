from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path

import pytest
from storage_conformance.suite import check_blob_store_conformance

from aethergraph.storage.contracts import (
    ArtifactRecord,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalArtifactRepository,
    LocalBlobStore,
    LocalDatabaseRole,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 20, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW


class _MutableClock:
    def __init__(self) -> None:
        self.value = NOW

    def now(self) -> datetime:
        return self.value


def _store(
    database: LocalSQLiteDatabase,
    root: Path,
    *,
    clock: _Clock | _MutableClock | None = None,
) -> LocalBlobStore:
    return LocalBlobStore(database=database, workspace_root=root, clock=clock or _Clock())


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


async def _chunks(content: bytes):
    yield content[:3]
    yield content[3:]


@pytest.mark.asyncio
async def test_local_blob_store_passes_shared_provider_conformance(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = _store(database, tmp_path)
    LocalArtifactRepository(database=database)

    await check_blob_store_conformance(store)

    assert list((tmp_path / "local" / "staging").iterdir()) == []
    await database.close()


@pytest.mark.asyncio
async def test_content_deduplicates_physically_without_cross_scope_access(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = _store(database, tmp_path)
    LocalArtifactRepository(database=database)
    first_scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    second_scope = StorageScope(tenant_id="tenant-1", project_id="project-2")
    content = b"shared immutable bytes"

    first = await store.put(first_scope, _chunks(content))
    second = await store.put(second_scope, _chunks(content))

    assert first == second
    assert await store.head(StorageScope(tenant_id="other"), first.blob_locator) is None
    content_files = [path for path in (tmp_path / "local" / "blobs").rglob("*") if path.is_file()]
    assert len(content_files) == 1

    with pytest.raises(StorageConflictError):
        await store.delete(first_scope, first.blob_locator, provider_version="wrong")
    assert await store.delete(first_scope, first.blob_locator) is True
    assert content_files[0].exists()
    assert await store.delete(second_scope, second.blob_locator) is True
    assert not content_files[0].exists()
    await database.close()


@pytest.mark.asyncio
async def test_integrity_failures_remove_staging_and_publish_nothing(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = _store(database, tmp_path)
    LocalArtifactRepository(database=database)
    scope = StorageScope(project_id="project-1")
    content = b"expected content"

    with pytest.raises(StorageIntegrityError, match="expected_hash"):
        await store.put(scope, _chunks(content), expected_hash="0" * 64)
    locator = f"blob:sha256:{hashlib.sha256(content).hexdigest()}"
    assert await store.head(scope, locator) is None
    assert list((tmp_path / "local" / "staging").iterdir()) == []

    async def invalid_chunks():
        yield "not-bytes"

    with pytest.raises(StorageIntegrityError, match="must be bytes"):
        await store.put(scope, invalid_chunks())  # type: ignore[arg-type]
    assert list((tmp_path / "local" / "staging").iterdir()) == []

    stored = await store.put(scope, _chunks(content))
    content_file = next(
        path for path in (tmp_path / "local" / "blobs").rglob("*") if path.is_file()
    )
    content_file.write_bytes(b"x" * len(content))
    with pytest.raises(StorageIntegrityError, match="content is inconsistent"):
        await store.put(scope, _chunks(content), expected_hash=stored.content_hash)
    await database.close()


@pytest.mark.asyncio
async def test_read_only_store_reads_but_rejects_put_and_delete(tmp_path: Path) -> None:
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable = _store(writable_database, tmp_path)
    LocalArtifactRepository(database=writable_database)
    scope = StorageScope(project_id="project-1")
    stored = await writable.put(scope, _chunks(b"read-only content"))
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly = _store(readonly_database, tmp_path)
    assert b"".join([chunk async for chunk in readonly.read(scope, stored.blob_locator)]) == (
        b"read-only content"
    )
    with pytest.raises(StorageReadOnlyError):
        await readonly.put(scope, _chunks(b"write"))
    with pytest.raises(StorageReadOnlyError):
        await readonly.delete(scope, stored.blob_locator)
    with pytest.raises(StorageReadOnlyError):
        await readonly.reconcile_artifact_orphans(scope, older_than=NOW + timedelta(hours=1))
    await readonly_database.close()


@pytest.mark.asyncio
async def test_orphan_reconciliation_is_bounded_touched_and_reference_safe(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    clock = _MutableClock()
    store = _store(database, tmp_path, clock=clock)
    artifacts = LocalArtifactRepository(database=database)
    owner = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_owner = StorageScope(tenant_id="tenant-1", project_id="project-2")

    referenced = await store.put(owner, _chunks(b"referenced"))
    preview = await store.put(owner, _chunks(b"preview"))
    await artifacts.put(
        ArtifactRecord(
            artifact_id="referenced",
            content_hash=referenced.content_hash,
            hash_algorithm=referenced.hash_algorithm,
            size_bytes=referenced.size_bytes,
            media_type="text/plain",
            kind="result",
            blob_locator=referenced.blob_locator,
            preview_locator=preview.blob_locator,
            owner_scope=owner,
            created_at=NOW,
            provider_version=referenced.provider_version,
        )
    )
    orphan = await store.put(owner, _chunks(b"orphan"))
    shared = await store.put(owner, _chunks(b"shared"))
    await store.put(other_owner, _chunks(b"shared"))
    touched = await store.put(owner, _chunks(b"touched"))

    clock.value = NOW + timedelta(hours=2)
    assert await store.put(owner, _chunks(b"touched")) == touched
    recent = await store.put(owner, _chunks(b"recent"))
    result = await store.reconcile_artifact_orphans(
        owner,
        older_than=NOW + timedelta(hours=1),
        limit=10,
    )

    assert result.examined == 2
    assert result.deleted_scoped_blobs == 2
    assert result.deleted_physical_blobs == 1
    assert result.freed_bytes == len(b"orphan")
    assert result.has_more is False
    assert await store.head(owner, orphan.blob_locator) is None
    assert await store.head(owner, shared.blob_locator) is None
    assert await store.head(other_owner, shared.blob_locator) is not None
    assert await store.head(owner, referenced.blob_locator) is not None
    assert await store.head(owner, preview.blob_locator) is not None
    assert await store.head(owner, touched.blob_locator) is not None
    assert await store.head(owner, recent.blob_locator) is not None

    bounded_owner = StorageScope(tenant_id="tenant-1", project_id="project-bounded")
    for index in range(3):
        await store.put(bounded_owner, _chunks(f"bounded-{index}".encode()))
    clock.value = NOW + timedelta(hours=4)
    first = await store.reconcile_artifact_orphans(
        bounded_owner,
        older_than=NOW + timedelta(hours=3),
        limit=2,
    )
    second = await store.reconcile_artifact_orphans(
        bounded_owner,
        older_than=NOW + timedelta(hours=3),
        limit=2,
    )
    assert first.deleted_scoped_blobs == 2 and first.has_more is True
    assert second.deleted_scoped_blobs == 1 and second.has_more is False
    assert await store.head(owner, referenced.blob_locator) is not None

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        await store.reconcile_artifact_orphans(owner, older_than=datetime.now())
    with pytest.raises(ValueError, match="between 1 and 500"):
        await store.reconcile_artifact_orphans(owner, older_than=clock.value, limit=0)
    blob_columns = {
        str(row["name"]) for row in await database.fetch_all("PRAGMA table_info(local_blobs)")
    }
    tombstone_columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_blob_gc_tombstones)")
    }
    assert {"created_at", "last_touched_at"} <= blob_columns
    assert tombstone_columns == {"content_hash", "size_bytes", "queued_at"}
    plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT b.blob_locator FROM local_blobs b
            WHERE b.scope_key = ? AND b.last_touched_at < ?
              AND NOT EXISTS (
                  SELECT 1 FROM local_artifacts a
                  WHERE a.owner_scope_identity = b.scope_key
                    AND (
                        a.blob_locator = b.blob_locator
                        OR a.preview_locator = b.blob_locator
                    )
              )
            ORDER BY b.last_touched_at, b.blob_locator LIMIT ?
            """,
            (
                '{"project_id":"project-bounded","tenant_id":"tenant-1"}',
                clock.value.isoformat(),
                10,
            ),
        )
    )
    assert "ix_local_blobs_orphan" in plan
    assert "SCAN b" not in plan
    await database.close()


@pytest.mark.asyncio
async def test_orphan_cleanup_and_cross_instance_publish_remain_coherent(
    tmp_path: Path,
) -> None:
    first_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    first_clock = _MutableClock()
    first = _store(first_database, tmp_path, clock=first_clock)
    LocalArtifactRepository(database=first_database)
    second_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    second_clock = _MutableClock()
    second = _store(second_database, tmp_path, clock=second_clock)
    LocalArtifactRepository(database=second_database)

    try:
        for index in range(12):
            content = f"concurrent-{index}".encode()
            old_scope = StorageScope(project_id=f"old-{index}")
            new_scope = StorageScope(project_id=f"new-{index}")
            first_clock.value = NOW
            old = await first.put(old_scope, _chunks(content))
            first_clock.value = NOW + timedelta(hours=2)
            second_clock.value = NOW + timedelta(hours=2)

            cleanup, published = await asyncio.gather(
                first.reconcile_artifact_orphans(
                    old_scope,
                    older_than=NOW + timedelta(hours=1),
                    limit=1,
                ),
                second.put(new_scope, _chunks(content)),
            )

            assert cleanup.deleted_scoped_blobs == 1
            assert published.blob_locator == old.blob_locator
            assert await first.head(old_scope, old.blob_locator) is None
            assert await second.head(new_scope, published.blob_locator) is not None
            assert (
                b"".join([chunk async for chunk in second.read(new_scope, published.blob_locator)])
                == content
            )
    finally:
        await second_database.close()
        await first_database.close()

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from storage_conformance.suite import check_blob_store_conformance

from aethergraph.storage.contracts import (
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalBlobStore,
    LocalDatabaseRole,
    LocalSQLiteDatabase,
)


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
    store = LocalBlobStore(database=database, workspace_root=tmp_path)

    await check_blob_store_conformance(store)

    assert list((tmp_path / "local" / "staging").iterdir()) == []
    await database.close()


@pytest.mark.asyncio
async def test_content_deduplicates_physically_without_cross_scope_access(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    store = LocalBlobStore(database=database, workspace_root=tmp_path)
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
    store = LocalBlobStore(database=database, workspace_root=tmp_path)
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
    writable = LocalBlobStore(database=writable_database, workspace_root=tmp_path)
    scope = StorageScope(project_id="project-1")
    stored = await writable.put(scope, _chunks(b"read-only content"))
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly = LocalBlobStore(database=readonly_database, workspace_root=tmp_path)
    assert b"".join([chunk async for chunk in readonly.read(scope, stored.blob_locator)]) == (
        b"read-only content"
    )
    with pytest.raises(StorageReadOnlyError):
        await readonly.put(scope, _chunks(b"write"))
    with pytest.raises(StorageReadOnlyError):
        await readonly.delete(scope, stored.blob_locator)
    await readonly_database.close()

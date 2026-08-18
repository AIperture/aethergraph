from __future__ import annotations

import asyncio
from pathlib import Path
import sqlite3

import pytest

from aethergraph.storage.contracts import (
    StorageFormatError,
    StorageHealthError,
    StorageOpenMode,
    StorageReadOnlyError,
)
from aethergraph.storage.providers.local_sqlite import (
    LOCAL_DATABASE_SCHEMA_VERSION,
    LocalDatabaseRole,
    LocalSQLiteDatabase,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("role", list(LocalDatabaseRole))
async def test_each_role_uses_central_policy_and_exact_schema_metadata(
    tmp_path: Path,
    role: LocalDatabaseRole,
) -> None:
    database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=role,
        mode=StorageOpenMode.READ_WRITE,
        busy_timeout_ms=3210,
        durability="full",
    )

    metadata = await database.fetch_all(
        "SELECT role, schema_version FROM ag_storage_meta WHERE singleton = 1"
    )
    pragmas = await database.fetch_all(
        "SELECT (SELECT * FROM pragma_journal_mode), (SELECT * FROM pragma_foreign_keys), "
        "(SELECT * FROM pragma_busy_timeout)"
    )

    assert tuple(metadata[0]) == (role.value, LOCAL_DATABASE_SCHEMA_VERSION)
    assert tuple(pragmas[0]) == ("wal", 1, 3210)
    assert database.path.parent == tmp_path.resolve() / "local"
    assert (await database.health()).ready is True

    checkpoint = await database.checkpoint()
    assert checkpoint.busy_pages == 0
    assert checkpoint.log_pages >= checkpoint.checkpointed_pages
    await database.close()
    await database.close()
    with pytest.raises(StorageHealthError, match="closed"):
        await database.health()


@pytest.mark.asyncio
async def test_transactions_are_serialized_and_rollback_on_failure(tmp_path: Path) -> None:
    database = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    await database.execute("CREATE TABLE counters (id TEXT PRIMARY KEY, value INTEGER NOT NULL)")
    await database.execute("INSERT INTO counters(id, value) VALUES ('shared', 0)")

    def increment(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE counters SET value = value + 1 WHERE id = 'shared'")

    await asyncio.gather(*(database.transaction(increment) for _ in range(40)))
    rows = await database.fetch_all("SELECT value FROM counters WHERE id = 'shared'")
    assert rows[0][0] == 40

    def fail_after_write(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE counters SET value = 100 WHERE id = 'shared'")
        raise RuntimeError("rollback")

    with pytest.raises(RuntimeError, match="rollback"):
        await database.transaction(fail_after_write)
    rows = await database.fetch_all("SELECT value FROM counters WHERE id = 'shared'")
    assert rows[0][0] == 40
    await database.close()


@pytest.mark.asyncio
async def test_read_only_open_requires_current_database_and_rejects_writes(
    tmp_path: Path,
) -> None:
    with pytest.raises(StorageFormatError, match="does not exist"):
        LocalSQLiteDatabase.open(
            workspace_root=tmp_path,
            role=LocalDatabaseRole.EVENTS,
            mode=StorageOpenMode.READ_ONLY,
        )

    writable = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.EVENTS,
        mode=StorageOpenMode.READ_WRITE,
    )
    await writable.execute("CREATE TABLE records (id TEXT PRIMARY KEY)")
    await writable.execute("INSERT INTO records(id) VALUES ('event-1')")
    await writable.close()

    readonly = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.EVENTS,
        mode=StorageOpenMode.READ_ONLY,
    )
    assert [row[0] for row in await readonly.fetch_all("SELECT id FROM records")] == ["event-1"]
    with pytest.raises(StorageReadOnlyError):
        await readonly.execute("DELETE FROM records")
    with pytest.raises(StorageReadOnlyError):
        await readonly.transaction(lambda connection: None)
    with pytest.raises(StorageReadOnlyError):
        await readonly.checkpoint()
    await readonly.close()


def test_existing_unversioned_or_wrong_role_database_is_rejected_without_migration(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    local.mkdir()
    control_path = local / "control.sqlite3"
    connection = sqlite3.connect(control_path)
    connection.execute("CREATE TABLE legacy_records (id TEXT PRIMARY KEY)")
    connection.commit()
    connection.close()

    with pytest.raises(StorageFormatError, match="not a supported current schema"):
        LocalSQLiteDatabase.open(
            workspace_root=tmp_path,
            role=LocalDatabaseRole.CONTROL,
            mode=StorageOpenMode.READ_WRITE,
        )
    connection = sqlite3.connect(control_path)
    assert connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall() == [("legacy_records",)]
    connection.close()

    control_path.unlink()
    events = LocalSQLiteDatabase.open(
        workspace_root=tmp_path,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )
    path = events.path
    asyncio.run(events.close())
    connection = sqlite3.connect(path)
    connection.execute("UPDATE ag_storage_meta SET role = 'events'")
    connection.commit()
    connection.close()
    with pytest.raises(StorageFormatError, match="role or schema version"):
        LocalSQLiteDatabase.open(
            workspace_root=tmp_path,
            role=LocalDatabaseRole.CONTROL,
            mode=StorageOpenMode.READ_WRITE,
        )


@pytest.mark.parametrize(
    ("busy_timeout_ms", "durability", "error"),
    [
        (0, "normal", ValueError),
        (5000, "unsafe", ValueError),
    ],
)
def test_database_policy_rejects_invalid_options(
    tmp_path: Path,
    busy_timeout_ms: int,
    durability: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        LocalSQLiteDatabase.open(
            workspace_root=tmp_path,
            role=LocalDatabaseRole.SEARCH,
            mode=StorageOpenMode.READ_WRITE,
            busy_timeout_ms=busy_timeout_ms,
            durability=durability,
        )
    assert not (tmp_path / "local").exists()

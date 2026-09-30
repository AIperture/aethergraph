"""Explicit offline recovery using the existing local component migration owners."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

from aethergraph.storage.contracts import StorageFormatError, StorageOpenMode

from .control_repositories import _install as install_control
from .database import LocalDatabaseRole, LocalSQLiteDatabase, _validate_existing_database
from .manifest import read_local_workspace_manifest
from .observation_repository import _install as install_observations


def migrate_local_history(workspace: Path, backup: Path) -> dict[str, object]:
    """Migrate supported historical components with an offline backup receipt.

    Intro:
        Requires stopped writers and a clean manifest. Holds an exclusive SQLite
        write transaction across both existing component migrations; failures roll
        back all schema changes. The supplied backup must not already exist.

    Examples:
        Recover a stopped workspace:
            ```python
            receipt = migrate_local_history(Path("history"), Path("control.before.sqlite3"))
            assert receipt["status"] == "migrated"
            ```

        Verify current schemas with a fresh backup:
            ```python
            receipt = migrate_local_history(Path("history"), Path("control.current.sqlite3"))
            assert receipt["before"] == receipt["after"]
            ```

    Args:
        workspace: Exact operator-authorized local workspace, with all writers stopped.
        backup: New backup file outside the workspace; never overwritten.

    Returns:
        dict[str, object]: Workspace identity, component versions and backup checksum.

    Notes:
        Only control and observation component upgrades are supported. This operation
        does not infer legacy schemas or repair an unclean shutdown. Restoration
        requires stopped writers and replacement of the control database from backup.
    """
    root, target = workspace.resolve(), backup.resolve()
    manifest = read_local_workspace_manifest(root)
    if not manifest.clean_shutdown:
        raise StorageFormatError("history_migration_not_offline: workspace is active or unclean")
    if target.is_relative_to(root):
        raise ValueError("History backup must be outside the workspace")
    source = root / "local" / "control.sqlite3"
    _validate_existing_database(source, LocalDatabaseRole.CONTROL)
    connection = sqlite3.connect(
        f"{source.as_uri()}?mode=rw", uri=True, timeout=0.1, isolation_level=None
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        if connection.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            raise StorageFormatError(
                "Historical control database must use the local provider WAL mode"
            )
        connection.execute("BEGIN EXCLUSIVE")
        if not read_local_workspace_manifest(root).clean_shutdown:
            raise StorageFormatError("history_migration_not_offline: workspace became active")
        before = dict(connection.execute("SELECT name, version FROM ag_storage_components"))
        for name in ("control", "observations"):
            if before.get(name) not in (1, 2):
                raise StorageFormatError(
                    f"Unsupported historical {name} component: {before.get(name)!r}"
                )
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise StorageFormatError("Historical control database failed integrity validation")
        # Reserve the backup atomically; the reader includes committed WAL data.
        with target.open("xb"):
            pass
        reader = sqlite3.connect(f"{source.as_uri()}?mode=ro", uri=True)
        destination = sqlite3.connect(target)
        try:
            reader.backup(destination)
        finally:
            destination.close()
            reader.close()
        database = LocalSQLiteDatabase(
            role=LocalDatabaseRole.CONTROL,
            path=source,
            mode=StorageOpenMode.READ_WRITE,
            _connection=connection,
        )
        install_control(database)
        install_observations(database)
        after = dict(connection.execute("SELECT name, version FROM ag_storage_components"))
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise StorageFormatError("Migrated control database failed foreign-key validation")
        backup_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        connection.commit()
        return {
            "status": "migrated",
            "workspace_id": manifest.workspace_id,
            "before": before,
            "after": after,
            "backup": str(target),
            "backup_sha256": backup_hash,
        }
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()

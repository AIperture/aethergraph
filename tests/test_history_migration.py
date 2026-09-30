"""Offline recovery keeps both component upgrades atomic and backs up old evidence."""

import asyncio
from datetime import UTC, datetime
import json
import sqlite3

import pytest

from aethergraph.storage.contracts import StorageError, StorageFormatError, StorageOpenMode
from aethergraph.storage.providers.local_sqlite import LocalDatabaseRole, LocalSQLiteDatabase
from aethergraph.storage.providers.local_sqlite.history_migration import migrate_local_history


def seed(root, *, broken=False):
    root.mkdir()
    (root / "workspace.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "workspace_id": "history",
                "owner_scope": {},
                "provider": "local.sqlite",
                "config_fingerprint": "sha256:test",
                "created_at": datetime.now(UTC).isoformat(),
                "runtime_compatibility": {"storage_contract_version": 1},
                "lifecycle": {"clean_shutdown": True, "last_maintenance_at": None},
            }
        )
    )
    database = LocalSQLiteDatabase.open(
        workspace_root=root, role=LocalDatabaseRole.CONTROL, mode=StorageOpenMode.READ_WRITE
    )
    database.install_component(
        name="control",
        version=1,
        statements=(
            "CREATE TABLE local_runs(run_id TEXT PRIMARY KEY, started_at TEXT, metadata_json TEXT)",
            "INSERT INTO local_runs VALUES ('child', '2026-09-30', "
            '\'{"service_context":{"parent":{"run_id":"parent","session_id":"session"}}}\')',
        ),
    )
    database.install_component(
        name="observations",
        version=1,
        statements=(
            "CREATE TABLE local_llm_calls(llm_call_id TEXT PRIMARY KEY"
            + (", lifecycle_status TEXT)" if broken else ")"),
            "INSERT INTO local_llm_calls(llm_call_id) VALUES ('call')",
        ),
    )
    asyncio.run(database.close())
    return root / "local" / "control.sqlite3"


def versions(path):
    with sqlite3.connect(path) as db:
        return dict(db.execute("SELECT name,version FROM ag_storage_components"))


def test_offline_upgrade_preserves_evidence_and_backup(tmp_path):
    root, backup = tmp_path / "history", tmp_path / "before.sqlite3"
    source = seed(root)
    receipt = migrate_local_history(root, backup)
    assert receipt["before"] == {"control": 1, "observations": 1}
    assert receipt["after"] == {"control": 2, "observations": 2}
    assert versions(backup) == receipt["before"]
    with sqlite3.connect(source) as db:
        assert db.execute("SELECT parent_run_id,parent_session_id FROM local_runs").fetchone() == (
            "parent",
            "session",
        )
        assert db.execute("SELECT lifecycle_status FROM local_llm_calls").fetchone() == (
            "completed",
        )
    repeat = migrate_local_history(root, tmp_path / "again.sqlite3")
    assert repeat["before"] == repeat["after"]


def test_late_migration_failure_rolls_back_earlier_component(tmp_path):
    root, backup = tmp_path / "history", tmp_path / "before.sqlite3"
    source = seed(root, broken=True)
    with pytest.raises(StorageError) as failure:
        migrate_local_history(root, backup)
    assert "duplicate column" in str(failure.value.__cause__)
    assert versions(source) == versions(backup) == {"control": 1, "observations": 1}
    with sqlite3.connect(source) as db:
        assert "parent_run_id" not in [
            row[1] for row in db.execute("PRAGMA table_info(local_runs)")
        ]


def test_active_writer_and_unclean_workspace_are_rejected(tmp_path):
    root, backup = tmp_path / "history", tmp_path / "before.sqlite3"
    source = seed(root)
    with sqlite3.connect(source) as writer:
        writer.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            migrate_local_history(root, backup)
        writer.rollback()
    assert not backup.exists()
    manifest = json.loads((root / "workspace.json").read_text())
    manifest["lifecycle"]["clean_shutdown"] = False
    (root / "workspace.json").write_text(json.dumps(manifest))
    with pytest.raises(StorageFormatError, match="active or unclean"):
        migrate_local_history(root, backup)
    assert not backup.exists()


def test_unknown_version_and_existing_backup_are_preserved(tmp_path):
    root, backup = tmp_path / "history", tmp_path / "before.sqlite3"
    source = seed(root)
    backup.write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        migrate_local_history(root, backup)
    assert backup.read_bytes() == b"do not overwrite"
    with sqlite3.connect(source) as db:
        db.execute("UPDATE ag_storage_components SET version=99 WHERE name='observations'")
    with pytest.raises(StorageFormatError, match="Unsupported historical"):
        migrate_local_history(root, tmp_path / "new.sqlite3")
    assert versions(source)["control"] == 1

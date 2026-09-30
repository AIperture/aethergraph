"""Canonical parent lineage is immutable, scoped, paginated and migratable."""

from dataclasses import replace
import json

import pytest
from test_local_control_repositories import _database, _run
from test_observability_workspace import _provider_and_request

from aethergraph.observability import ObservabilityIdentity, open_observability_workspace
from aethergraph.storage.contracts import (
    PageRequest,
    RunQuery,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageMigrationRequiredError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalRunRepository


@pytest.mark.asyncio
async def test_child_query_pages_across_sessions_and_preserves_scope_and_parent_fence(tmp_path):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    try:
        runs = LocalRunRepository(database=database)
        for index in range(5):
            original = _run(f"child-{index}")
            await runs.create(
                replace(
                    original,
                    scope=replace(original.scope, session_id=f"isolated-{index}"),
                    parent_run_id="parent",
                    parent_session_id="parent-session",
                )
            )
        await runs.create(
            replace(_run("unrelated"), parent_run_id="other", parent_session_id="parent-session")
        )
        await runs.create(
            replace(
                _run("foreign", project_id="other-project"),
                parent_run_id="parent",
                parent_session_id="parent-session",
            )
        )
        query = RunQuery(
            scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
            parent_run_id="parent",
            page=PageRequest(limit=2),
        )
        first = await runs.query(query)
        second = await runs.query(
            replace(query, page=PageRequest(limit=2, cursor=first.next_cursor))
        )
        third = await runs.query(
            replace(query, page=PageRequest(limit=2, cursor=second.next_cursor))
        )
        assert [row.run_id for row in (*first.items, *second.items, *third.items)] == [
            f"child-{index}" for index in reversed(range(5))
        ]
        assert third.next_cursor is None
        with pytest.raises(StorageConfigurationError, match="mismatched"):
            await runs.query(
                replace(query, parent_run_id="other", page=PageRequest(cursor=first.next_cursor))
            )
        with pytest.raises(StorageIntegrityError, match="immutable"):
            await runs.compare_and_set(
                replace(first.items[0], revision=2, parent_run_id="other"), 1
            )
        assert not (
            await runs.query(replace(query, scope=StorageScope(project_id="missing")))
        ).items
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_child_facade_requires_access_to_parent_and_children(tmp_path):
    provider, request = _provider_and_request(tmp_path)
    bundle = provider.open(request)
    try:
        for identity, user, parent in (
            ("parent", "user-1", None),
            ("child", "user-1", "parent"),
            ("private-child", "user-2", "parent"),
            ("other-parent", "user-2", None),
        ):
            record = _run(identity)
            await bundle.runs.create(
                replace(
                    record,
                    scope=replace(
                        record.scope, tenant_id=None, user_id=user, session_id=f"session-{identity}"
                    ),
                    parent_run_id=parent,
                    parent_session_id="session-parent" if parent else None,
                )
            )
    finally:
        await bundle.close()
    facade = open_observability_workspace(
        tmp_path, identity=ObservabilityIdentity(mode="cloud", user_id="user-1")
    )
    try:
        page = await facade.page_runs(parent_run_id="parent", limit=1)
        assert [row["run_id"] for row in page["items"]] == ["child"]
        assert page["items"][0]["parent"] == {"run_id": "parent", "session_id": "session-parent"}
        assert not (await facade.page_runs(parent_run_id="other-parent"))["items"]
        assert not (await facade.page_runs(parent_run_id="missing"))["items"]
        assert not (await facade.page_runs(parent_run_id="parent", session_id="session-parent"))[
            "items"
        ]
    finally:
        await facade.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [False, True])
async def test_v1_parent_metadata_migrates_once_and_read_only_requires_upgrade(tmp_path, malformed):
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    # A retained v1 control table, before canonical parent columns existed.
    database.install_component(
        name="control",
        version=1,
        statements=(
            """
        CREATE TABLE local_runs (
            run_id TEXT PRIMARY KEY, graph_id TEXT NOT NULL, tenant_id TEXT,
            project_id TEXT, org_id TEXT, user_id TEXT, session_id TEXT,
            node_id TEXT, agent_id TEXT, scope_key TEXT, kind TEXT NOT NULL,
            status TEXT NOT NULL, revision INTEGER NOT NULL, started_at TEXT NOT NULL,
            finished_at TEXT, tags_json TEXT NOT NULL, error TEXT, metadata_json TEXT NOT NULL,
            artifact_count INTEGER NOT NULL, first_artifact_at TEXT, last_artifact_at TEXT,
            recent_artifact_ids_json TEXT NOT NULL, result_available INTEGER NOT NULL,
            result_updated_at TEXT, schema_version INTEGER NOT NULL
        )
        """,
        ),
    )
    metadata = json.dumps(
        {
            "service_context": {
                "parent": {} if malformed else {"run_id": "parent", "session_id": "parent-session"}
            },
            "metadata": {"keep": "value"},
        }
    )
    await database.transaction(
        lambda conn: conn.execute(
            "INSERT INTO local_runs(run_id, graph_id, project_id, session_id, kind, status, revision, started_at, tags_json, metadata_json, artifact_count, recent_artifact_ids_json, result_available, schema_version) "
            "VALUES ('child', 'graph', 'project-1', 'child-session', 'graphfn', 'running', 1, '2026-09-26T00:00:00+00:00', '[]', ?, 0, '[]', 0, 1)",
            (metadata,),
        )
    )
    await database.close()
    readonly = _database(tmp_path, StorageOpenMode.READ_ONLY)
    try:
        with pytest.raises(StorageMigrationRequiredError, match="requires migration"):
            LocalRunRepository(database=readonly)
    finally:
        await readonly.close()
    writable = _database(tmp_path, StorageOpenMode.READ_WRITE)
    try:
        if malformed:
            with pytest.raises(StorageIntegrityError):
                LocalRunRepository(database=writable)
            rows = await writable.fetch_all(
                "SELECT version FROM ag_storage_components WHERE name = 'control'"
            )
            assert rows[0]["version"] == 1
            rows = await writable.fetch_all(
                "SELECT metadata_json FROM local_runs WHERE run_id = 'child'"
            )
            assert rows[0]["metadata_json"] == metadata
            return
        runs = LocalRunRepository(database=writable)
        record = await runs.get(StorageScope(project_id="project-1"), "child")
        assert (record.parent_run_id, record.parent_session_id) == ("parent", "parent-session")
        assert "parent" not in record.metadata["service_context"]
        assert record.metadata["metadata"]["keep"] == "value"
        LocalRunRepository(database=writable)  # Idempotent component opening.
        assert (
            await runs.query(
                RunQuery(scope=StorageScope(project_id="project-1"), parent_run_id="parent")
            )
        ).items == (record,)
    finally:
        await writable.close()

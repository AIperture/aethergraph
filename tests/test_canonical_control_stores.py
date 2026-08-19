from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.services.state_stores import GraphSnapshot
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import (
    RunRecord,
    RunStatus,
    SessionKind,
)
from aethergraph.services.control import (
    CanonicalRunResultStore,
    CanonicalRunStore,
    CanonicalSessionStore,
    bind_canonical_control_stores,
)
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.storage.contracts import StorageIntegrityError, StorageOpenMode, StorageScope
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalRunRepository,
    LocalRunResultRepository,
    LocalSessionRepository,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 22, tzinfo=UTC)
OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def __init__(self) -> None:
        self.value = NOW

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(seconds=1)
        return value


class _Identity:
    def __init__(self, user_id: str, org_id: str) -> None:
        self.user_id = user_id
        self.org_id = org_id


class _StateStore:
    def __init__(self) -> None:
        self.snapshots: dict[str, GraphSnapshot] = {}

    async def save_snapshot(self, scope: StorageScope, snap: GraphSnapshot) -> None:
        self.snapshots[snap.run_id] = snap

    async def load_latest_snapshot(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> GraphSnapshot | None:
        return self.snapshots.get(run_id)


@pytest.fixture
def dummy_meter(monkeypatch: pytest.MonkeyPatch):
    class _Meter:
        async def record_run(self, **kwargs) -> None:
            return None

    meter = _Meter()
    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.current_metering",
        lambda: meter,
    )
    return meter


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=StorageOpenMode.READ_WRITE,
    )


def _stores(
    database: LocalSQLiteDatabase,
    clock: _Clock,
) -> tuple[
    CanonicalRunStore,
    CanonicalRunResultStore,
    CanonicalSessionStore,
    LocalRunRepository,
    LocalRunResultRepository,
    LocalSessionRepository,
]:
    runs = LocalRunRepository(database=database)
    results = LocalRunResultRepository(database=database)
    sessions = LocalSessionRepository(database=database)
    return (
        CanonicalRunStore(repository=runs, owner_scope=OWNER, clock=clock.now),
        CanonicalRunResultStore(repository=results, runs=runs, owner_scope=OWNER),
        CanonicalSessionStore(repository=sessions, owner_scope=OWNER, clock=clock.now),
        runs,
        results,
        sessions,
    )


@pytest.mark.asyncio
async def test_run_projection_preserves_scope_metadata_queries_and_occurrences(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    clock = _Clock()
    store, _, _, repository, _, _ = _stores(database, clock)
    record = RunRecord(
        run_id="run-1",
        graph_id="graph-1",
        kind="taskgraph",
        status=RunStatus.running,
        started_at=NOW,
        tags=["tag-1"],
        user_id="user-1",
        org_id="org-1",
        session_id="session-1",
        app_id="deprecated-app",
        meta={"phase": "start"},
    )
    await store.create(record)

    canonical = await repository.get(
        StorageScope(project_id="project-1", run_id="run-1"),
        "run-1",
    )
    assert canonical is not None
    assert "app_id" not in canonical.scope.as_filter()
    assert canonical.metadata["compatibility_metadata"]["app_id"] == {
        "value": "deprecated-app",
        "deprecated": True,
        "scheduled_removal": "future breaking release",
    }
    projected = await store.get("run-1")
    assert projected is not None
    assert projected.app_id == "deprecated-app"
    assert projected.meta == {"phase": "start"}
    with pytest.raises(ValueError, match="public metadata reserves"):
        await store.update_status(
            "run-1",
            RunStatus.running,
            meta_update={"app_id": "must-be-explicit"},
        )
    with pytest.raises(ValueError, match="app_id must be a non-empty"):
        await store.create(replace(record, run_id="run-empty-app", app_id=""))
    with pytest.raises(ValueError, match="public metadata reserves"):
        await store.create(
            replace(
                record,
                run_id="run-reserved-meta",
                app_id=None,
                meta={"app_id": "must-be-explicit"},
            )
        )

    await store.update_status(
        "run-1",
        RunStatus.waiting,
        meta_update={"phase": "waiting"},
    )
    await store.record_artifact(
        "run-1",
        artifact_id="content-1",
        occurrence_id="occurrence-1",
        created_at=NOW + timedelta(seconds=2),
    )
    await store.record_artifact(
        "run-1",
        artifact_id="content-1",
        occurrence_id="occurrence-1",
        created_at=NOW + timedelta(seconds=2),
    )
    updated = await store.get("run-1")
    assert updated is not None
    assert updated.status == RunStatus.waiting
    assert updated.meta["phase"] == "waiting"
    assert updated.artifact_count == 1
    assert updated.recent_artifact_ids == ["content-1"]
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await store.record_artifact(
            "run-1",
            artifact_id="different-content",
            occurrence_id="occurrence-1",
            created_at=NOW + timedelta(seconds=2),
        )
    assert await store.list(status=RunStatus.waiting, limit=10) == [updated]
    with pytest.raises(ValueError, match="must not exceed"):
        await store.list(limit=2, offset=999)

    finished_at = NOW + timedelta(minutes=1)
    finished = replace(
        record,
        run_id="run-finished",
        app_id=None,
        status=RunStatus.succeeded,
        finished_at=finished_at,
    )
    await store.create(finished)
    await store.update_status(finished.run_id, RunStatus.succeeded)
    preserved = await store.get(finished.run_id)
    assert preserved is not None and preserved.finished_at == finished_at
    assert await store.list(limit=1, offset=1) == [updated]

    malformed = replace(record, run_id="run-malformed", app_id=None)
    await store.create(malformed)
    malformed_record = await repository.get(
        StorageScope(project_id="project-1", run_id=malformed.run_id),
        malformed.run_id,
    )
    assert malformed_record is not None
    await repository.compare_and_set(
        replace(
            malformed_record,
            revision=2,
            metadata={
                "compatibility_metadata": {
                    "app_id": {
                        "value": "legacy-app",
                        "deprecated": False,
                    }
                }
            },
        ),
        1,
    )
    with pytest.raises(ValueError, match="compatibility metadata is malformed"):
        await store.get(malformed.run_id)
    await database.close()


@pytest.mark.asyncio
async def test_session_projection_cas_query_occurrence_and_delete(tmp_path: Path) -> None:
    database = _database(tmp_path)
    clock = _Clock()
    _, _, store, _, _, repository = _stores(database, clock)
    created = await store.create(
        session_id="session-1",
        kind=SessionKind.chat,
        user_id="user-1",
        org_id="org-1",
        title="Initial",
    )
    assert created.title_source == "manual"
    replayed = await store.create(
        session_id="session-1",
        kind=SessionKind.chat,
        user_id="user-1",
        org_id="org-1",
        title="Ignored replay title",
    )
    assert replayed == created
    with pytest.raises(ValueError, match="identity collision"):
        await store.create(
            session_id="session-1",
            kind=SessionKind.chat,
            user_id="user-1",
            org_id="org-1",
            source="different",
        )
    winners = await asyncio.gather(
        store.create(
            session_id="session-race",
            kind=SessionKind.chat,
            user_id="user-2",
            org_id="org-1",
            title="First contender",
        ),
        store.create(
            session_id="session-race",
            kind=SessionKind.chat,
            user_id="user-2",
            org_id="org-1",
            title="Second contender",
        ),
    )
    assert winners[0] == winners[1]
    assert winners[0].title in {"First contender", "Second contender"}
    updated = await store.update(
        created.session_id,
        title="Generated",
        title_source="auto",
        external_ref="external-1",
    )
    assert updated is not None
    assert updated.title_source == "auto"
    with pytest.raises(ValueError, match="title_source"):
        await store.update(
            created.session_id,
            title="Invalid provenance",
            title_source="invalid",  # type: ignore[arg-type]
        )
    assert await store.get(created.session_id) == updated
    cleared = await store.update(
        created.session_id,
        title="",
        external_ref="",
    )
    assert cleared is not None
    assert cleared.title == ""
    assert cleared.external_ref == ""
    await store.touch(created.session_id, updated_at=NOW + timedelta(minutes=1))
    await store.record_artifact(
        created.session_id,
        occurrence_id="occurrence-1",
        created_at=NOW + timedelta(minutes=2),
    )
    await store.record_artifact(
        created.session_id,
        occurrence_id="occurrence-1",
        created_at=NOW + timedelta(minutes=2),
    )
    projected = await store.get(created.session_id)
    assert projected is not None and projected.artifact_count == 1
    assert await store.storage_scope(created.session_id) == StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
        session_id=created.session_id,
    )
    assert await store.storage_scope("missing") is None
    assert await store.list_for_user(user_id="user-1", kind=SessionKind.chat) == [projected]
    await store.delete(created.session_id)
    assert await store.get(created.session_id) is None
    assert (
        await repository.get(
            StorageScope(project_id="project-1", session_id=created.session_id),
            created.session_id,
        )
        is None
    )
    await database.close()


@pytest.mark.asyncio
async def test_run_manager_recovers_deleted_canonical_result_from_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dummy_meter,
) -> None:
    database = _database(tmp_path)
    clock = _Clock()
    runs, results, _, repository, _, _ = _stores(database, clock)
    state_store = _StateStore()
    manager = RunManager(
        run_store=runs,
        result_store=results,
        state_store=state_store,
        registry=UnifiedRegistry(),
    )

    async def fake_resolve(self, graph_id: str):
        return object()

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        return {"out": 7}

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )
    record, _, _, _ = await manager.start_run(
        graph_id="graph-1",
        inputs={"x": 1},
        identity=_Identity(user_id="user-1", org_id="org-1"),
    )
    waited, outputs = await manager.wait_run(record.run_id, return_outputs=True)
    assert waited.status == RunStatus.succeeded
    assert outputs == {"out": 7}
    direct = await results.get(record.run_id)
    assert direct is not None
    with pytest.raises(TypeError, match="outputs must be an object"):
        await results.save(
            record.run_id,
            replace(direct, outputs=[]),  # type: ignore[arg-type]
        )
    assert await results.get(record.run_id) == direct

    await results.delete(record.run_id)
    cleared = await repository.get(
        StorageScope(project_id="project-1", run_id=record.run_id),
        record.run_id,
    )
    assert cleared is not None and cleared.result_available is False
    await state_store.save_snapshot(
        StorageScope(org_id="org-1", user_id="user-1", run_id=record.run_id, graph_id="graph-1"),
        GraphSnapshot(
            run_id=record.run_id,
            graph_id="graph-1",
            rev=3,
            created_at=0.0,
            spec_hash="demo",
            state={"graph_outputs": {"out": 99}},
        ),
    )
    recovered_record, recovered_outputs = await manager.wait_run(
        record.run_id,
        return_outputs=True,
    )
    assert recovered_record.status == RunStatus.succeeded
    assert recovered_outputs == {"out": 99}
    recovered = await results.get(record.run_id)
    assert recovered is not None
    assert recovered.outputs == {"out": 99}
    assert recovered.source == "snapshot_recovered"
    assert recovered.snapshot_rev == 3
    await database.close()


def test_control_binding_uses_exact_bundle_fields_and_has_no_close() -> None:
    runs = object()
    results = object()
    sessions = object()
    bundle = SimpleNamespace(runs=runs, run_results=results, sessions=sessions)
    stores = bind_canonical_control_stores(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=OWNER,
        clock=lambda: NOW,
    )
    assert stores.runs._repository is runs
    assert stores.run_results._repository is results
    assert stores.run_results._runs is runs
    assert stores.sessions._repository is sessions
    assert not hasattr(stores.runs, "close")
    assert not hasattr(stores.run_results, "close")
    assert not hasattr(stores.sessions, "close")


def test_control_projection_public_docstrings_follow_required_style() -> None:
    methods = {
        CanonicalRunStore: (
            "__init__",
            "create",
            "update_status",
            "get",
            "list",
            "record_artifact",
        ),
        CanonicalRunResultStore: ("__init__", "save", "get", "delete"),
        CanonicalSessionStore: (
            "__init__",
            "create",
            "get",
            "list_for_user",
            "touch",
            "update",
            "delete",
            "record_artifact",
        ),
    }
    for owner, names in methods.items():
        for name in names:
            docstring = inspect.getdoc(getattr(owner, name)) or ""
            positions = tuple(
                docstring.index(section) for section in ("Examples:", "Args:", "Returns:", "Notes:")
            )
            assert positions == tuple(sorted(positions)), (owner.__name__, name)
            assert docstring.count("```python") >= 2, (owner.__name__, name)

    app_field = next(item for item in fields(RunRecord) if item.name == "app_id")
    assert app_field.metadata["deprecated"] is True

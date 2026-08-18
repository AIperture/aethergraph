from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.services.state_stores import GraphSnapshot, GraphStateStore, StateEvent
from aethergraph.services.state_stores.canonical_store import CanonicalGraphStateStore
from aethergraph.services.state_stores.scope import scope_for_runtime_env
from aethergraph.storage.contracts import (
    RunRecord,
    RunStatus,
    StorageConflictError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://tests/graph-state"
_SECRET = b"canonical-graph-state-secret-32-bytes"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 7, tzinfo=UTC)

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


def _open_bundle(root: Path, clock: _Clock):
    provider = LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    )
    return provider.open(
        StorageOpenRequest(
            workspace_id="graph-state-tests",
            workspace_root=root.resolve(),
            owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={"continuation_token_secret_ref": _SECRET_REF},
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=clock,
            secrets=_Secrets(),
        )
    )


def _scope(*, run_id: str = "run-1", graph_id: str = "graph-1") -> StorageScope:
    return StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
        session_id="session-1",
        run_id=run_id,
        graph_id=graph_id,
        agent_id="agent-1",
    )


def _snapshot(*, rev: int, state: dict | None = None) -> GraphSnapshot:
    return GraphSnapshot(
        run_id="run-1",
        graph_id="graph-1",
        rev=rev,
        created_at=float(rev),
        spec_hash="spec-1",
        state=state or {"nodes": {}},
        started_at=datetime(2026, 8, 16, 7, tzinfo=UTC),
    )


def test_runtime_graph_state_scope_excludes_provenance_and_deprecated_app_metadata() -> None:
    env = SimpleNamespace(
        identity=SimpleNamespace(org_id="org-1", user_id="user-1", client_id="client-1"),
        run_id="run-1",
        graph_id="graph-1",
        session_id="session-1",
        agent_id="agent-1",
        app_id="app-1",
    )

    assert scope_for_runtime_env(env).as_filter() == {
        "org_id": "org-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "graph_id": "graph-1",
    }


def test_graph_state_public_docstrings_follow_required_format() -> None:
    methods = (
        "save_snapshot",
        "load_latest_snapshot",
        "append_event",
        "load_events_since",
        "list_run_ids",
    )
    for owner in (
        GraphStateStore,
        CanonicalGraphStateStore,
    ):
        for name in methods:
            docstring = inspect.getdoc(getattr(owner, name)) or ""
            assert docstring.index("Examples:") < docstring.index("Args:")
            assert docstring.index("Args:") < docstring.index("Returns:")
            assert docstring.index("Returns:") < docstring.index("Notes:")
            assert docstring.count("```python") >= 2


@pytest.mark.asyncio
async def test_canonical_graph_state_round_trip_is_scoped_and_retry_safe(tmp_path: Path) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    store = CanonicalGraphStateStore(
        state_store=bundle.state,
        event_store=bundle.events,
        run_repository=bundle.runs,
    )
    scope = _scope()
    snapshot = _snapshot(rev=3)
    event = StateEvent(
        run_id="run-1",
        graph_id="graph-1",
        rev=4,
        ts=datetime(2026, 8, 16, 7, 1, tzinfo=UTC).timestamp(),
        kind="STATUS",
        payload={"node_id": "node-1", "status": "RUNNING"},
    )

    try:
        await store.save_snapshot(scope, snapshot)
        await store.save_snapshot(scope, snapshot)
        await store.append_event(scope, event)
        await store.append_event(scope, event)

        assert await store.load_latest_snapshot(scope, "run-1") == snapshot
        assert (
            await store.load_latest_snapshot(
                StorageScope(**{**scope.as_filter(), "org_id": "org-2"}), "run-1"
            )
            is None
        )
        assert await store.load_events_since(scope, "run-1", 3) == [event]
        assert await store.load_events_since(scope, "run-1", 4) == []
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_graph_state_rejects_stale_snapshot_revision(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path, _Clock())
    store = CanonicalGraphStateStore(
        state_store=bundle.state,
        event_store=bundle.events,
        run_repository=bundle.runs,
    )
    try:
        await store.save_snapshot(_scope(), _snapshot(rev=5))

        with pytest.raises(StorageConflictError, match="cannot move backward"):
            await store.save_snapshot(_scope(), _snapshot(rev=4))
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_graph_state_lists_runs_through_run_repository(tmp_path: Path) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    store = CanonicalGraphStateStore(
        state_store=bundle.state,
        event_store=bundle.events,
        run_repository=bundle.runs,
    )
    first_scope = _scope()
    second_scope = _scope(run_id="run-2", graph_id="graph-2")
    try:
        await bundle.runs.create(
            RunRecord(
                run_id="run-1",
                graph_id="graph-1",
                kind="taskgraph",
                status=RunStatus.RUNNING,
                scope=first_scope,
                revision=1,
                started_at=clock.now(),
            )
        )
        await bundle.runs.create(
            RunRecord(
                run_id="run-2",
                graph_id="graph-2",
                kind="taskgraph",
                status=RunStatus.RUNNING,
                scope=second_scope,
                revision=1,
                started_at=clock.now(),
            )
        )

        owner_scope = StorageScope(
            tenant_id="tenant-1",
            project_id="project-1",
            org_id="org-1",
            user_id="user-1",
        )
        assert await store.list_run_ids(owner_scope) == ["run-2", "run-1"]
        assert await store.list_run_ids(owner_scope, graph_id="graph-1") == ["run-1"]
    finally:
        await bundle.close()

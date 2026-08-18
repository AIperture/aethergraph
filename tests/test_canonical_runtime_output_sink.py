from __future__ import annotations

from datetime import UTC, datetime
from inspect import getdoc
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aethergraph.contracts.services.runtime_output import RuntimeOutputFrame
from aethergraph.observability import (
    CanonicalRuntimeOutputSink,
    bind_canonical_runtime_output,
)
from aethergraph.storage.contracts import (
    RuntimeOutputQuery,
    StorageCapacityError,
    StorageOpenMode,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalRuntimeOutputSink,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 4, tzinfo=UTC)
_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1", org_id="org-1")


def _database(root: Path) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.EVENTS,
        mode=StorageOpenMode.READ_WRITE,
    )


def _frame(sequence: int, *, text: str | None = None) -> RuntimeOutputFrame:
    return RuntimeOutputFrame(
        execution_id="execution-1",
        run_id="run-1",
        session_id="session-1",
        graph_id="graph-1",
        node_id="node-1",
        tool_name="reporter",
        stream="stdout",
        sequence=sequence,
        text=text or f"line-{sequence}",
        source="python.stream",
    )


@pytest.mark.asyncio
async def test_canonical_runtime_output_projects_exact_scope_identity_and_tags(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    provider_sink = LocalRuntimeOutputSink(database=database)
    sink = CanonicalRuntimeOutputSink(
        sink=provider_sink,
        owner_scope=_OWNER,
        tags=("host-test",),
    )

    sink.emit(_frame(1, text="hello\x00"))
    await sink.flush_execution("execution-1")
    rows = await database.fetch_all("SELECT * FROM local_runtime_output")

    assert len(rows) == 1
    row = rows[0]
    assert row["output_id"] == "runtime:execution-1:1"
    assert row["tenant_id"] == "tenant-1"
    assert row["project_id"] == "project-1"
    assert row["session_id"] == "session-1"
    assert row["run_id"] == "run-1"
    assert row["node_id"] == "node-1"
    assert row["text"] == "hello"
    assert json.loads(row["tags_json"]) == ["runtime-console", "host-test"]
    page = await sink.query(RuntimeOutputQuery(scope=StorageScope(run_id="run-1")))
    assert len(page.items) == 1
    assert page.items[0].scope == StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
    )
    with pytest.raises(ValueError, match="conflicts with owner_scope project_id"):
        await sink.query(RuntimeOutputQuery(scope=StorageScope(project_id="other", run_id="run-1")))
    await database.close()


@pytest.mark.asyncio
async def test_canonical_runtime_output_reuses_shared_bounding_policy(tmp_path: Path) -> None:
    database = _database(tmp_path)
    provider_sink = LocalRuntimeOutputSink(database=database)
    sink = CanonicalRuntimeOutputSink(
        sink=provider_sink,
        owner_scope=_OWNER,
        max_rows_per_run=2,
    )

    for sequence in range(1, 8):
        sink.emit(_frame(sequence))
    await sink.flush_run("run-1")
    rows = await database.fetch_all(
        "SELECT text, truncated FROM local_runtime_output ORDER BY cursor"
    )

    assert [row["text"] for row in rows] == [
        "line-1",
        "line-2",
        "[runtime output truncated]",
    ]
    assert [bool(row["truncated"]) for row in rows] == [False, False, True]
    await database.close()


@pytest.mark.asyncio
async def test_canonical_runtime_output_capacity_failure_restores_service_bounds(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path)
    provider_sink = LocalRuntimeOutputSink(database=database, max_pending_frames=1)
    sink = CanonicalRuntimeOutputSink(
        sink=provider_sink,
        owner_scope=_OWNER,
        max_rows_per_run=2,
    )
    sink.emit(_frame(1))
    with pytest.raises(StorageCapacityError):
        sink.emit(_frame(2))
    await sink.flush_execution("execution-1")

    sink.emit(_frame(2))
    await sink.flush_execution("execution-1")
    rows = await database.fetch_all(
        "SELECT text, truncated FROM local_runtime_output ORDER BY cursor"
    )
    assert [row["text"] for row in rows] == ["line-1", "line-2"]
    assert not any(bool(row["truncated"]) for row in rows)
    await database.close()


def test_canonical_runtime_output_factory_maps_only_bundle_sink_without_io() -> None:
    provider_sink = object()
    bundle = SimpleNamespace(runtime_output=provider_sink)

    sink = bind_canonical_runtime_output(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=_OWNER,
        tags=("host-test",),
    )

    assert isinstance(sink, CanonicalRuntimeOutputSink)
    assert sink._sink is provider_sink
    assert sink._owner_scope == _OWNER


@pytest.mark.parametrize(
    "tags",
    [
        ("runtime-console",),
        ("app_id:legacy",),
        ("application_id:alias",),
        ("client_id:compatibility",),
    ],
)
def test_canonical_runtime_output_rejects_duplicate_or_deprecated_tags(
    tags: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        CanonicalRuntimeOutputSink(
            sink=object(),  # type: ignore[arg-type]
            owner_scope=_OWNER,
            tags=tags,
        )


def test_canonical_runtime_output_public_docstrings_follow_strict_contract() -> None:
    methods = (
        CanonicalRuntimeOutputSink.__init__,
        CanonicalRuntimeOutputSink.emit,
        CanonicalRuntimeOutputSink.flush_execution,
        CanonicalRuntimeOutputSink.flush_run,
        CanonicalRuntimeOutputSink.query,
        bind_canonical_runtime_output,
    )
    for method in methods:
        docstring = getdoc(method)
        assert docstring is not None
        assert docstring.count("```python") == 2
        positions = [
            docstring.index(section) for section in ("Examples:", "Args:", "Returns:", "Notes:")
        ]
        assert positions == sorted(positions)

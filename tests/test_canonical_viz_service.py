from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from typing import get_type_hints

import pytest
from storage_conformance.external_provider import DeterministicExternalProvider

from aethergraph.contracts.services.viz import VizEvent, VizEventSink
from aethergraph.services.viz.canonical_service import (
    CanonicalVizService,
    build_canonical_viz_service,
)
from aethergraph.services.viz.facade import VizFacade
from aethergraph.storage.contracts import (
    EventQuery,
    PageRequest,
    SortDirection,
    StorageCapacityError,
    StorageConfigurationError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalEventStore,
    LocalSQLiteDatabase,
)

_NOW = datetime(2026, 8, 16, 18, tzinfo=UTC)
_OWNER = StorageScope(tenant_id="tenant-1", project_id="project-1")


class _Clock:
    def __init__(self) -> None:
        self.value = _NOW

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Identities:
    def __init__(self) -> None:
        self.value = 0

    def next(self) -> str:
        self.value += 1
        return f"viz-{self.value}"


class _Secrets:
    async def resolve(self, reference: str) -> str:
        return f"resolved:{reference}"


def _event(
    kind: str,
    *,
    run_id: str = "run-1",
    step: int = 1,
    app_id: str | None = None,
    client_id: str | None = None,
) -> VizEvent:
    values: dict[str, object] = {
        "value": None,
        "vector": None,
        "matrix": None,
        "artifact_id": None,
    }
    if kind == "scalar":
        values["value"] = float(step)
    elif kind == "vector":
        values["vector"] = [1.0, 2.0]
    elif kind == "matrix":
        values["matrix"] = [[1.0, 2.0], [3.0, 4.0]]
    elif kind == "image":
        values["artifact_id"] = f"artifact-{step}"
    return VizEvent(
        run_id=run_id,
        graph_id="graph-1",
        node_id=f"node-{step}",
        tool_name="renderer",
        tool_version="1",
        track_id=f"track-{kind}",
        figure_id="figure-1",
        viz_kind=kind,  # type: ignore[arg-type]
        step=step,
        mode="append",
        value=values["value"],  # type: ignore[arg-type]
        vector=values["vector"],  # type: ignore[arg-type]
        matrix=values["matrix"],  # type: ignore[arg-type]
        artifact_id=values["artifact_id"],  # type: ignore[arg-type]
        meta={"caption": f"frame {step}"},
        tags=["verified"],
        app_id=app_id,
        client_id=client_id,
    )


def _service(event_store) -> CanonicalVizService:
    return CanonicalVizService(
        event_store=event_store,
        owner_scope=_OWNER,
        clock=_Clock().now,
        event_id_factory=_Identities().next,
    )


def _local_events(root: Path) -> tuple[LocalSQLiteDatabase, LocalEventStore]:
    database = LocalSQLiteDatabase.open(
        workspace_root=root.resolve(),
        role=LocalDatabaseRole.EVENTS,
        mode=StorageOpenMode.READ_WRITE,
        busy_timeout_ms=5_000,
        durability="normal",
    )
    return database, LocalEventStore(database=database, stream="runtime")


async def _exercise_provider_service(service: CanonicalVizService) -> None:
    await service.append(_event("scalar", step=1, app_id="app-compat", client_id="client-old"))
    await service.append(_event("image", step=2))
    await service.append(_event("vector", run_id="run-2", step=3))

    run = await service.query_run("run-1")
    assert [(event.viz_kind, event.step) for event in run] == [("scalar", 1), ("image", 2)]
    assert run[0].deprecated_app_id == "app-compat"
    assert run[0].deprecated_client_id == "client-old"
    assert run[0].vector is None
    assert run[1].artifact_id == "artifact-2"
    with pytest.raises(TypeError):
        run[0].meta["caption"] = "changed"  # type: ignore[index]

    scalar = await service.query_run("run-1", kinds=("scalar",))
    assert [event.viz_kind for event in scalar] == ["scalar"]

    first = await service.query_run_page("run-1", page=PageRequest(limit=1))
    assert len(first.items) == 1
    assert first.next_cursor is not None
    second = await service.query_run_page(
        "run-1",
        page=PageRequest(limit=1, cursor=first.next_cursor),
    )
    assert len(second.items) == 1
    assert {first.items[0].event_id, second.items[0].event_id} == {"viz-1", "viz-2"}

    with pytest.raises(StorageCapacityError, match="1-event ceiling"):
        await service.query_run("run-1", max_events=1)


@pytest.mark.asyncio
async def test_canonical_viz_service_passes_local_provider_projection(tmp_path: Path) -> None:
    database, event_store = _local_events(tmp_path)
    service = _service(event_store)

    await _exercise_provider_service(service)

    raw = await event_store.query(
        EventQuery(
            scope=StorageScope(
                tenant_id="tenant-1",
                project_id="project-1",
                run_id="run-1",
            ),
            kinds=("viz.scalar",),
            order=SortDirection.ASCENDING,
        )
    )
    assert len(raw.items) == 1
    assert raw.items[0].scope.as_filter() == {
        "tenant_id": "tenant-1",
        "project_id": "project-1",
        "run_id": "run-1",
    }
    assert raw.items[0].payload["compatibility"] == {
        "app_id": {
            "value": "app-compat",
            "deprecated": True,
            "scheduled_removal": "future breaking release",
        },
        "client_id": {
            "value": "client-old",
            "deprecated": True,
            "scheduled_removal": "future breaking release",
        },
    }
    assert not {"app_id", "client_id"}.intersection(raw.items[0].scope.as_filter())

    await database.close()


@pytest.mark.asyncio
async def test_canonical_viz_service_passes_external_provider_projection(tmp_path: Path) -> None:
    provider = DeterministicExternalProvider()
    request = StorageOpenRequest(
        workspace_id="external-workspace",
        workspace_root=tmp_path.resolve(),
        owner_scope=_OWNER,
        selection=StorageProviderSelection(
            provider="test.external",
            config={
                "endpoint": "memory://external-conformance",
                "credential_ref": "secret://external-storage",
            },
        ),
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )
    bundle = provider.open(request)
    service = build_canonical_viz_service(
        bundle=bundle,
        owner_scope=request.owner_scope,
        clock=request.clock.now,
        event_id_factory=_Identities().next,
    )

    await _exercise_provider_service(service)

    assert provider.open_calls == 1
    assert tuple(tmp_path.iterdir()) == ()
    await bundle.close()


@pytest.mark.asyncio
async def test_canonical_viz_service_applies_time_and_kind_before_local_paging(
    tmp_path: Path,
) -> None:
    database, event_store = _local_events(tmp_path)
    service = _service(event_store)
    for index, kind in enumerate(("vector", "scalar", "image", "scalar"), start=1):
        event = _event(kind, step=index)
        event.created_at = (_NOW + timedelta(seconds=index)).isoformat()
        await service.append(event)

    page = await service.query_run_page(
        "run-1",
        kinds=("scalar",),
        since=_NOW + timedelta(seconds=3),
        page=PageRequest(limit=1),
    )

    assert [(event.viz_kind, event.step) for event in page.items] == [("scalar", 4)]
    assert page.next_cursor is None
    await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: replace(event, run_id=" run-1"),
        lambda event: replace(event, step=True),
        lambda event: replace(event, mode="unknown"),
        lambda event: replace(event, value=None),
        lambda event: replace(event, vector=[1.0]),
        lambda event: replace(event, app_id=" "),
    ],
)
async def test_canonical_viz_service_rejects_inexact_or_conflicting_inputs(
    mutate: Callable[[VizEvent], VizEvent],
) -> None:
    service = _service(SimpleNamespace(append=lambda _event: None))

    with pytest.raises((TypeError, StorageConfigurationError)):
        await service.append(mutate(_event("scalar")))


@pytest.mark.asyncio
async def test_canonical_viz_service_rejects_owner_provenance_conflicts() -> None:
    service = CanonicalVizService(
        event_store=SimpleNamespace(append=lambda _event: None),
        owner_scope=StorageScope(org_id="org-1"),
        clock=_Clock().now,
    )

    with pytest.raises(StorageConfigurationError, match="owner_scope org_id"):
        await service.append(replace(_event("scalar"), org_id="other-org"))


@pytest.mark.asyncio
async def test_canonical_viz_query_uses_promoted_scope_index(tmp_path: Path) -> None:
    database, _event_store = _local_events(tmp_path)
    connection = sqlite3.connect(tmp_path / "local" / "events.sqlite3")
    try:
        plan = connection.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM local_events "
            "WHERE stream = ? AND tenant_id IS ? AND project_id IS ? "
            "AND org_id IS ? AND user_id IS ? AND session_id IS ? AND run_id IS ? "
            "AND graph_id IS ? AND node_id IS ? AND agent_id IS ? AND scope_key IS ? "
            "AND kind IN (?, ?) ORDER BY cursor ASC LIMIT ?",
            (
                "runtime",
                "tenant-1",
                "project-1",
                None,
                None,
                None,
                "run-1",
                None,
                None,
                None,
                None,
                "viz.scalar",
                "viz.image",
                501,
            ),
        ).fetchall()
    finally:
        connection.close()

    assert any("ix_local_events_scope" in str(row) for row in plan), plan
    await database.close()


def test_canonical_viz_public_methods_follow_required_docstring_format() -> None:
    methods = (
        CanonicalVizService.__init__,
        CanonicalVizService.append,
        CanonicalVizService.query_run_page,
        CanonicalVizService.query_run,
        build_canonical_viz_service,
        VizEventSink.append,
    )
    required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
    for method in methods:
        docstring = inspect.getdoc(method)
        assert docstring is not None
        positions = tuple(docstring.index(section) for section in required)
        assert positions == tuple(sorted(positions))
        assert docstring.count("```python") >= 2


def test_viz_facade_depends_on_sink_protocol_and_keeps_atomic_writer_probe() -> None:
    assert get_type_hints(VizFacade)["viz_service"] is VizEventSink
    assert VizEvent.__dataclass_fields__["app_id"].metadata == {
        "deprecated": True,
        "role": "optional compatibility metadata",
    }
    source = inspect.getsource(VizFacade.image_from_bytes)
    assert "inspect.isawaitable(write_result)" in source

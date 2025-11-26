from datetime import datetime, timedelta, timezone

import pytest

from aethergraph.core.runtime.run_types import RunRecord, RunStatus
from aethergraph.storage.runs.inmen_store import InMemoryRunStore


@pytest.mark.asyncio
async def test_run_store_create_and_get():
    store = InMemoryRunStore()

    now = datetime.now(tz=timezone.utc)
    rec = RunRecord(
        run_id="run-1",
        graph_id="graph-a",
        kind="taskgraph",
        status=RunStatus.running,
        started_at=now,
        tags=["tag1"],
        user_id="user-1",
        org_id="org-1",
    )

    await store.create(rec)

    loaded = await store.get("run-1")
    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert loaded.graph_id == "graph-a"
    assert loaded.status == RunStatus.running
    assert loaded.tags == ["tag1"]
    assert loaded.user_id == "user-1"
    assert loaded.org_id == "org-1"

    # ensure it's a copy (mutations don't leak back)
    loaded.tags.append("tag2")
    again = await store.get("run-1")
    assert again.tags == ["tag1"]


@pytest.mark.asyncio
async def test_run_store_update_status_and_error():
    store = InMemoryRunStore()

    now = datetime.now(tz=timezone.utc)
    rec = RunRecord(
        run_id="run-2",
        graph_id="graph-b",
        kind="graphfn",
        status=RunStatus.running,
        started_at=now,
    )
    await store.create(rec)

    finished = now + timedelta(seconds=5)
    await store.update_status(
        "run-2",
        RunStatus.failed,
        finished_at=finished,
        error="boom",
    )

    loaded = await store.get("run-2")
    assert loaded is not None
    assert loaded.status == RunStatus.failed
    assert loaded.finished_at == finished
    assert loaded.error == "boom"


@pytest.mark.asyncio
async def test_run_store_list_filters_and_sorts():
    store = InMemoryRunStore()

    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rec1 = RunRecord(
        run_id="run-1",
        graph_id="graph-a",
        kind="taskgraph",
        status=RunStatus.succeeded,
        started_at=base,
    )
    rec2 = RunRecord(
        run_id="run-2",
        graph_id="graph-a",
        kind="taskgraph",
        status=RunStatus.failed,
        started_at=base + timedelta(seconds=10),
    )
    rec3 = RunRecord(
        run_id="run-3",
        graph_id="graph-b",
        kind="graphfn",
        status=RunStatus.succeeded,
        started_at=base + timedelta(seconds=20),
    )

    await store.create(rec1)
    await store.create(rec2)
    await store.create(rec3)

    # list all
    all_runs = await store.list()
    assert [r.run_id for r in all_runs] == ["run-3", "run-2", "run-1"]

    # filter by graph_id
    a_runs = await store.list(graph_id="graph-a")
    assert {r.run_id for r in a_runs} == {"run-1", "run-2"}

    # filter by status
    succ = await store.list(status=RunStatus.succeeded)
    assert {r.run_id for r in succ} == {"run-1", "run-3"}

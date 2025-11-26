import pytest

from aethergraph.contracts.errors.errors import GraphHasPendingWaits
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.storage.runs.inmen_store import InMemoryRunStore


@pytest.mark.asyncio
async def test_run_manager_success(monkeypatch):
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    rm = RunManager(run_store=store, registry=reg)

    # Patch _resolve_target so we don't depend on real graphs
    async def fake_resolve(self, graph_id: str):
        return object()  # dummy

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    # Patch run_or_resume_async to simulate a successful run
    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        assert inputs == {"x": 1}
        assert run_id is not None
        return {"out": 42}

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )

    record, outputs, has_waits, continuations = await rm.start_run(
        graph_id="my-graph",
        inputs={"x": 1},
        run_id=None,
        tags=["t1"],
        user_id="u1",
        org_id="o1",
    )

    assert record.graph_id == "my-graph"
    assert record.status == RunStatus.succeeded
    assert outputs == {"out": 42}
    assert has_waits is False
    assert continuations == []

    # Check store
    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.succeeded
    assert loaded.user_id == "u1"
    assert loaded.org_id == "o1"
    assert loaded.tags == ["t1"]


@pytest.mark.asyncio
async def test_run_manager_waits(monkeypatch):
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    rm = RunManager(run_store=store, registry=reg)

    async def fake_resolve(self, graph_id: str):
        return object()

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        # Simulate GraphHasPendingWaits
        raise GraphHasPendingWaits(
            "waiting",
            waiting_nodes=["node1"],
            continuations=[{"node_id": "node1", "kind": "text"}],
        )

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )

    record, outputs, has_waits, continuations = await rm.start_run(
        graph_id="my-graph",
        inputs={"x": 1},
    )

    assert record.status == RunStatus.running
    assert outputs is None
    assert has_waits is True
    assert continuations == [{"node_id": "node1", "kind": "text"}]

    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.running


@pytest.mark.asyncio
async def test_run_manager_failure(monkeypatch):
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    rm = RunManager(run_store=store, registry=reg)

    async def fake_resolve(self, graph_id: str):
        return object()

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )

    record, outputs, has_waits, continuations = await rm.start_run(
        graph_id="my-graph",
        inputs={"x": 1},
    )

    assert record.status == RunStatus.failed
    assert outputs is None
    assert has_waits is False
    assert continuations == []
    assert record.error == "boom"

    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.failed
    assert loaded.error == "boom"

import asyncio

import pytest

from aethergraph.contracts.errors.errors import GraphHasPendingWaits
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.storage.runs.inmen_store import InMemoryRunStore


class Identity:
    # Mock identity object - the real is under aethergraph.api.v1.deps RequestIdentity
    def __init__(self, user_id: str, org_id: str):
        self.user_id = user_id
        self.org_id = org_id


@pytest.fixture
def dummy_meter(monkeypatch):
    """
    Patch current_metering() used inside RunManager._run_and_finalize()
    so tests don't depend on real metering services.
    """

    class DummyMeter:
        def __init__(self):
            self.calls: list[dict] = []

        async def record_run(self, **kwargs):
            self.calls.append(kwargs)

    meter = DummyMeter()

    def fake_current_metering():
        return meter

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.current_metering",
        fake_current_metering,
    )
    return meter


@pytest.mark.asyncio
async def test_run_manager_start_run_success(monkeypatch, dummy_meter):
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    rm = RunManager(run_store=store, registry=reg)

    # Patch _resolve_target so we don't depend on real graphs
    async def fake_resolve(self, graph_id: str):
        return object()  # dummy target

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

    identity = Identity(user_id="u1", org_id="o1")
    record, outputs, has_waits, continuations = await rm.start_run(
        graph_id="my-graph",
        inputs={"x": 1},
        run_id=None,
        tags=["t1"],
        identity=identity,
    )

    # RunRecord returned
    assert record.graph_id == "my-graph"
    assert record.status == RunStatus.succeeded
    assert outputs == {"out": 42}
    assert has_waits is False
    assert continuations == []

    # Check store persisted the final state
    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.succeeded
    assert loaded.user_id == "u1"
    assert loaded.org_id == "o1"
    assert "t1" in loaded.tags  # tag preserved, other tags auto-added

    # Metering should have been called once with status "succeeded"
    assert len(dummy_meter.calls) == 1
    call = dummy_meter.calls[0]
    assert call["run_id"] == record.run_id
    assert call["graph_id"] == "my-graph"
    assert call["status"] == "succeeded"


@pytest.mark.asyncio
async def test_run_manager_start_run_waits(monkeypatch, dummy_meter):
    """
    GraphHasPendingWaits currently results in:
      - record.status == RunStatus.failed
      - has_waits = True
      - continuations = e.continuations
      - outputs is None
    """
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

    assert record.graph_id == "my-graph"
    # With the current implementation, we mark this as failed (for now).
    assert record.status == RunStatus.failed
    assert outputs is None
    assert has_waits is True
    assert continuations == [{"node_id": "node1", "kind": "text"}]

    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.failed

    # Metering should be called once with status "waiting"
    assert len(dummy_meter.calls) == 1
    call = dummy_meter.calls[0]
    assert call["status"] == "waiting"


@pytest.mark.asyncio
async def test_run_manager_start_run_failure(monkeypatch, dummy_meter):
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

    assert record.graph_id == "my-graph"
    assert record.status == RunStatus.failed
    assert outputs is None
    assert has_waits is False
    assert continuations == []
    assert record.error == "boom"

    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.failed
    assert loaded.error == "boom"

    # Metering should be called once with status "failed"
    assert len(dummy_meter.calls) == 1
    call = dummy_meter.calls[0]
    assert call["status"] == "failed"


@pytest.mark.asyncio
async def test_run_manager_submit_run_non_blocking(monkeypatch, dummy_meter):
    """
    submit_run should:
      - Create a RunRecord with status=running.
      - Schedule background execution.
      - Eventually update the record to succeeded in the store.
    """
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    rm = RunManager(run_store=store, registry=reg, max_concurrent_runs=10)

    # Dummy target
    async def fake_resolve(self, graph_id: str):
        return object()

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    # Simulate a small async delay to force background behavior
    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        await asyncio.sleep(0.01)
        return {"out": 123}

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )

    # Call submit_run: this should return quickly with a RunRecord
    identity = Identity(user_id="u1", org_id="o1")
    record = await rm.submit_run(
        graph_id="my-graph",
        inputs={"x": 2},
        identity=identity,
        tags=["t1"],
    )

    assert record.graph_id == "my-graph"
    # At this instant, it might still be running or already succeeded depending on timing.
    # We only require that the record exists in the store and eventually becomes succeeded.
    loaded_initial = await store.get(record.run_id)
    assert loaded_initial is not None
    assert loaded_initial.graph_id == "my-graph"

    # Give the background task time to run to completion
    await asyncio.sleep(0.05)

    loaded_final = await store.get(record.run_id)
    assert loaded_final is not None
    assert loaded_final.status == RunStatus.succeeded

    # Metering should have been called once with status "succeeded"
    assert len(dummy_meter.calls) == 1
    call = dummy_meter.calls[0]
    assert call["run_id"] == record.run_id
    assert call["status"] == "succeeded"

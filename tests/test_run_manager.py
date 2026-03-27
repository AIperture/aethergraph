import asyncio

import pytest

from aethergraph.contracts.errors.errors import GraphBuildError, GraphHasPendingWaits
from aethergraph.core.runtime.run_cancellation import RunCancellationRegistry
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.services.registry.unified_registry import UnifiedRegistry
from aethergraph.services.runner.facade import RunFacade
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
    GraphHasPendingWaits results in:
      - record.status == RunStatus.waiting
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
    assert record.status == RunStatus.waiting
    assert outputs is None
    assert has_waits is True
    assert continuations == [{"node_id": "node1", "kind": "text"}]

    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.waiting

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
        # agent_id="agent_123",
    )

    assert record.graph_id == "my-graph"
    assert record.status == RunStatus.failed
    assert outputs is None
    assert has_waits is False
    assert continuations == []
    assert record.error == "boom"
    assert record.meta.get("error_kind") == "runtime"
    assert record.meta.get("error_code") is None
    assert record.meta.get("error_stage") is None
    assert record.meta.get("error_hints") == []
    assert record.meta.get("error_message") == "boom"

    loaded = await store.get(record.run_id)
    assert loaded is not None
    assert loaded.status == RunStatus.failed
    assert loaded.error == "boom"

    # Metering should be called once with status "failed"
    assert len(dummy_meter.calls) == 1
    call = dummy_meter.calls[0]
    assert call["status"] == "failed"


@pytest.mark.asyncio
async def test_run_manager_start_run_build_failure(monkeypatch, dummy_meter):
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
        raise GraphBuildError(
            "build failed",
            code="graph_inputs_missing_required",
            stage="input_bind",
            hints=[{"code": "provide_required_inputs", "message": "Provide x"}],
        )

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
    assert record.meta.get("error_kind") == "build"
    assert record.meta.get("error_code") == "graph_inputs_missing_required"
    assert record.meta.get("error_stage") == "input_bind"
    assert record.meta.get("error_message", "").endswith("build failed")
    assert record.meta.get("error_hints") == [
        {"code": "provide_required_inputs", "message": "Provide x"}
    ]


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


@pytest.mark.asyncio
async def test_run_manager_submit_run_persists_launch_metadata_and_run_config(
    monkeypatch, dummy_meter
):
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    rm = RunManager(run_store=store, registry=reg, max_concurrent_runs=10)

    async def fake_resolve(self, graph_id: str):
        return object()

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    captured_kwargs = {}

    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        captured_kwargs.update(kwargs)
        return {"out": 7}

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )

    record, outputs, has_waits, continuations = await rm.run_and_wait(
        graph_id="my-graph",
        inputs={"x": 2},
        identity=Identity(user_id="u1", org_id="o1"),
        tags=["app:demo"],
        session_id="sess-1",
        app_id="app-1",
        app_name="Demo App",
        agent_id="agent-1",
        run_config={"resume_from_run_id": "run-old", "resume_mode": "failed_nodes"},
    )

    assert outputs == {"out": 7}
    assert has_waits is False
    assert continuations == []

    persisted = await store.get(record.run_id)
    assert persisted is not None
    assert persisted.meta["original_inputs"] == {"x": 2}
    assert persisted.meta["original_run_config"] == {
        "resume_from_run_id": "run-old",
        "resume_mode": "failed_nodes",
    }
    assert persisted.meta["original_tags"] == ["app:demo"]
    assert persisted.meta["original_session_id"] == "sess-1"
    assert persisted.meta["original_app_id"] == "app-1"
    assert persisted.meta["app_name"] == "Demo App"
    assert persisted.meta["agent_id"] == "agent-1"
    assert persisted.meta["resume_from_run_id"] == "run-old"
    assert persisted.meta["resume_mode"] == "failed_nodes"
    assert captured_kwargs["resume_from_run_id"] == "run-old"
    assert captured_kwargs["resume_mode"] == "failed_nodes"


@pytest.mark.asyncio
async def test_run_manager_cancel_run_stays_cancellation_requested_until_worker_exits(
    monkeypatch, dummy_meter
):
    store = InMemoryRunStore()
    reg = UnifiedRegistry()
    cancel_registry = RunCancellationRegistry()
    rm = RunManager(
        run_store=store,
        registry=reg,
        cancellation_registry=cancel_registry,
        max_concurrent_runs=10,
    )

    async def fake_resolve(self, graph_id: str):
        return object()

    monkeypatch.setattr(
        "aethergraph.core.runtime.run_manager.RunManager._resolve_target",
        fake_resolve,
    )

    async def fake_run_or_resume_async(target, inputs, run_id=None, **kwargs):
        del target, inputs, kwargs
        handle = await cancel_registry.get(run_id)
        assert handle is not None
        while not handle.is_cancel_requested():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.03)
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        "aethergraph.core.runtime.graph_runner.run_or_resume_async",
        fake_run_or_resume_async,
    )

    record = await rm.submit_run(
        graph_id="my-graph",
        inputs={"x": 1},
        identity=Identity(user_id="u1", org_id="o1"),
    )
    await asyncio.sleep(0.01)

    await rm.cancel_run(record.run_id)
    interim = await store.get(record.run_id)
    assert interim is not None
    assert interim.status == RunStatus.cancellation_requested

    await asyncio.sleep(0.08)
    final = await store.get(record.run_id)
    assert final is not None
    assert final.status == RunStatus.canceled
    assert final.meta["cancel_reason"] == "user_requested"
    assert final.meta["cancel_backend_kind"] is not None


@pytest.mark.asyncio
async def test_run_facade_bound_cancellation_helpers(monkeypatch):
    cancel_registry = RunCancellationRegistry()

    class FakeRunManager:
        async def cancel_run(self, run_id: str) -> None:
            handle = await cancel_registry.create(run_id)
            await handle.request_cancel()

    monkeypatch.setattr(
        "aethergraph.services.runner.facade.get_run_cancellation_registry",
        lambda: cancel_registry,
    )

    facade = RunFacade(run_manager=FakeRunManager(), current_run_id="run-abc")
    event = await facade.thread_cancel_event()
    assert event.is_set() is False
    assert await facade.is_cancel_requested() is False

    handle = await cancel_registry.create("run-abc")
    await handle.request_cancel()

    assert await facade.is_cancel_requested() is True
    with pytest.raises(RuntimeError, match="cancellation requested"):
        await facade.raise_if_cancel_requested()

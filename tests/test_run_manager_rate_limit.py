import asyncio

from fastapi import HTTPException
import pytest

from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import RunRecord, RunStatus


class Identity:
    # Mock identity object - the real is under aethergraph.api.v1.deps RequestIdentity
    def __init__(self, user_id: str, org_id: str):
        self.user_id = user_id
        self.org_id = org_id


class DummyRunStore:
    async def create(self, record: RunRecord) -> None:
        # minimal stub
        self._last_created = record

    async def update_status(self, *args, **kwargs) -> None:
        # minimal stub
        pass


@pytest.mark.asyncio
async def test_run_manager_max_concurrent_runs_blocks_extra(monkeypatch):
    """
    When max_concurrent_runs=1:
      - The first submit_run acquires the slot and keeps it busy.
      - A second concurrent submit_run should raise HTTPException(429).
    """
    store = DummyRunStore()
    rm = RunManager(
        run_store=store,
        registry=None,
        sched_registry=None,
        max_concurrent_runs=1,
    )

    # Make _resolve_target return a dummy object so we don't depend on registry.
    class DummyTarget:
        pass

    async def fake_resolve_target(self, graph_id: str):
        return DummyTarget()

    monkeypatch.setattr(
        RunManager,
        "_resolve_target",
        fake_resolve_target,
    )

    # Patch _run_and_finalize so the first run "blocks" until we release an event.
    block_event = asyncio.Event()

    async def fake_run_and_finalize(
        self,
        *,
        record,
        target,
        graph_id,
        inputs,
        identity,
    ):
        # Keep the slot occupied until the test unblocks it
        await block_event.wait()
        record.status = RunStatus.succeeded
        return record, {}, False, []

    monkeypatch.setattr(
        RunManager,
        "_run_and_finalize",
        fake_run_and_finalize,
    )

    # Start the first run: it will schedule _bg which waits on block_event

    task1 = asyncio.create_task(
        rm.submit_run(
            "graph-1",
            inputs={},
            identity=Identity(user_id="u1", org_id="o1"),
        )
    )

    # Give the first submit_run a moment to acquire the slot and start _bg
    await asyncio.sleep(0.01)

    # Now start a second run; it should hit the concurrency limit and raise HTTPException(429)
    with pytest.raises(HTTPException) as excinfo:
        await rm.submit_run(
            "graph-1",
            inputs={},
            identity=Identity(user_id="u2", org_id="o2"),
        )

    assert excinfo.value.status_code == 429

    # Unblock the first run so the background task can finish cleanly
    block_event.set()
    await task1

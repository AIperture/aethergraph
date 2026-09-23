"""Graph functions cancel through their owning task and retain cleanup ownership."""

import asyncio
from uuid import uuid4

import pytest

from aethergraph.config.config import AppSettings
from aethergraph.core.graph.graph_fn import GraphFunction
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import use_services
from aethergraph.services.container.default_container import build_default_container


@pytest.fixture
def container(tmp_path):
    return build_default_container(
        cfg=AppSettings(workspace=str(tmp_path), embed={"enabled": False}), root=str(tmp_path)
    )


def register(fn, inputs):
    graph = GraphFunction(
        name=f"test.cancel_graph_function.{uuid4().hex}", fn=fn, inputs=inputs, outputs=["ok"]
    )
    current_registry().register(nspace="graphfn", name=graph.name, version=graph.version, obj=graph)
    return graph.name


@pytest.mark.asyncio
async def test_running_graph_function_cancels_once_and_waits_for_cleanup(container):
    started = [asyncio.Event(), asyncio.Event()]
    cleanup = asyncio.Event()
    release = asyncio.Event()
    sibling_release = asyncio.Event()
    cleaned = []

    async def work(index, *, context):
        del context
        started[index].set()
        try:
            await sibling_release.wait()
        except asyncio.CancelledError:
            cleanup.set()
            await release.wait()
            cleaned.append(index)
            raise
        return {"ok": True}

    with use_services(container):
        try:
            name = register(work, ["index"])
            runs = [
                await container.run_manager.submit_run(name, inputs={"index": i}) for i in range(2)
            ]
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 5)
            await container.run_manager.cancel_run(runs[0].run_id)
            await asyncio.wait_for(cleanup.wait(), 5)
            await container.run_manager.cancel_run(runs[0].run_id)
            interim = await container.run_manager.get_record(runs[0].run_id)
            assert interim.status.value == "cancellation_requested"
            assert not cleaned
            release.set()
            ended = await asyncio.wait_for(container.run_manager.wait_run(runs[0].run_id), 5)
            assert ended.status.value == "canceled"
            assert ended.meta["cancel_backend_kind"] == "local_task"
            assert ended.meta["cancel_finalized_at"]
            assert cleaned == [0]
            assert (
                await container.run_manager.get_record(runs[1].run_id)
            ).status.value == "running"
            sibling_release.set()
            ended = await asyncio.wait_for(container.run_manager.wait_run(runs[1].run_id), 5)
            assert ended.status.value == "succeeded"
        finally:
            release.set()
            sibling_release.set()
            await container.run_manager.close()
            await container.close_storage()


@pytest.mark.asyncio
async def test_cancel_during_admission_never_enters_graph_function(container):
    entered = []

    async def work(*, context):
        del context
        entered.append(True)
        return {"ok": True}

    async def cancel_before_scheduling(record):
        await container.run_manager.cancel_run(record.run_id)

    with use_services(container):
        try:
            name = register(work, [])
            run = await container.run_manager.submit_run(
                name, inputs={}, admission_callback=cancel_before_scheduling
            )
            ended = await asyncio.wait_for(container.run_manager.wait_run(run.run_id), 5)
            assert ended.status.value == "canceled"
            assert not entered
            assert not container.run_manager._running
        finally:
            await container.run_manager.close()
            await container.close_storage()

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from aethergraph.core.runtime import runtime_services
from aethergraph.core.runtime.run_cancellation import RunCancellationRegistry
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_registration import RunRegistrationGuard
from aethergraph.core.runtime.runtime_env import RuntimeEnv
from aethergraph.services.container.default_container import DefaultContainer
from aethergraph.services.continuations.continuation import Continuation
from aethergraph.services.resume.multi_scheduler_resume_bus import (
    MultiSchedulerResumeBus,
    SchedulerUnavailableError,
)
from aethergraph.services.schedulers.registry import SchedulerRegistry


def _continuation(*, continuation_id: str = "cont-1") -> Continuation:
    return Continuation(
        continuation_id=continuation_id,
        revision=1,
        run_id="run-1",
        node_id="node-1",
        kind="approval",
    )


class _Store:
    def __init__(self) -> None:
        self.continuation = _continuation()
        self.deleted: list[tuple[str, str]] = []

    async def get_by_id(
        self, run_id: str, node_id: str, continuation_id: str
    ) -> Continuation | None:
        del run_id, node_id
        if self.continuation and self.continuation.continuation_id == continuation_id:
            return self.continuation
        return None

    async def close(self, continuation, *, status, closed_at) -> Continuation:
        self.deleted.append((continuation.run_id, continuation.node_id))
        self.continuation = replace(
            continuation,
            revision=continuation.revision + 1,
            status=status,
            closed_at=closed_at,
        )
        return self.continuation


class _Scheduler:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None,
        *,
        fail: bool = False,
    ) -> None:
        self.loop = loop
        self.fail = fail
        self.calls: list[tuple[str, str, dict[str, Any], int]] = []
        self.terminated = False

    async def on_resume_event(
        self,
        run_id: str,
        node_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.calls.append((run_id, node_id, payload, threading.get_ident()))
        if self.fail:
            raise RuntimeError("dispatch failed")

    async def terminate(self) -> None:
        self.terminated = True


@pytest.mark.asyncio
async def test_resume_bus_dispatches_once_and_deletes_after_success() -> None:
    registry = SchedulerRegistry()
    scheduler = _Scheduler(asyncio.get_running_loop())
    registry.register("run-1", scheduler)
    store = _Store()
    bus = MultiSchedulerResumeBus(registry=registry, store=store)

    await bus.enqueue_resume(
        continuation=_continuation(),
        payload={"approved": True},
    )

    assert scheduler.calls == [("run-1", "node-1", {"approved": True}, threading.get_ident())]
    assert store.deleted == [("run-1", "node-1")]

    with pytest.raises(PermissionError, match="no longer waiting"):
        await bus.enqueue_resume(
            continuation=_continuation(),
            payload={"approved": True},
        )
    assert len(scheduler.calls) == 1


@pytest.mark.asyncio
async def test_resume_bus_cross_thread_dispatches_on_scheduler_loop() -> None:
    scheduler_loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run_loop() -> None:
        asyncio.set_event_loop(scheduler_loop)
        started.set()
        scheduler_loop.run_forever()

    thread = threading.Thread(target=_run_loop)
    thread.start()
    assert started.wait(timeout=2)
    try:
        registry = SchedulerRegistry()
        scheduler = _Scheduler(scheduler_loop)
        registry.register("run-1", scheduler)
        store = _Store()
        bus = MultiSchedulerResumeBus(registry=registry, store=store)

        await bus.enqueue_resume(
            continuation=_continuation(),
            payload={"answer": "yes"},
        )

        assert len(scheduler.calls) == 1
        assert scheduler.calls[0][3] == thread.ident
        assert store.deleted == [("run-1", "node-1")]
    finally:
        scheduler_loop.call_soon_threadsafe(scheduler_loop.stop)
        thread.join(timeout=2)
        scheduler_loop.close()


@pytest.mark.asyncio
async def test_resume_bus_retains_continuation_when_scheduler_is_absent() -> None:
    store = _Store()
    bus = MultiSchedulerResumeBus(registry=SchedulerRegistry(), store=store)

    with pytest.raises(SchedulerUnavailableError, match="No active scheduler"):
        await bus.enqueue_resume(
            continuation=_continuation(),
            payload={},
        )

    assert store.continuation is not None
    assert store.deleted == []


@pytest.mark.asyncio
async def test_resume_bus_retains_continuation_when_scheduler_loop_is_absent() -> None:
    registry = SchedulerRegistry()
    scheduler = _Scheduler(None)
    registry.register("run-1", scheduler)
    store = _Store()
    bus = MultiSchedulerResumeBus(registry=registry, store=store)

    with pytest.raises(SchedulerUnavailableError, match="loop is unavailable"):
        await bus.enqueue_resume(
            continuation=_continuation(),
            payload={},
        )

    assert scheduler.calls == []
    assert store.continuation is not None


@pytest.mark.asyncio
async def test_resume_bus_rejects_unknown_continuation_without_dispatch() -> None:
    registry = SchedulerRegistry()
    scheduler = _Scheduler(asyncio.get_running_loop())
    registry.register("run-1", scheduler)
    store = _Store()
    bus = MultiSchedulerResumeBus(registry=registry, store=store)

    with pytest.raises(PermissionError, match="no longer waiting"):
        await bus.enqueue_resume(
            continuation=_continuation(continuation_id="unknown"),
            payload={},
        )

    assert scheduler.calls == []
    assert store.continuation is not None


@pytest.mark.asyncio
async def test_resume_bus_retains_continuation_after_dispatch_failure() -> None:
    registry = SchedulerRegistry()
    scheduler = _Scheduler(asyncio.get_running_loop(), fail=True)
    registry.register("run-1", scheduler)
    store = _Store()
    bus = MultiSchedulerResumeBus(registry=registry, store=store)

    with pytest.raises(RuntimeError, match="dispatch failed"):
        await bus.enqueue_resume(
            continuation=_continuation(),
            payload={},
        )

    assert len(scheduler.calls) == 1
    assert store.continuation is not None
    assert store.deleted == []


def test_run_registration_guard_is_single_registry_lifetime_owner() -> None:
    registry = SchedulerRegistry()
    scheduler = _Scheduler(None)
    container = SimpleNamespace(sched_registry=registry)

    with RunRegistrationGuard(run_id="run-1", scheduler=scheduler, container=container):
        assert registry.get("run-1") is scheduler

    assert registry.get("run-1") is None


@pytest.mark.asyncio
async def test_run_manager_cancellation_uses_registered_scheduler_control() -> None:
    registry = SchedulerRegistry()
    scheduler = _Scheduler(asyncio.get_running_loop())
    registry.register("run-1", scheduler)
    manager = RunManager(
        sched_registry=registry,
        cancellation_registry=RunCancellationRegistry(),
    )

    assert await manager.cancel_run("run-1") is None

    assert scheduler.terminated is True


def test_global_scheduler_boundary_is_deleted() -> None:
    source_root = Path(__file__).parents[1] / "src" / "aethergraph"

    assert not (source_root / "core" / "execution" / "global_scheduler.py").exists()
    assert "schedulers" not in DefaultContainer.__dataclass_fields__
    assert "schedulers" not in RuntimeEnv.__dict__
    assert not hasattr(runtime_services, "ensure_global_scheduler_started")

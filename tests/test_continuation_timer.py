from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from aethergraph.core.runtime.continuation_timer import ContinuationTimerService, _fire_id
from aethergraph.observability import OperationObserver
from aethergraph.observability.models import ObservationRecord
from aethergraph.services.container.default_container import SERVICE_KEYS, DefaultContainer
from aethergraph.services.continuations.continuation import (
    Continuation,
    ContinuationDraft,
    ContinuationStatus,
)
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore
from aethergraph.services.resume.multi_scheduler_resume_bus import (
    MultiSchedulerResumeBus,
    SchedulerUnavailableError,
)
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.services.schedulers.registry import SchedulerRegistry
from aethergraph.storage.continuation_store.timer_leases import (
    SQLiteContinuationTimerLeaseStore,
)


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.value = now

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class _Router:
    def __init__(
        self,
        continuation_store: InMemoryContinuationStore,
        *,
        failures: int = 0,
        unavailable: bool = False,
    ) -> None:
        self.store = continuation_store
        self.failures = failures
        self.unavailable = unavailable
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    async def resume_continuation(
        self, continuation: Continuation, payload: dict[str, Any]
    ) -> None:
        self.calls.append(
            (continuation.run_id, continuation.node_id, continuation.continuation_id, payload)
        )
        if self.unavailable:
            raise SchedulerUnavailableError("scheduler unavailable")
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("delivery failed")
        await self.store.close(
            continuation,
            status=ContinuationStatus.RESUMED,
            closed_at=datetime.now(UTC),
        )


class _Sink:
    def __init__(self) -> None:
        self.records: list[ObservationRecord] = []

    async def append_observation(self, record: ObservationRecord, **_: Any) -> str:
        self.records.append(record)
        return record.observation_id


class _Scheduler:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def on_resume_event(
        self,
        run_id: str,
        node_id: str,
        payload: dict[str, Any],
    ) -> None:
        self.calls.append((run_id, node_id, payload))

    async def terminate(self) -> None:
        return None


async def _save_due(
    store: InMemoryContinuationStore,
    *,
    now: datetime,
    run_id: str = "run-1",
    node_id: str = "wait-1",
    poll: dict[str, Any] | None = None,
) -> Continuation:
    created = await store.create(
        ContinuationDraft(
            run_id=run_id,
            node_id=node_id,
            kind="external",
            deadline=now - timedelta(seconds=1),
            poll=poll,
            next_wakeup_at=now - timedelta(seconds=1),
        )
    )
    return created.record


def _timer(
    tmp_path: Path,
    *,
    store: InMemoryContinuationStore,
    router: _Router,
    clock: _Clock,
    worker_id: str = "worker-a",
    max_attempts: int = 5,
    observer: OperationObserver | None = None,
) -> ContinuationTimerService:
    return ContinuationTimerService(
        continuation_store=store,
        lease_store=SQLiteContinuationTimerLeaseStore(tmp_path / "timer-leases.db"),
        resume_router=router,  # type: ignore[arg-type]
        clock=clock,  # type: ignore[arg-type]
        worker_id=worker_id,
        poll_interval_s=0.01,
        lease_s=5,
        max_attempts=max_attempts,
        retry_base_s=1,
        retry_max_s=4,
        observer=observer,
    )


@pytest.mark.asyncio
async def test_deadline_timer_delivers_once_and_persists_receipt(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    continuation = await _save_due(store, now=now)
    router = _Router(store)
    sink = _Sink()
    timer = _timer(
        tmp_path,
        store=store,
        router=router,
        clock=clock,
        observer=OperationObserver(sink),
    )

    assert await timer.run_once() == 1
    assert await timer.run_once() == 0

    assert len(router.calls) == 1
    assert router.calls[0][3]["timer_kind"] == "deadline"
    assert (
        await store.get(continuation.run_id, continuation.node_id)
    ).status is ContinuationStatus.RESUMED
    fire_id = _fire_id(
        continuation_id=continuation.continuation_id,
        scheduled_for=continuation.next_wakeup_at,
    )
    receipt = await timer.lease_store.get(continuation.run_id, continuation.node_id, fire_id)
    assert receipt is not None and receipt.status == "delivered"
    assert [record.name for record in sink.records] == ["claim", "delivery"]


@pytest.mark.asyncio
async def test_timer_delivers_through_canonical_resume_router(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    continuation = await _save_due(store, now=now)
    registry = SchedulerRegistry()
    scheduler = _Scheduler(asyncio.get_running_loop())
    registry.register(continuation.run_id, scheduler)
    resume_bus = MultiSchedulerResumeBus(registry=registry, store=store)
    resume_router = ResumeRouter(store=store, runner=resume_bus)
    timer = ContinuationTimerService(
        continuation_store=store,
        lease_store=SQLiteContinuationTimerLeaseStore(tmp_path / "timer-leases.db"),
        resume_router=resume_router,
        clock=clock,  # type: ignore[arg-type]
        worker_id="worker-a",
    )

    assert await timer.run_once() == 1

    assert scheduler.calls == [
        (
            continuation.run_id,
            continuation.node_id,
            {
                "timer_fired": True,
                "timer_kind": "deadline",
                "scheduled_for": continuation.next_wakeup_at.isoformat(),
            },
        )
    ]
    assert (
        await store.get(continuation.run_id, continuation.node_id)
    ).status is ContinuationStatus.RESUMED


@pytest.mark.asyncio
async def test_poll_timer_uses_poll_payload(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    await _save_due(store, now=now, poll={"interval_sec": 30})
    router = _Router(store)
    timer = _timer(tmp_path, store=store, router=router, clock=clock)

    assert await timer.run_once() == 1

    assert router.calls[0][3] == {
        "timer_fired": True,
        "timer_kind": "poll",
        "scheduled_for": (now - timedelta(seconds=1)).isoformat(),
    }


@pytest.mark.asyncio
async def test_duplicate_workers_deliver_one_fire(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    await _save_due(store, now=now)
    router = _Router(store)
    first = _timer(
        tmp_path,
        store=store,
        router=router,
        clock=clock,
        worker_id="worker-a",
    )
    second = _timer(
        tmp_path,
        store=store,
        router=router,
        clock=clock,
        worker_id="worker-b",
    )

    counts = await asyncio.gather(first.run_once(), second.run_once())

    assert sum(counts) == 1
    assert len(router.calls) == 1


@pytest.mark.asyncio
async def test_stale_lease_is_reclaimed_after_worker_restart(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    continuation = await _save_due(store, now=now)
    router = _Router(store)
    sink = _Sink()
    lease_store = SQLiteContinuationTimerLeaseStore(tmp_path / "timer-leases.db")
    fire_id = _fire_id(
        continuation_id=continuation.continuation_id,
        scheduled_for=continuation.next_wakeup_at,
    )
    first_claim = await lease_store.claim(
        fire_id=fire_id,
        continuation_id=continuation.continuation_id,
        run_id=continuation.run_id,
        node_id=continuation.node_id,
        scheduled_for=continuation.next_wakeup_at,
        worker_id="dead-worker",
        now=now,
        lease_until=now + timedelta(seconds=5),
    )
    assert first_claim is not None
    clock.advance(6)
    restarted = _timer(
        tmp_path,
        store=store,
        router=router,
        clock=clock,
        worker_id="restarted-worker",
        observer=OperationObserver(sink),
    )

    assert await restarted.run_once() == 1

    receipt = await lease_store.get(continuation.run_id, continuation.node_id, fire_id)
    assert receipt is not None
    assert receipt.status == "delivered"
    assert receipt.attempts == 2
    assert [record.name for record in sink.records] == [
        "lease_expired",
        "claim",
        "delivery",
    ]


@pytest.mark.asyncio
async def test_retry_state_survives_timer_service_restart(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    continuation = await _save_due(store, now=now)
    failing_router = _Router(store, failures=1)
    first = _timer(tmp_path, store=store, router=failing_router, clock=clock)

    assert await first.run_once() == 1
    assert await store.get(continuation.run_id, continuation.node_id) is not None

    clock.advance(1)
    succeeding_router = _Router(store)
    restarted = _timer(
        tmp_path,
        store=store,
        router=succeeding_router,
        clock=clock,
        worker_id="worker-after-restart",
    )
    assert await restarted.run_once() == 1
    assert len(succeeding_router.calls) == 1


@pytest.mark.asyncio
async def test_absent_scheduler_retries_without_consuming_continuation(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    continuation = await _save_due(store, now=now)
    router = _Router(store, unavailable=True)
    timer = _timer(tmp_path, store=store, router=router, clock=clock)

    assert await timer.run_once() == 1

    assert await store.get(continuation.run_id, continuation.node_id) is not None
    fire_id = _fire_id(
        continuation_id=continuation.continuation_id,
        scheduled_for=continuation.next_wakeup_at,
    )
    receipt = await timer.lease_store.get(continuation.run_id, continuation.node_id, fire_id)
    assert receipt is not None
    assert receipt.status == "retry"


@pytest.mark.asyncio
async def test_retry_limit_dead_letters_and_preserves_continuation(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    continuation = await _save_due(store, now=now)
    router = _Router(store, failures=1)
    sink = _Sink()
    timer = _timer(
        tmp_path,
        store=store,
        router=router,
        clock=clock,
        max_attempts=1,
        observer=OperationObserver(sink),
    )

    assert await timer.run_once() == 1

    fire_id = _fire_id(
        continuation_id=continuation.continuation_id,
        scheduled_for=continuation.next_wakeup_at,
    )
    receipt = await timer.lease_store.get(continuation.run_id, continuation.node_id, fire_id)
    assert receipt is not None
    assert receipt.status == "dead_letter"
    assert await store.get(continuation.run_id, continuation.node_id) is not None
    assert [record.name for record in sink.records] == ["claim", "dead_letter"]


@pytest.mark.asyncio
async def test_timer_start_and_shutdown_are_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 8, 13, tzinfo=UTC)
    clock = _Clock(now)
    store = InMemoryContinuationStore(secret=b"secret")
    router = _Router(store)
    timer = _timer(tmp_path, store=store, router=router, clock=clock)

    await timer.start()
    task = timer._task
    await timer.start()
    assert timer._task is task

    await timer.stop()
    await timer.stop()
    assert timer._task is None


@pytest.mark.asyncio
async def test_timer_lease_schema_upgrade_preserves_terminal_receipts(tmp_path: Path) -> None:
    path = tmp_path / "timer-leases.db"
    now = datetime(2026, 8, 13, tzinfo=UTC)
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE continuation_timer_leases_v2 (
                fire_id TEXT PRIMARY KEY,
                continuation_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                scheduled_for REAL NOT NULL,
                worker_id TEXT,
                status TEXT NOT NULL,
                lease_until REAL,
                attempts INTEGER NOT NULL,
                next_attempt_at REAL,
                last_error TEXT,
                updated_at REAL NOT NULL
            )
            """)
        connection.execute(
            """
            INSERT INTO continuation_timer_leases_v2 (
                fire_id, continuation_id, run_id, node_id, scheduled_for,
                worker_id, status, lease_until, attempts, next_attempt_at,
                last_error, updated_at
            ) VALUES (?, ?, ?, ?, ?, NULL, 'delivered', NULL, 2, NULL, NULL, ?)
            """,
            ("fire-old", "cont-old", "run-old", "node-old", now.timestamp(), now.timestamp()),
        )

    store = SQLiteContinuationTimerLeaseStore(path)
    receipt = await store.get("run-old", "node-old", "fire-old")

    assert receipt is not None
    assert receipt.revision == 1
    assert receipt.finished_at == now


def test_legacy_wakeup_boundary_is_deleted() -> None:
    source_root = Path(__file__).parents[1] / "src" / "aethergraph"
    deleted_files = (
        source_root / "contracts" / "services" / "wakeup.py",
        source_root / "core" / "runtime" / "wakeup_watcher.py",
        source_root / "services" / "wakeup" / "memory_queue.py",
        source_root / "services" / "wakeup" / "scanner_producer.py",
        source_root / "services" / "wakeup" / "worker.py",
        source_root / "services" / "continuations" / "factory.py",
        source_root / "storage" / "continuation_store" / "fs_cont.py",
        source_root / "storage" / "continuation_store" / "inmem_cont.py",
    )

    assert all(not path.exists() for path in deleted_files)
    assert "continuation_timer" in DefaultContainer.__dataclass_fields__
    assert "wakeup_queue" not in DefaultContainer.__dataclass_fields__
    assert "continuation_timer" in SERVICE_KEYS
    assert "wakeup_queue" not in SERVICE_KEYS

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import logging
from typing import Any
import uuid

from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.observability import OperationObserver, resolve_operation_observer
from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.resume.router import ResumeRouter
from aethergraph.storage.continuation_store.timer_leases import (
    SQLiteContinuationTimerLeaseStore,
    TimerLease,
)


@dataclass(frozen=True)
class _DueContinuation:
    fire_id: str
    run_id: str
    node_id: str
    token: str
    scheduled_for: datetime
    timer_kind: str


def _as_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _fire_id(*, run_id: str, node_id: str, token: str, scheduled_for: datetime) -> str:
    raw = "\x00".join((run_id, node_id, token, scheduled_for.isoformat()))
    return f"ctf_{sha256(raw.encode('utf-8')).hexdigest()}"


class ContinuationTimerService:
    """Claim and deliver durable continuation timers through `ResumeRouter`."""

    def __init__(
        self,
        *,
        continuation_store: AsyncContinuationStore,
        lease_store: SQLiteContinuationTimerLeaseStore,
        resume_router: ResumeRouter,
        clock: SystemClock,
        worker_id: str | None = None,
        poll_interval_s: float = 1.0,
        lease_s: float = 30.0,
        batch_size: int = 50,
        max_attempts: int = 5,
        retry_base_s: float = 1.0,
        retry_max_s: float = 60.0,
        observer: OperationObserver | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.continuation_store = continuation_store
        self.lease_store = lease_store
        self.resume_router = resume_router
        self.clock = clock
        self.worker_id = worker_id or f"timer-{uuid.uuid4().hex}"
        self.poll_interval_s = max(0.01, float(poll_interval_s))
        self.lease_s = max(0.1, float(lease_s))
        self.batch_size = max(1, int(batch_size))
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_s = max(0.01, float(retry_base_s))
        self.retry_max_s = max(self.retry_base_s, float(retry_max_s))
        self.observer = observer
        self.logger = logger or logging.getLogger("aethergraph.runtime.continuation_timer")
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the owned continuation-timer background task.

        Intro:
            Creates one idempotent polling task that scans durable continuations
            and competes for SQLite leases using this service's worker identity.

        Examples:
            Start during application lifespan:
            ```python
            await timer.start()
            ```

            Call start twice safely:
            ```python
            await timer.start()
            await timer.start()
            ```

        Args:
            None.

        Returns:
            None: The background task exists before returning.

        Notes:
            Lifespan ownership must pair this method with `stop`.
        """
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        """Stop and join the owned continuation-timer task.

        Intro:
            Signals the polling loop and waits for its current bounded iteration
            to finish without abandoning a worker-owned delivery coroutine.

        Examples:
            Stop during normal shutdown:
            ```python
            await timer.stop()
            ```

            Stop an already stopped service safely:
            ```python
            await timer.stop()
            await timer.stop()
            ```

        Args:
            None.

        Returns:
            None: The background task has finished or was never started.

        Notes:
            An uncompleted durable lease remains reclaimable after lease expiry if
            the process is terminated before this method completes.
        """
        self._stop.set()
        task = self._task
        if task is None:
            return
        try:
            await task
        finally:
            self._task = None

    async def run_once(self) -> int:
        """Claim and process one bounded batch of due continuations.

        Intro:
            Scans durable waits, derives stable fire identities, atomically leases
            eligible occurrences, and delivers through the canonical `ResumeRouter`.

        Examples:
            Run one deterministic test tick:
            ```python
            processed = await timer.run_once()
            ```

            Drain multiple bounded ticks:
            ```python
            while await timer.run_once():
                pass
            ```

        Args:
            None.

        Returns:
            int: Number of timer fires claimed by this worker, including retries
            that become dead letters during the tick.

        Notes:
            Busy, delayed, delivered, and dead-lettered fires are not counted.
        """
        now = self.clock.now().astimezone(UTC)
        waits = await self.continuation_store.list_waits()
        due = self._due_continuations(waits, now=now)[: self.batch_size]
        processed = 0
        for candidate in due:
            lease = self.lease_store.claim(
                fire_id=candidate.fire_id,
                run_id=candidate.run_id,
                node_id=candidate.node_id,
                token=candidate.token,
                worker_id=self.worker_id,
                now=now,
                lease_until=now + timedelta(seconds=self.lease_s),
            )
            if lease is None:
                continue
            processed += 1
            if lease.reclaimed:
                await self._observe("lease_expired", candidate=candidate, lease=lease)
            await self._observe("claim", candidate=candidate, lease=lease)
            await self._deliver(candidate, lease=lease, now=now)
        return processed

    async def _run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception:
                self.logger.exception("Continuation timer iteration failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval_s)
            except TimeoutError:
                continue

    def _due_continuations(
        self,
        waits: list[dict[str, Any]],
        *,
        now: datetime,
    ) -> list[_DueContinuation]:
        due: list[_DueContinuation] = []
        for raw in waits:
            if not isinstance(raw, dict) or bool(raw.get("closed", False)):
                continue
            run_id = str(raw.get("run_id") or "")
            node_id = str(raw.get("node_id") or "")
            token = str(raw.get("token") or "")
            scheduled_for = _as_utc_datetime(raw.get("next_wakeup_at"))
            if not run_id or not node_id or not token or scheduled_for is None:
                continue
            if scheduled_for > now:
                continue
            timer_kind = "poll" if raw.get("poll") else "deadline"
            due.append(
                _DueContinuation(
                    fire_id=_fire_id(
                        run_id=run_id,
                        node_id=node_id,
                        token=token,
                        scheduled_for=scheduled_for,
                    ),
                    run_id=run_id,
                    node_id=node_id,
                    token=token,
                    scheduled_for=scheduled_for,
                    timer_kind=timer_kind,
                )
            )
        return sorted(due, key=lambda item: (item.scheduled_for, item.run_id, item.node_id))

    async def _deliver(
        self,
        candidate: _DueContinuation,
        *,
        lease: TimerLease,
        now: datetime,
    ) -> None:
        payload = {
            "timer_fired": True,
            "timer_kind": candidate.timer_kind,
            "scheduled_for": candidate.scheduled_for.isoformat(),
        }
        try:
            await self.resume_router.resume(
                candidate.run_id,
                candidate.node_id,
                candidate.token,
                payload,
            )
        except Exception as exc:
            dead_letter = lease.attempts >= self.max_attempts
            retry_delay = min(
                self.retry_max_s,
                self.retry_base_s * (2 ** max(0, lease.attempts - 1)),
            )
            next_attempt_at = None if dead_letter else now + timedelta(seconds=retry_delay)
            changed = self.lease_store.record_failure(
                fire_id=candidate.fire_id,
                worker_id=self.worker_id,
                now=now,
                next_attempt_at=next_attempt_at,
                error=f"{type(exc).__name__}: {exc}",
                dead_letter=dead_letter,
            )
            if not changed:
                self.logger.error(
                    "Continuation timer lost lease while recording failure for %s/%s",
                    candidate.run_id,
                    candidate.node_id,
                )
            operation = "dead_letter" if dead_letter else "retry"
            await self._observe(operation, candidate=candidate, lease=lease, error=exc)
            return

        changed = self.lease_store.complete(
            fire_id=candidate.fire_id,
            worker_id=self.worker_id,
            now=now,
        )
        if not changed:
            raise RuntimeError(
                f"Continuation timer lease ownership lost for {candidate.run_id}/{candidate.node_id}"
            )
        await self._observe("delivery", candidate=candidate, lease=lease)

    async def _observe(
        self,
        operation: str,
        *,
        candidate: _DueContinuation,
        lease: TimerLease,
        error: BaseException | None = None,
    ) -> None:
        observer = self.observer or resolve_operation_observer()
        span = await observer.start_span(
            service="continuation_timer",
            operation=operation,
            metadata={
                "fire_id": candidate.fire_id,
                "run_id": candidate.run_id,
                "node_id": candidate.node_id,
                "worker_id": self.worker_id,
                "timer_kind": candidate.timer_kind,
                "attempts": lease.attempts,
                "scheduled_for": candidate.scheduled_for.isoformat(),
            },
        )
        if error is None:
            await span.finish()
        else:
            await span.fail(error)

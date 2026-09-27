from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import math
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.services.runs import RunStore
from aethergraph.contracts.storage.trigger_store import TriggerStore
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunStatus, RunVisibility
from aethergraph.observability.canonical_service import CanonicalObservationService
from aethergraph.observability.models import ObservationRecord

from .trigger_service import _trigger_observation_scope
from .types import TriggerClaim, TriggerRecord

if TYPE_CHECKING:
    from aethergraph.core.runtime.run_manager import RunManager


_RUNNING_STATUSES = (
    RunStatus.pending,
    RunStatus.running,
    RunStatus.cancellation_requested,
)


@dataclass
class TriggerEngine:
    """Claim scheduled occurrences and submit new runs through one durable path."""

    store: TriggerStore
    run_manager: RunManager
    observation_sink: CanonicalObservationService | None = None
    run_store: RunStore | None = None
    logger: Any | None = None
    claim_limit: int = 100
    lease_seconds: float = 60.0
    worker_id: str = field(default_factory=lambda: f"trigger-worker-{uuid4().hex[:12]}")

    _stop_event: asyncio.Event | None = field(default=None, init=False, repr=False)
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def start(self, poll_interval_s: float = 5.0) -> None:
        """Start the single service-owned trigger claim task.

        Creates the stop signal before scheduling work, so immediate shutdown
        cannot lose its request while the task is starting.

        Examples:
            Start during runtime startup:
            ```python
            await engine.start()
            ```

            Reuse an already running service:
            ```python
            await engine.start(poll_interval_s=1.0)
            await engine.start(poll_interval_s=1.0)
            ```

        Args:
            poll_interval_s: Finite positive time between durable claim scans.

        Returns:
            None: One owned task exists before returning.

        Notes:
            Pair with `stop`; startup does not register or load graph definitions.
        """
        if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
            raise ValueError("Trigger polling interval must be finite and positive")
        if self._task is not None:
            if not self._task.done():
                return
            self._task.result()
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_forever(poll_interval_s))

    async def run_forever(self, poll_interval_s: float = 5.0) -> None:
        """Await the owned trigger service through the existing public loop adapter.

        Starts the same task used by runtime lifespans; cancellation of this
        legacy owning adapter requests cooperative shutdown and joins that task.

        Examples:
            Run as a standalone service:
            ```python
            await engine.run_forever()
            ```

            Use a shorter scan interval:
            ```python
            await engine.run_forever(poll_interval_s=1.0)
            ```

        Args:
            poll_interval_s: Finite positive time between durable claim scans.

        Returns:
            None: The owned loop has stopped.

        Notes:
            Embedded and server hosts use `start` and `stop` directly.
        """
        await self.start(poll_interval_s)
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            await self.stop()
            raise

    async def _run_forever(self, poll_interval_s: float) -> None:
        assert self._stop_event is not None
        started_at = datetime.now(UTC)
        first_scan = True
        if self.logger:
            self.logger.info("TriggerEngine started worker_id=%s", self.worker_id)
        while not self._stop_event.is_set():
            now = datetime.now(UTC)
            try:
                await self._process_due_triggers(
                    now,
                    skip_missed_before=started_at if first_scan else None,
                )
                first_scan = False
            except Exception:
                if self.logger:
                    self.logger.exception("Error processing triggers")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_interval_s)
            except TimeoutError:
                continue
        if self.logger:
            self.logger.info("TriggerEngine stopped worker_id=%s", self.worker_id)

    async def stop(self) -> None:
        """Stop and join the single trigger task without abandoning active claims.

        Signals the loop and gives its current claim batch up to thirty seconds
        to settle. A timeout retains task ownership and leaves shutdown incomplete.

        Examples:
            Close during runtime shutdown:
            ```python
            await engine.stop()
            ```

            Stop an already stopped service:
            ```python
            await engine.stop()
            await engine.stop()
            ```

        Args:
            None.

        Returns:
            None: The owned task has stopped or was never started.

        Notes:
            A timeout must prevent the owning host from closing storage beneath
            a claim still being processed. Calling `stop` again retries the join.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._task is not None:
            await asyncio.wait_for(asyncio.shield(self._task), timeout=30.0)
            self._task = None

    async def _process_due_triggers(
        self,
        now: datetime,
        *,
        skip_missed_before: datetime | None = None,
    ) -> None:
        claims = await self.store.claim_due(
            now,
            worker_id=self.worker_id,
            lease_until=now + timedelta(seconds=self.lease_seconds),
            limit=self.claim_limit,
            skip_missed_before=skip_missed_before,
        )
        for claim in claims:
            await self._process_claim(claim, now)

    async def fire_event(
        self,
        event_key: str,
        payload: dict[str, Any] | None = None,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> None:
        """Fire matching event triggers inside one explicit tenant scope."""
        if org_id is None and user_id is None and client_id is None:
            raise ValueError("Event firing requires an explicit tenant scope")
        now = datetime.now(UTC)
        triggers = await self.store.list_by_event_key(
            event_key, org_id=org_id, user_id=user_id, client_id=client_id
        )
        for trig in triggers:
            try:
                if await self._overlap_limit_reached(trig):
                    await self._log_trigger_fire(
                        trig, now, action="skipped_overlap", run_id=None, fire_id=None
                    )
                    continue
                inputs = dict(trig.default_inputs or {})
                if payload:
                    inputs["event"] = payload
                record = await self._submit(trig, inputs=inputs, run_id=None, fire_id=None)
                trig.last_fired_at = now
                await self.store.update(trig)
                await self._log_trigger_fire(
                    trig, now, action="fired", run_id=record.run_id, fire_id=None
                )
            except Exception:
                if self.logger:
                    self.logger.exception(
                        "Error firing trigger %s for event %s", trig.trigger_id, event_key
                    )

    async def _process_claim(self, claim: TriggerClaim, now: datetime) -> None:
        trig = claim.trigger
        try:
            if await self._overlap_limit_reached(trig):
                await self.store.skip_claim(
                    claim.fire_id,
                    worker_id=self.worker_id,
                    reason="overlap",
                    completed_at=now,
                )
                await self._log_trigger_fire(
                    trig,
                    now,
                    action="skipped_overlap",
                    run_id=None,
                    fire_id=claim.fire_id,
                )
                return

            run_id = f"trg-{claim.fire_id.removeprefix('trigfire-')}"
            existing = await self.run_store.get(run_id) if self.run_store else None
            if existing is None:
                record = await self._submit(
                    trig,
                    inputs=dict(trig.default_inputs or {}),
                    run_id=run_id,
                    fire_id=claim.fire_id,
                )
            else:
                record = existing
            completed = await self.store.complete_claim(
                claim.fire_id,
                worker_id=self.worker_id,
                run_id=record.run_id,
                completed_at=now,
            )
            if not completed:
                raise RuntimeError(f"Trigger claim ownership lost: {claim.fire_id}")
            await self._log_trigger_fire(
                trig,
                now,
                action="fired" if existing is None else "deduplicated",
                run_id=record.run_id,
                fire_id=claim.fire_id,
            )
        except Exception as exc:
            retry_delay = min(300.0, float(2 ** min(claim.attempts, 8)))
            await self.store.fail_claim(
                claim.fire_id,
                worker_id=self.worker_id,
                error=str(exc),
                retry_at=now + timedelta(seconds=retry_delay),
            )
            if self.logger:
                self.logger.exception(
                    "Error processing trigger claim %s for %s", claim.fire_id, trig.trigger_id
                )

    async def _submit(
        self,
        trig: TriggerRecord,
        *,
        inputs: dict[str, Any],
        run_id: str | None,
        fire_id: str | None,
    ) -> Any:
        identity = RequestIdentity(
            user_id=trig.user_id,
            org_id=trig.org_id,
            mode=trig.mode or "local",
            client_id=trig.client_id,
        )
        tags = [f"trigger:{trig.trigger_id}"]
        if fire_id is not None:
            tags.append(f"trigger-fire:{fire_id}")
        run_config = {}
        if trig.origin_binding is not None:
            run_config["origin_binding"] = trig.origin_binding.model_dump(mode="json")
        if trig.parent_run is not None:
            run_config["parent_run"] = asdict(trig.parent_run)
        return await self.run_manager.submit_run(
            graph_id=trig.graph_id,
            inputs=inputs,
            run_id=run_id,
            session_id=trig.session_id,
            identity=identity,
            origin=RunOrigin.schedule,
            run_config=run_config or None,
            visibility=RunVisibility.normal,
            importance=RunImportance.normal,
            agent_id=trig.agent_id,
            app_id=trig.app_id,
            tags=tags,
        )

    async def _overlap_limit_reached(self, trig: TriggerRecord) -> bool:
        if trig.max_overlap_runs is None:
            return False
        if self.run_store is None:
            raise RuntimeError(
                f"Trigger {trig.trigger_id} declares max_overlap_runs without a run store"
            )
        running = await self._count_running_for_trigger(trig.trigger_id)
        return running >= trig.max_overlap_runs

    async def _count_running_for_trigger(self, trigger_id: str) -> int:
        if self.run_store is None:
            raise RuntimeError("A run store is required for trigger overlap counting")
        tag = f"trigger:{trigger_id}"
        count = 0
        page_size = 500
        for status in _RUNNING_STATUSES:
            offset = 0
            while True:
                records = await self.run_store.list(
                    graph_id=None,
                    status=status,
                    limit=page_size,
                    offset=offset,
                )
                count += sum(tag in (record.tags or []) for record in records)
                if len(records) < page_size:
                    break
                offset += len(records)
        return count

    async def _log_trigger_fire(
        self,
        trig: TriggerRecord,
        now: datetime,
        *,
        action: str,
        run_id: str | None,
        fire_id: str | None,
    ) -> None:
        if self.observation_sink is None:
            return
        try:
            await self.observation_sink.append_observation(
                ObservationRecord(
                    observation_id=f"trig-fire-{uuid4().hex[:12]}",
                    occurred_at=now.isoformat(),
                    category="trigger_fire",
                    name=action,
                    summary=f"Trigger fire {action}",
                    scope=_trigger_observation_scope(trig, run_id=run_id),
                    attributes={
                        "action": action,
                        "trigger_id": trig.trigger_id,
                        "fire_id": fire_id,
                        "kind": trig.kind,
                        "graph_id": trig.graph_id,
                        "run_id": run_id,
                        "meta": trig.meta or {},
                    },
                )
            )
        except Exception:
            if self.logger:
                self.logger.exception("Failed to log trigger fire for %s", trig.trigger_id)

# aethergraph/services/triggers/engine.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.services.runs import RunStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunStatus, RunVisibility
from aethergraph.storage.triggers.trigger_docstore import TriggerStore

from .types import TriggerRecord

if TYPE_CHECKING:
    from aethergraph.core.runtime.run_manager import RunManager


@dataclass
class TriggerEngine:
    """
    Background engine that:

    - Periodically scans for time-based triggers that are due and fires them.
    - Allows explicit event-based firing via fire_event(event_key, payload).

    It does NOT manage graph resumption; it only *starts new runs*.
    """

    store: TriggerStore
    run_manager: RunManager
    event_log: EventLog | None = None
    run_store: RunStore | None = None  # optional, for overlap checks
    logger: Any | None = None

    _stop_event: asyncio.Event = asyncio.Event()  # for graceful shutdown

    async def run_forever(self, poll_interval_s: float = 5.0) -> None:
        """
        Main loop for time-based triggers. Call this from app startup
        in an asyncio.create_task.
        """
        self._stop_event.clear()
        if self.logger:
            self.logger.info("TriggerEngine started")
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            try:
                await self._process_due_triggers(now)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing triggers: {e}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_interval_s)
            except asyncio.TimeoutError:
                # timeout is expected; it just means it's time for the next poll
                continue

        if self.logger:
            self.logger.info("TriggerEngine stopped")

    async def stop(self) -> None:
        self._stop_event.set()

    # --------- main logic for time-based triggers ---------
    async def _process_due_triggers(self, now: datetime) -> None:
        due = await self.store.list_due(now)
        for trig in due:
            try:
                await self._fire_trigger(trig, now, extra_inputs=None)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error firing trigger {trig.trigger_id}: {e}")

    async def fire_event(
        self,
        event_key: str,
        payload: dict[str, Any] | None = None,
        *,
        # optional tenant scoping for event-based triggers; if provided, only triggers matching these will be fired
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> None:
        """
        Fire all active event-based triggers for this event_key.

        payload, if provided, will be merged into default_inputs under key 'event'
        (we can change this merging policy).
        """
        now = datetime.now(timezone.utc)
        triggers = await self.store.list_by_event_key(
            event_key, org_id=org_id, user_id=user_id, client_id=client_id
        )

        for trig in triggers:
            try:
                extra_inputs = {"event": payload} if payload else None
                await self._fire_trigger(trig, now, extra_inputs=extra_inputs)
            except Exception as e:
                if self.logger:
                    self.logger.error(
                        f"Error firing trigger {trig.trigger_id} for event {event_key}: {e}"
                    )

    async def _fire_trigger(
        self,
        trig: TriggerRecord,
        now: datetime,
        extra_inputs: dict[str, Any] | None = None,
    ) -> None:
        if not trig.active:
            return

        # Overlap policy check
        if trig.max_overlap_runs is not None:
            running = await self._count_running_for_trigger(trig.trigger_id)
            if running > trig.max_overlap_runs:
                await self._log_trigger_file(trig, now, action="skipped_overlap", run_id=None)
                # still bump next_fire_at so we don't spin
                await self._update_next_fire(trig, now)

                return

        # build inputs
        identity = RequestIdentity(
            user_id=trig.user_id,
            org_id=trig.org_id,
            mode=trig.mode,
            client_id=trig.client_id,
        )
        inputs = dict(trig.default_inputs or {})  # shallow copy
        if extra_inputs:
            inputs.update(extra_inputs)

        record = await self.run_manager.submit_run(
            graph_id=trig.graph_id,
            inputs=inputs,
            session_id=trig.session_id,
            identity=identity,
            origin=RunOrigin.schedule,
            visibility=RunVisibility.normal,
            importance=RunImportance.normal,
            agent_id=trig.agent_id,
            app_id=trig.app_id,
            tags=[f"trigger:{trig.trigger_id}"],
        )

        # Update trigger and log
        trig.last_fired_at = now
        await self._update_next_fire(trig, now)
        await self._log_trigger_fire(trig, now, action="fired", run_id=record.run_id)

    async def _update_next_fire(self, trig: TriggerRecord, now: datetime) -> None:
        """
        Compute and persist next_fire_at after a fire or skip. For event triggers,
        this is always None.
        """
        from aethergraph.services.triggers.trigger_service import (
            _cron_next,  # avoid circular import
        )

        if not trig.active or trig.kind == "event":
            trig.next_fire_at = None
        elif trig.kind == "one_shot":
            # one_shot triggers only fire once, so we deactivate them after firing
            trig.active = False
            trig.next_fire_at = None
        elif trig.kind == "interval":
            if trig.interval_seconds is None:
                trig.next_fire_at = None
            else:
                trig.next_fire_at = now + timedelta(seconds=trig.interval_seconds)
        elif trig.kind == "cron":
            if not trig.cron_expr:
                trig.next_fire_at = None
            else:
                trig.next_fire_at = _cron_next(trig.cron_expr, now)
        else:
            # unknown kind; play it safe and don't schedule next fire
            trig.next_fire_at = None

        await self.store.update(trig)

    async def _log_trigger_fire(
        self,
        trig: TriggerRecord,
        now: datetime,
        action: str,
        run_id: str | None,
    ) -> None:
        if not self.event_log:
            return

        try:
            await self.event_log.append(
                {
                    "id": f"trig-fire-{trig.trigger_id}-{now.timestamp()}",
                    "ts": now.timestamp(),
                    "scope_id": trig.trigger_id,
                    "kind": "trigger_fire",
                    "payload": {
                        "action": action,
                        "trigger_id": trig.trigger_id,
                        "kind": trig.kind,
                        "graph_id": trig.graph_id,
                        "run_id": run_id,
                        "meta": trig.meta or {},
                    },
                }
            )
        except Exception:  # noqa: BLE001
            if self.logger:
                self.logger.error(f"Failed to log trigger fire event for trigger {trig.trigger_id}")

    async def _count_running_for_trigger(self, trigger_id: str) -> int:
        """
        Count non-terminal runs tagged with this trigger.

        Requires a RunStore that can list by tag. If unavailable, returns 0.
        """
        if not self.run_store:
            return 0

        try:
            records = await self.run_store.list(
                graph_id=None,
                status=None,
                user_id=None,
                org_id=None,
                session_id=None,
                limit=1000,  # arbitrary large limit; we just want to check if there are any, not paginate
                offset=0,
            )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error listing runs for overlap check: {e}")
            return 0

        running_statuses = {RunStatus.pending, RunStatus.running, RunStatus.cancellation_requested}
        count = 0
        for r in records:
            if r.status not in running_statuses:
                continue
            tags = r.tags or []
            if f"trigger:{trigger_id}" in tags:
                count += 1
        return count

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from logging import getLogger
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from croniter import croniter  # type: ignore[import]

from aethergraph.contracts.services.trigger import TriggerKind, TriggerService
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.scope.scope import Scope
from aethergraph.services.triggers.types import TriggerRecord
from aethergraph.storage.triggers.trigger_docstore import TriggerStore


# ------------ cron helper -----------
def _cron_next(cron_expr: str, now: datetime) -> datetime | None:
    """
    Compute the next fire time for a cron expression using croniter.
    """

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    itr = croniter(cron_expr, now)
    next_time = itr.get_next(datetime)
    return next_time


@dataclass
class TriggerServiceImpl(TriggerService):
    store: TriggerStore
    event_log: EventLog | None = None
    logger: Any | None = None

    async def create_from_scope(
        self,
        *,
        scope: Scope,
        graph_id: str,
        default_inputs: dict[str, Any],
        kind: TriggerKind,
        cron_expr: str | None = None,
        interval_seconds: int | None = None,
        run_at: datetime | None = None,
        event_key: str | None = None,
        tz: str | None = None,
        max_overlap_runs: int | None = None,
        catch_up_missed: bool = False,
        origin: str = "schedule",
        trigger_name: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> TriggerRecord:
        trigger_id = f"trig-{uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)

        trig = TriggerRecord.from_scope(
            trigger_id=trigger_id,
            scope=scope,
            graph_id=graph_id,
            default_inputs=default_inputs,
            kind=kind,
            origin=origin,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            run_at=run_at,
            event_key=event_key,
            tz=tz,
            max_overlap_runs=max_overlap_runs,
            catch_up_missed=catch_up_missed,
            meta=meta,
            trigger_name=trigger_name,
        )

        trig.next_fire_at = self._initial_next_fire_at(trig, now)

        await self.store.create(trig)
        await self._log_trigger_event(trig, action="created")
        return trig

    async def cancel(self, trigger_id: str) -> None:
        trig: TriggerRecord | None = await self.store.get(trigger_id)
        if not trig:
            return
        trig.active = False
        trig.next_fire_at = None
        await self.store.update(trig)
        await self._log_trigger_event(trig, action="canceled")

    async def delete(self, trigger_id: str) -> None:
        trig: TriggerRecord | None = await self.store.get(trigger_id)
        if not trig:
            return
        await self.store.delete(trigger_id)
        await self._log_trigger_event(trig, action="deleted")

    async def get(self, trigger_id: str) -> TriggerRecord | None:
        return await self.store.get(trigger_id)

    async def list_for_owner(
        self, *, org_id: str | None, user_id: str | None
    ) -> list[TriggerRecord]:
        all_trigs = await self.store.list_all(org_id=org_id, user_id=user_id)
        return all_trigs

    # ------------ helpers -----------

    def _initial_next_fire_at(self, trig: TriggerRecord, now: datetime) -> datetime | None:
        """
        Compute initial next_fire_at for a brand new trigger.

        For event triggers, this is always None (event-driven).
        For one_shot, we respect trig.run_at.
        For cron/interval, we compute from 'now'.
        """
        tz = ZoneInfo(trig.tz or "UTC")  # default to UTC if not specified
        now_local = now.astimezone(tz)

        if not trig.active:
            return None

        if trig.kind == "event":
            return None

        if trig.kind == "one_shot":
            if not trig.run_at:
                return None

            run_at = trig.run_at
            # Attach or convert to the trigger's timezone if naïve
            tz = ZoneInfo(trig.tz or "UTC")
            run_at = run_at.astimezone(tz) if run_at.tzinfo else run_at.replace(tzinfo=tz)

            run_at_utc = run_at.astimezone(timezone.utc)
            return run_at_utc if run_at_utc > now else None

        if trig.kind == "interval":
            if trig.interval_seconds is None:
                return None
            return now + timedelta(seconds=trig.interval_seconds)

        if trig.kind == "cron":
            if not trig.cron_expr:
                return None
            local_next = _cron_next(trig.cron_expr, now_local)
            return local_next.astimezone(timezone.utc) if local_next else None

        return None

    async def _log_trigger_event(self, trig: TriggerRecord, action: str) -> None:
        if not self.event_log:
            return

        try:
            evt = {
                "id": f"trig-evt-{uuid4().hex[:8]}",
                "ts": datetime.now(timezone.utc).timestamp(),
                "scope_id": trig.trigger_id,
                "kind": "trigger",
                "payload": {
                    "action": action,
                    "trigger_id": trig.trigger_id,
                    "kind": trig.kind,
                    "graph_id": trig.graph_id,
                    "meta": trig.meta or {},
                },
            }
            await self.event_log.append(evt)
        except Exception:
            if self.logger:
                self.logger.exception("Failed to log trigger event for %s", trig.trigger_id)
            else:
                getLogger(__name__).exception("Failed to log trigger event for %s", trig.trigger_id)

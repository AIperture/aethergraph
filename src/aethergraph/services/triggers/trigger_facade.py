from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from aethergraph.contracts.services.trigger import TriggerKind
from aethergraph.services.scope.scope import Scope, ScopeLevel
from aethergraph.services.triggers.engine import TriggerEngine
from aethergraph.services.triggers.trigger_service import TriggerService
from aethergraph.services.triggers.types import TriggerRecord


@dataclass
class TriggerConfig:
    kind: TriggerKind
    cron_expr: str | None = None
    run_at: datetime | None = None
    event_key: str | None = None
    interval_seconds: int | None = None
    memory_level: ScopeLevel | None = None
    tz: str | None = (
        None  # timezone for cron triggers, e.g., "America/New_York". Defaults to UTC if not set.
    )


@dataclass
class TriggerFacade:
    """
    Exposed on NodeContext as `context.triggers`.

    Designed to be thin; heavy lifting is done by TriggerService + ScopeFactory.
    """

    trigger_service: TriggerService
    trigger_engine: TriggerEngine
    scope: Scope

    # ------------ low-level: generic trigger management, mostly delegating to TriggerService --------------
    async def create(
        self,
        *,
        graph_id: str,
        default_inputs: dict[str, Any],
        config: TriggerConfig,
        trigger_name: str | None = None,
    ) -> TriggerRecord:
        """
        Generic entry: build a memory Scope from the current node then create a trigger.
        """

        return await self.trigger_service.create_from_scope(
            scope=self.scope,
            graph_id=graph_id,
            default_inputs=default_inputs,
            kind=config.kind,
            cron_expr=config.cron_expr,
            interval_seconds=config.interval_seconds,
            run_at=config.run_at,
            event_key=config.event_key,
            tz=config.tz,
            origin="schedule" if config.kind != "event" else "event",
            trigger_name=trigger_name,
        )

    # ------------ higher-level: event triggers with convenient defaults --------------
    async def cron(
        self,
        *,
        graph_id: str,
        default_inputs: dict[str, Any],
        cron_expr: str,
        tz: str | None = None,
        memory_level: ScopeLevel | None = None,
        trigger_name: str | None = None,
    ) -> TriggerRecord:
        cfg = TriggerConfig(
            kind="cron",
            cron_expr=cron_expr,
            tz=tz,
            memory_level=memory_level,
        )
        return await self.create(
            graph_id=graph_id,
            default_inputs=default_inputs,
            config=cfg,
            trigger_name=trigger_name,
        )

    async def at(
        self,
        *,
        graph_id: str,
        default_inputs: dict[str, Any],
        run_at: datetime,
        tz: str | None = None,
        memory_level: ScopeLevel | None = None,
        trigger_name: str | None = None,
    ) -> TriggerRecord:
        cfg = TriggerConfig(
            kind="one_shot",
            run_at=run_at,
            memory_level=memory_level,
            tz=tz,
        )
        return await self.create(
            graph_id=graph_id,
            default_inputs=default_inputs,
            config=cfg,
            trigger_name=trigger_name,
        )

    async def after(
        self,
        *,
        graph_id: str,
        default_inputs: dict[str, Any],
        delay_seconds: float,
        memory_level: ScopeLevel | None = None,
        trigger_name: str | None = None,
    ) -> TriggerRecord:
        """
        Convenience wrapper: schedule a one-shot trigger to fire after N seconds.

        Semantics:
        - Computes an absolute UTC timestamp: now_utc + delay_seconds
        - Delegates to `at(...)` with that run_at
        - `tz` is *not* needed here because we're using an absolute instant in time.
        """
        if delay_seconds <= 0:
            # Fire "immediately" – effectively as soon as the engine loop ticks.
            run_at = datetime.now(timezone.utc)
        else:
            run_at = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

        return await self.at(
            graph_id=graph_id,
            default_inputs=default_inputs,
            run_at=run_at,
            # no tz: run_at is already an aware UTC instant, treated as absolute
            memory_level=memory_level,
            trigger_name=trigger_name,
        )

    async def interval(
        self,
        *,
        graph_id: str,
        default_inputs: dict[str, Any],
        interval_seconds: int,
        memory_level: ScopeLevel | None = None,
        trigger_name: str | None = None,
    ) -> TriggerRecord:
        cfg = TriggerConfig(
            kind="interval",
            interval_seconds=interval_seconds,
            memory_level=memory_level,
        )
        return await self.create(
            graph_id=graph_id,
            default_inputs=default_inputs,
            config=cfg,
            trigger_name=trigger_name,
        )

    async def event(
        self,
        *,
        graph_id: str,
        default_inputs: dict[str, Any],
        event_key: str,
        memory_level: ScopeLevel | None = None,
        trigger_name: str | None = None,
    ) -> TriggerRecord:
        cfg = TriggerConfig(
            kind="event",
            event_key=event_key,
            memory_level=memory_level,
        )
        return await self.create(
            graph_id=graph_id,
            default_inputs=default_inputs,
            config=cfg,
            trigger_name=trigger_name,
        )

    async def cancel(self, trigger_id: str) -> None:
        """
        Soft-cancel / deactivate a trigger.

        This marks the trigger as inactive and clears next_fire_at.
        The TriggerEngine's poller (which uses list_active()) will
        stop executing it.
        """
        await self.trigger_service.cancel(trigger_id)

    async def get(self, trigger_id: str) -> TriggerRecord | None:
        return await self.trigger_service.get(trigger_id)

    async def fire_event(
        self,
        event_key: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Fire all active triggers for this event_key.

        payload, if provided, will be merged into default_inputs under key 'event'
        (we can change this merging policy).
        """
        org_id = self.scope.org_id
        user_id = self.scope.user_id
        client_id = self.scope.client_id
        await self.trigger_engine.fire_event(
            event_key=event_key,
            payload=payload,
            org_id=org_id,
            user_id=user_id,
            client_id=client_id,
        )

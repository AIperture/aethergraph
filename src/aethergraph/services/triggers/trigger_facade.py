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
    """Configuration for trigger execution.

    Attributes:
        kind (TriggerKind): The type of trigger to execute.
        cron_expr (str | None): Cron expression for scheduled triggers. Defaults to None.
        run_at (datetime | None): Specific datetime to run the trigger. Defaults to None.
        event_key (str | None): Key identifier for event-based triggers. Defaults to None.
        interval_seconds (int | None): Interval in seconds for periodic triggers. Defaults to None.
        memory_level (ScopeLevel | None): Scope level for trigger memory/state management. Defaults to None.
        tz (str | None): Timezone for cron triggers (e.g., "America/New_York"). Defaults to UTC if not set.
    """

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
    Expose trigger scheduling and event firing APIs on `context.triggers`.

    This facade is intentionally thin: it builds small convenience configs and
    delegates persistence and execution behavior to `trigger_service` and
    `trigger_engine`.

    Examples:
        Create a cron trigger from node code:
        ```python
        trig = await context.triggers.cron(
            graph_id="daily-report",
            default_inputs={"channel": "ops"},
            cron_expr="0 9 * * 1-5",
            tz="America/Los_Angeles",
            trigger_name="weekday-report",
        )
        ```

        Fire an event trigger manually:
        ```python
        await context.triggers.fire_event(
            "invoice.paid",
            payload={"invoice_id": "inv_123"},
        )
        ```

    Args:
        trigger_service: Service used to create, cancel, and fetch trigger records.
        trigger_engine: Engine used to fan out and execute event-based triggers.
        scope: Bound runtime scope used for tenant-aware trigger creation/firing.

    Returns:
        TriggerFacade: Dataclass wrapper exposing node-friendly trigger methods.

    Notes:
        The facade does not run polling loops itself; scheduled execution is
        handled by `TriggerEngine.run_forever(...)` elsewhere in runtime wiring.
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
        Create a trigger from an explicit `TriggerConfig`.

        This is the generic entrypoint used by convenience helpers (`cron`, `at`,
        `after`, `interval`, `event`) and delegates to
        `trigger_service.create_from_scope(...)` using the bound `scope`.

        Examples:
            Create a one-shot trigger:
            ```python
            trig = await context.triggers.create(
                graph_id="reminder-graph",
                default_inputs={"message": "Ping me"},
                config=TriggerConfig(kind="one_shot", run_at=run_at_utc),
                trigger_name="single-reminder",
            )
            ```

            Create an event trigger:
            ```python
            trig = await context.triggers.create(
                graph_id="billing-graph",
                default_inputs={"source": "billing"},
                config=TriggerConfig(kind="event", event_key="invoice.paid"),
            )
            ```

        Args:
            graph_id: Graph to execute when the trigger fires.
            default_inputs: Base inputs merged into submitted runs.
            config: Trigger configuration payload describing kind and timing.
            trigger_name: Optional human-readable trigger label.

        Returns:
            TriggerRecord: Persisted trigger record returned by the trigger
                service.

        Notes:
            `config.memory_level` is currently accepted for API compatibility but
            is not consumed by this facade method.
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
        """
        Create a cron-based recurring trigger.

        This helper builds a `TriggerConfig(kind="cron", ...)` and forwards to
        `create(...)`.

        Examples:
            Weekday 9AM local trigger:
            ```python
            trig = await context.triggers.cron(
                graph_id="daily-report",
                default_inputs={"team": "ops"},
                cron_expr="0 9 * * 1-5",
                tz="America/Los_Angeles",
            )
            ```

            Hourly trigger in UTC:
            ```python
            trig = await context.triggers.cron(
                graph_id="sync-graph",
                default_inputs={},
                cron_expr="0 * * * *",
                trigger_name="hourly-sync",
            )
            ```

        Args:
            graph_id: Graph to execute when schedule fires.
            default_inputs: Base run inputs for each fire.
            cron_expr: Cron expression interpreted by backend scheduler.
            tz: Optional timezone for cron interpretation; defaults to UTC when
                omitted by the service.
            memory_level: Reserved scope-level hint for future behavior.
            trigger_name: Optional human-readable trigger label.

        Returns:
            TriggerRecord: Persisted cron trigger record.
        """
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
        """
        Create a one-shot trigger scheduled for an absolute datetime.

        This helper builds `TriggerConfig(kind="one_shot", run_at=...)` and
        forwards to `create(...)`.

        Examples:
            Schedule with timezone-aware UTC datetime:
            ```python
            trig = await context.triggers.at(
                graph_id="notify-graph",
                default_inputs={"message": "check deployment"},
                run_at=run_at_utc,
                trigger_name="deploy-reminder",
            )
            ```

            Schedule with explicit timezone label:
            ```python
            trig = await context.triggers.at(
                graph_id="reminder-graph",
                default_inputs={"message": "standup"},
                run_at=run_at_local,
                tz="America/New_York",
            )
            ```

        Args:
            graph_id: Graph to execute at the scheduled instant.
            default_inputs: Base run inputs.
            run_at: Absolute fire datetime.
            tz: Optional timezone hint for normalization by service logic.
            memory_level: Reserved scope-level hint for future behavior.
            trigger_name: Optional human-readable trigger label.

        Returns:
            TriggerRecord: Persisted one-shot trigger record.

        Notes:
            For naive datetimes, timezone interpretation is delegated to service
            behavior and trigger timezone settings.
        """
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
        Schedule a one-shot trigger that fires after a delay.

        This helper computes `run_at` in UTC as `now + delay_seconds` and
        delegates to `at(...)`.

        Examples:
            Fire after 5 minutes:
            ```python
            trig = await context.triggers.after(
                graph_id="cleanup-graph",
                default_inputs={"job": "stale-cache"},
                delay_seconds=300,
            )
            ```

            Request immediate scheduling:
            ```python
            trig = await context.triggers.after(
                graph_id="notify-graph",
                default_inputs={"message": "run now"},
                delay_seconds=0,
            )
            ```

        Args:
            graph_id: Graph to execute when delay elapses.
            default_inputs: Base run inputs.
            delay_seconds: Delay in seconds before first fire.
            memory_level: Reserved scope-level hint for future behavior.
            trigger_name: Optional human-readable trigger label.

        Returns:
            TriggerRecord: Persisted one-shot trigger record with computed
                `run_at`.

        Notes:
            Non-positive delay values are coerced to "as soon as possible" by
            using the current UTC timestamp.
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
        """
        Create an interval trigger that repeats every N seconds.

        This helper builds `TriggerConfig(kind="interval", interval_seconds=...)`
        and forwards to `create(...)`.

        Examples:
            Run every minute:
            ```python
            trig = await context.triggers.interval(
                graph_id="heartbeat-graph",
                default_inputs={"source": "scheduler"},
                interval_seconds=60,
            )
            ```

            Run every 10 minutes with a name:
            ```python
            trig = await context.triggers.interval(
                graph_id="sync-graph",
                default_inputs={"full": False},
                interval_seconds=600,
                trigger_name="periodic-sync",
            )
            ```

        Args:
            graph_id: Graph to execute on each interval tick.
            default_inputs: Base run inputs.
            interval_seconds: Repeat interval in seconds.
            memory_level: Reserved scope-level hint for future behavior.
            trigger_name: Optional human-readable trigger label.

        Returns:
            TriggerRecord: Persisted interval trigger record.

        Notes:
            Validation of minimum/maximum interval values is delegated to service
            or storage policies.
        """
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
        """
        Create an event-driven trigger for a specific event key.

        This helper builds `TriggerConfig(kind="event", event_key=...)` and
        forwards to `create(...)`. Event triggers fire only when
        `fire_event(...)` is invoked with the same key.

        Examples:
            Create a billing event trigger:
            ```python
            trig = await context.triggers.event(
                graph_id="billing-graph",
                default_inputs={"origin": "webhook"},
                event_key="invoice.paid",
            )
            ```

            Create with explicit label:
            ```python
            trig = await context.triggers.event(
                graph_id="audit-graph",
                default_inputs={},
                event_key="user.deleted",
                trigger_name="audit-user-delete",
            )
            ```

        Args:
            graph_id: Graph to execute when matching events are fired.
            default_inputs: Base run inputs merged with event payload.
            event_key: Event routing key used for matching.
            memory_level: Reserved scope-level hint for future behavior.
            trigger_name: Optional human-readable trigger label.

        Returns:
            TriggerRecord: Persisted event trigger record.

        Notes:
            Event payload merge policy is implemented in the engine and currently
            nests payload under `event`.
        """
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
        Soft-cancel and deactivate a trigger by id.

        This delegates to `trigger_service.cancel(...)`, which marks the trigger
        inactive and clears future scheduling metadata.

        Examples:
            Cancel a known trigger:
            ```python
            await context.triggers.cancel("trig-ab12cd34")
            ```

            Cancel after fetching:
            ```python
            trig = await context.triggers.get("trig-ab12cd34")
            if trig:
                await context.triggers.cancel(trig.trigger_id)
            ```

        Args:
            trigger_id: Trigger identifier to deactivate.

        Returns:
            None: This method returns no value.

        Notes:
            Canceling a missing trigger is typically a no-op in the service.
        """
        await self.trigger_service.cancel(trigger_id)

    async def get(self, trigger_id: str) -> TriggerRecord | None:
        """
        Fetch a trigger record by id.

        This delegates directly to `trigger_service.get(...)`.

        Examples:
            Read a trigger record:
            ```python
            trig = await context.triggers.get("trig-ab12cd34")
            ```

            Branch if missing:
            ```python
            trig = await context.triggers.get("trig-missing")
            if trig is None:
                ...
            ```

        Args:
            trigger_id: Trigger identifier to retrieve.

        Returns:
            TriggerRecord | None: Persisted trigger record, or `None` when not
                found.

        Notes:
            No additional scope filtering is applied in this facade method.
        """
        return await self.trigger_service.get(trigger_id)

    async def fire_event(
        self,
        event_key: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Fire active event triggers matching `event_key` in current scope.

        This delegates to `trigger_engine.fire_event(...)`, forwarding tenant
        fields (`org_id`, `user_id`, `client_id`) from `self.scope`.

        Examples:
            Fire event without payload:
            ```python
            await context.triggers.fire_event("invoice.paid")
            ```

            Fire event with payload:
            ```python
            await context.triggers.fire_event(
                "invoice.paid",
                payload={"invoice_id": "inv_123", "amount": 1999},
            )
            ```

        Args:
            event_key: Routing key used to select active event triggers.
            payload: Optional event payload forwarded to trigger run inputs.

        Returns:
            None: This method returns no value.

        Notes:
            Payload merge policy is currently engine-defined (`{"event": payload}`)
            when payload is provided.
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

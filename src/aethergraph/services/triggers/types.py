from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from aethergraph.contracts.integration import OriginBinding
from aethergraph.contracts.services.trigger import TriggerKind
from aethergraph.core.runtime.run_types import RunParent
from aethergraph.services.scope.scope import Scope, ScopeLevel


@dataclass
class TriggerRecord:
    """
    Persistent trigger description.

    Triggers are "scopeful": they remember enough identity / context so that
    runs they spawn share the same behavior for memory, artifacts, and KB
    as the scope at trigger-creation time. Each execution gets new run/node
    identities; the creating run remains a separate parent provenance link.
    """

    trigger_id: str
    trigger_name: str | None = (
        None  # optional human-friendly name for UI; not used by the system, just stored as metadata
    )

    # Ownership / identity
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    mode: str | None = None  # "cloud", "demo", "local", etc.

    app_id: str | None = field(
        default=None,
        metadata={
            "deprecated": True,
            "description": (
                "Deprecated; retained for compatibility and scheduled for removal "
                "in a future breaking release."
            ),
        },
    )
    agent_id: str | None = None
    session_id: str | None = None

    memory_level: ScopeLevel | None = None

    # What to run
    graph_id: str | None = None
    default_inputs: dict[str, Any] = field(default_factory=dict)
    origin: str = "schedule"  # "schedule", "event", "agent" etc

    # Trigger config
    kind: TriggerKind = "cron"
    cron_expr: str | None = None  # for "cron" kind
    interval_seconds: int | None = None  # for "interval" kind
    run_at: datetime | None = None  # for "one_shot" kind
    event_key: str | None = None  # for "event" kind
    tz: str | None = (
        None  # timezone for cron expressions (e.g. "America/Los_Angeles"); defaults to UTC if not set
    )

    max_overlap_runs: int | None = (
        None  # if set, max number of overlapping runs allowed; excess runs will be skipped
    )
    catch_up_missed: bool = False  # if true, missed runs (e.g. due to downtime) will be triggered on startup; if false, they will be skipped

    # Lifecycle
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_fired_at: datetime | None = None
    next_fire_at: datetime | None = None

    # Freeform metadata for UI / debugging
    meta: dict[str, Any] = field(default_factory=dict)

    origin_binding: OriginBinding | None = None
    parent_run: RunParent | None = None

    def __post_init__(self) -> None:
        if self.parent_run is not None:
            if isinstance(self.parent_run, dict):
                self.parent_run = RunParent(**self.parent_run)
            if (
                not isinstance(self.parent_run, RunParent)
                or self.parent_run.session_id != self.session_id
            ):
                raise ValueError("Trigger parent run must match its session")
        if self.origin_binding is not None:
            self.origin_binding = OriginBinding.model_validate(self.origin_binding)
            if self.origin_binding.session_id != self.session_id:
                raise ValueError("Trigger origin binding must match its session")

    # -------------- helpers --------------
    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable trigger service projection.

        The result contains schedule, ownership, launch-context, and lifecycle fields
        with timestamps normalized to ISO strings for public service transport.

        Examples:
            Serialize a scheduled trigger:
                ```python
                payload = trigger.to_dict()
                ```

            Read the stable trigger identity:
                ```python
                assert trigger.to_dict()["trigger_id"] == trigger.trigger_id
                ```

        Args:
            None.

        Returns:
            dict[str, Any]: Detached JSON-compatible trigger fields.

        Notes:
            `app_id` is retained only as deprecated optional compatibility metadata;
            canonical provider scope and authorization never depend on it.
        """

        def _dt(d: datetime | None) -> str | None:
            return d.isoformat() if d is not None else None

        return {
            "trigger_id": self.trigger_id,
            "trigger_name": self.trigger_name,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "client_id": self.client_id,
            "mode": self.mode,
            "app_id": self.app_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "memory_level": self.memory_level,
            "graph_id": self.graph_id,
            "default_inputs": self.default_inputs,
            "origin": self.origin,
            "parent_run": asdict(self.parent_run) if self.parent_run is not None else None,
            "origin_binding": None
            if self.origin_binding is None
            else self.origin_binding.model_dump(mode="json"),
            "kind": self.kind,
            "cron_expr": self.cron_expr,
            "interval_seconds": self.interval_seconds,
            "run_at": _dt(self.run_at),
            "event_key": self.event_key,
            "tz": self.tz,
            "max_overlap_runs": self.max_overlap_runs,
            "catch_up_missed": self.catch_up_missed,
            "active": self.active,
            "created_at": _dt(self.created_at),
            "last_fired_at": _dt(self.last_fired_at),
            "next_fire_at": _dt(self.next_fire_at),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriggerRecord:
        """Restore a trigger and validate its retained channel origin.

        Parse persisted timestamps and reconstruct the typed origin binding in
        the record's validation boundary.

        Examples:
            Restore a stored record:
            ```python
            restored = TriggerRecord.from_dict(trigger.to_dict())
            assert restored.trigger_id == trigger.trigger_id
            ```

            Preserve the channel route:
            ```python
            restored = TriggerRecord.from_dict(trigger.to_dict())
            assert restored.origin_binding == trigger.origin_binding
            ```

        Args:
            data: Serialized trigger service record.

        Returns:
            TriggerRecord: Validated scope, schedule and launch context.

        Notes:
            Records without an origin binding retain no implicit channel route.
            A binding for another session is rejected.
        """

        def _dt(v: Any) -> datetime | None:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v
            return datetime.fromisoformat(v)

        return cls(
            trigger_id=data["trigger_id"],
            trigger_name=data.get("trigger_name"),
            org_id=data.get("org_id"),
            user_id=data.get("user_id"),
            client_id=data.get("client_id"),
            mode=data.get("mode"),
            app_id=data.get("app_id"),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
            memory_level=data.get("memory_level"),
            graph_id=data.get("graph_id"),
            default_inputs=data.get("default_inputs") or {},
            origin=data.get("origin", "schedule"),
            parent_run=data.get("parent_run"),
            origin_binding=data.get("origin_binding"),
            kind=data.get("kind", "cron"),
            cron_expr=data.get("cron_expr"),
            interval_seconds=data.get("interval_seconds"),
            run_at=_dt(data.get("run_at")),
            event_key=data.get("event_key"),
            tz=data.get("tz"),
            max_overlap_runs=data.get("max_overlap_runs"),
            catch_up_missed=data.get("catch_up_missed", False),
            active=data.get("active", True),
            created_at=_dt(data.get("created_at")) or datetime.now(UTC),
            last_fired_at=_dt(data.get("last_fired_at")),
            next_fire_at=_dt(data.get("next_fire_at")),
            meta=data.get("meta") or {},
        )

    @classmethod
    def from_scope(
        cls,
        *,
        trigger_id: str,
        scope: Scope,
        graph_id: str,
        default_inputs: dict[str, Any],
        kind: TriggerKind,
        trigger_name: str | None = None,
        origin: str = "schedule",
        cron_expr: str | None = None,
        interval_seconds: int | None = None,
        run_at: datetime | None = None,
        event_key: str | None = None,
        tz: str | None = None,
        max_overlap_runs: int | None = None,
        catch_up_missed: bool = False,
        meta: dict[str, Any] | None = None,
        origin_binding: OriginBinding | None = None,
    ) -> TriggerRecord:
        """Build a scoped trigger with the originating run retained as its parent.

        Retain tenant, session and Agent identity together with an optional
        immutable channel route for later scheduled executions. A new scheduled
        run has its own identity and a parent link to the original creating run.

        Examples:
            Build an interval schedule:
            ```python
            trigger = TriggerRecord.from_scope(
                trigger_id="trigger-1", scope=scope, graph_id="poll",
                default_inputs={}, kind="interval", interval_seconds=10,
            )
            ```

            Retain a session's output route:
            ```python
            trigger = TriggerRecord.from_scope(
                trigger_id="trigger-2", scope=scope, graph_id="notify",
                default_inputs={}, kind="event", event_key="job.done",
                origin_binding=origin_binding,
            )
            ```

        Args:
            trigger_id: Exact trigger identity.
            scope: Tenant, session and Agent scope to retain.
            graph_id: Registered graph to execute.
            default_inputs: Base inputs for scheduled execution.
            kind: Schedule discriminator.
            trigger_name: Optional display label.
            origin: Scheduling origin metadata.
            cron_expr: Cron expression for cron schedules.
            interval_seconds: Interval cadence in seconds.
            run_at: One-shot due time.
            event_key: Event name for event schedules.
            tz: Optional scheduling time zone.
            max_overlap_runs: Optional simultaneous-run limit.
            catch_up_missed: Whether restart recovers missed occurrences.
            meta: Caller metadata.
            origin_binding: Optional session-matching channel origin.

        Returns:
            TriggerRecord: Record ready for scheduling validation and persistence.

        Notes:
            Channel routes never select a different session. Run and node identity
            are supplied anew when the scheduler admits an execution. When both
            exist, the creating run/session are retained only as parent provenance.
        """
        return cls(
            trigger_id=trigger_id,
            org_id=scope.org_id,
            user_id=scope.user_id,
            client_id=scope.client_id,
            mode=scope.mode,
            app_id=scope.app_id,
            agent_id=scope.agent_id,
            session_id=scope.session_id,
            memory_level=scope.memory_level,
            graph_id=graph_id,
            default_inputs=dict(default_inputs or {}),
            origin=origin,
            parent_run=RunParent(scope.run_id, scope.session_id)
            if scope.run_id and scope.session_id
            else None,
            origin_binding=origin_binding,
            kind=kind,
            trigger_name=trigger_name,
            cron_expr=cron_expr,
            interval_seconds=interval_seconds,
            run_at=run_at,
            event_key=event_key,
            tz=tz,
            max_overlap_runs=max_overlap_runs,
            catch_up_missed=catch_up_missed,
            meta=dict(meta or {}),
        )


@dataclass(frozen=True)
class TriggerClaim:
    """One durable, worker-owned trigger occurrence."""

    fire_id: str
    trigger: TriggerRecord
    scheduled_for: datetime
    worker_id: str
    lease_until: datetime
    attempts: int
    reclaimed: bool = False

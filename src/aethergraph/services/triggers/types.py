from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aethergraph.contracts.services.trigger import TriggerKind
from aethergraph.services.scope.scope import Scope, ScopeLevel


@dataclass
class TriggerRecord:
    """
    Persistent trigger description.

    Triggers are "scopeful": they remember enough identity / context so that
    runs they spawn share the same behavior for memory, artifacts, and KB
    as the scope at trigger-creation time (minus run/node IDs).
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

    app_id: str | None = None
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_fired_at: datetime | None = None
    next_fire_at: datetime | None = None

    # Freeform metadata for UI / debugging
    meta: dict[str, Any] = field(default_factory=dict)

    # -------------- helpers --------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation for DocStore."""

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
            kind=data.get("kind", "cron"),
            cron_expr=data.get("cron_expr"),
            interval_seconds=data.get("interval_seconds"),
            run_at=_dt(data.get("run_at")),
            event_key=data.get("event_key"),
            tz=data.get("tz"),
            max_overlap_runs=data.get("max_overlap_runs"),
            catch_up_missed=data.get("catch_up_missed", False),
            active=data.get("active", True),
            created_at=_dt(data.get("created_at")) or datetime.now(timezone.utc),
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
    ) -> TriggerRecord:
        """
        Build a TriggerRecord from a Scope, intentionally omitting run_id/node_id.
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

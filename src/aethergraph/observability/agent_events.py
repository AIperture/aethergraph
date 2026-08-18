from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
import uuid

from aethergraph.core.runtime.runtime_metering import current_meter_context
from aethergraph.observability.models import ObservationRecord, ObservationScope
from aethergraph.server.security.redaction import sanitize_content, sanitize_text


def _utc_ts() -> float:
    return datetime.now(UTC).timestamp()


@dataclass(frozen=True)
class AgentEventTypeMeta:
    event_type: str
    category: str
    display_label: str
    payload_schema_name: str | None = None
    payload_schema_version: int | None = None
    renderer_hint: str | None = None
    redaction_policy: str | None = None


@dataclass
class AgentEventTypeRegistry:
    _entries: dict[str, AgentEventTypeMeta] = field(default_factory=dict)

    def register(
        self,
        *,
        event_type: str,
        category: str,
        display_label: str,
        payload_schema_name: str | None = None,
        payload_schema_version: int | None = None,
        renderer_hint: str | None = None,
        redaction_policy: str | None = None,
    ) -> AgentEventTypeMeta:
        meta = AgentEventTypeMeta(
            event_type=event_type,
            category=category,
            display_label=display_label,
            payload_schema_name=payload_schema_name,
            payload_schema_version=payload_schema_version,
            renderer_hint=renderer_hint,
            redaction_policy=redaction_policy,
        )
        self._entries[event_type] = meta
        return meta

    def get(self, event_type: str) -> AgentEventTypeMeta | None:
        return self._entries.get(event_type)

    def list(self) -> list[AgentEventTypeMeta]:
        return list(self._entries.values())


def register_default_agent_event_types(registry: AgentEventTypeRegistry) -> AgentEventTypeRegistry:
    defaults = {
        "planning.started": ("planning", "Planning Started"),
        "planning.updated": ("planning", "Planning Updated"),
        "planning.completed": ("planning", "Planning Completed"),
        "step.started": ("step", "Step Started"),
        "step.completed": ("step", "Step Completed"),
        "step.failed": ("step", "Step Failed"),
        "tool.selected": ("tool", "Tool Selected"),
        "tool.called": ("tool", "Tool Called"),
        "tool.failed": ("tool", "Tool Failed"),
        "recovery.started": ("recovery", "Recovery Started"),
        "recovery.retry": ("recovery", "Recovery Retry"),
        "recovery.replan": ("recovery", "Recovery Replan"),
        "recovery.escalated": ("recovery", "Recovery Escalated"),
        "wait.requested": ("wait", "Wait Requested"),
        "wait.resolved": ("wait", "Wait Resolved"),
        "approval.requested": ("approval", "Approval Requested"),
        "approval.resolved": ("approval", "Approval Resolved"),
    }
    for event_type, (category, display_label) in defaults.items():
        registry.register(
            event_type=event_type,
            category=category,
            display_label=display_label,
            payload_schema_name=event_type,
            payload_schema_version=1,
        )
    return registry


async def emit_agent_event(
    *,
    event_type: str,
    summary: str,
    payload: dict[str, Any] | None = None,
    status: str = "info",
    tags: list[str] | None = None,
    producer_family: str = "agent",
    producer_name: str = "unknown",
    producer_version: str | None = None,
    payload_schema_name: str | None = None,
    payload_schema_version: int | None = 1,
    parent_event_id: str | None = None,
    caused_by_event_id: str | None = None,
    observation_sink: Any | None = None,
) -> dict[str, Any]:
    if observation_sink is None:
        from aethergraph.core.runtime.runtime_services import current_services

        observation_sink = getattr(current_services(), "observation_sink", None)
    if observation_sink is None:
        raise RuntimeError("Observation sink not available")

    ctx = dict(current_meter_context.get() or {})
    scope = {
        "org_id": ctx.get("org_id"),
        "user_id": ctx.get("user_id"),
        "client_id": ctx.get("client_id"),
        "run_id": ctx.get("run_id"),
        "session_id": ctx.get("session_id"),
        "agent_id": ctx.get("agent_id"),
        "app_id": ctx.get("app_id"),
        "graph_id": ctx.get("graph_id"),
        "node_id": ctx.get("node_id"),
        "trace_id": ctx.get("trace_id"),
        "span_id": ctx.get("span_id"),
    }
    event_id = f"agt_{uuid.uuid4().hex}"
    envelope = {
        "event_id": event_id,
        "ts": _utc_ts(),
        "kind": "agent_event",
        "event_type": event_type,
        "producer": {
            "family": producer_family,
            "name": producer_name,
            "version": producer_version,
        },
        "scope": {k: v for k, v in scope.items() if v is not None},
        "status": status,
        "summary": sanitize_text(summary),
        "tags": sanitize_content(list(tags or [])),
        "payload": sanitize_content(payload or {}),
        "payload_schema": {
            "name": payload_schema_name or event_type,
            "version": payload_schema_version,
        },
        "links": {
            "parent_event_id": parent_event_id,
            "caused_by_event_id": caused_by_event_id,
        },
    }
    await observation_sink.append_observation(
        ObservationRecord(
            observation_id=event_id,
            category="agent_event",
            name=event_type,
            summary=envelope["summary"],
            occurred_at=datetime.fromtimestamp(envelope["ts"], tz=UTC).isoformat(),
            status="error" if status == "error" else "ok",
            severity=status
            if status in {"debug", "info", "warning", "error", "critical"}
            else "info",
            scope=ObservationScope.from_dimensions(ctx),
            parent_observation_id=parent_event_id,
            caused_by_observation_id=caused_by_event_id,
            attributes={
                "event_type": event_type,
                "producer": envelope["producer"],
                "status": status,
                "tags": envelope["tags"],
                "payload": envelope["payload"],
                "payload_schema": envelope["payload_schema"],
                "links": envelope["links"],
            },
        )
    )
    return envelope

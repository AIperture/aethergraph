from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid

from aethergraph.core.runtime.runtime_metering import current_meter_context


def _utc_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())
        except Exception:
            return repr(value)
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())
        except Exception:
            return repr(value)
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value))
        except Exception:
            return repr(value)
    return repr(value)


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
    event_log: Any | None = None,
) -> dict[str, Any]:
    if event_log is None:
        from aethergraph.core.runtime.runtime_services import current_services

        event_log = getattr(current_services(), "eventlog", None)
    if event_log is None:
        raise RuntimeError("Event log not available")

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
        "summary": summary,
        "tags": list(tags or []),
        "payload": _json_safe(payload or {}),
        "payload_schema": {
            "name": payload_schema_name or event_type,
            "version": payload_schema_version,
        },
        "links": {
            "parent_event_id": parent_event_id,
            "caused_by_event_id": caused_by_event_id,
        },
    }
    scope_id = (
        scope.get("run_id")
        or (f"session:{scope['session_id']}" if scope.get("session_id") else None)
        or f"agent:{producer_name}"
    )
    row = {
        "id": event_id,
        "ts": envelope["ts"],
        "scope_id": scope_id,
        "kind": "agent_event",
        "payload": envelope,
        "tags": envelope["tags"],
        "user_id": scope.get("user_id"),
        "org_id": scope.get("org_id"),
    }
    await event_log.append(row)
    return envelope

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aethergraph.contracts.services.memory import Event, MemoryTenantFilter


def parse_time_value(value: str | datetime | float | int | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def event_time(event: Event | dict[str, Any]) -> datetime | None:
    return parse_time_value(
        getattr(event, "ts", None) if not isinstance(event, dict) else event.get("ts")
    )


def normalize_memory_tenant(
    *,
    org_id: str | None = None,
    user_id: str | None = None,
    client_id: str | None = None,
) -> MemoryTenantFilter | None:
    out: MemoryTenantFilter = {}
    if org_id:
        out["org_id"] = org_id
    if user_id:
        out["user_id"] = user_id
    if client_id:
        out["client_id"] = client_id
    return out or None


def tenant_matches_event(
    event: Event | dict[str, Any],
    tenant: MemoryTenantFilter | None,
) -> bool:
    if not tenant:
        return True

    def _get(key: str) -> Any:
        return getattr(event, key, None) if not isinstance(event, dict) else event.get(key)

    for key, expected in tenant.items():
        if expected is None:
            continue
        actual = _get(key)
        if actual is None or actual != expected:
            return False
    return True


def event_matches_filters(
    event: Event | dict[str, Any],
    *,
    tenant: MemoryTenantFilter | None = None,
    kinds: list[str] | None = None,
    tags: list[str] | None = None,
    since: str | datetime | float | int | None = None,
    until: str | datetime | float | int | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
    client_id: str | None = None,
    graph_id: str | None = None,
    node_id: str | None = None,
    topic: str | None = None,
    tool: str | None = None,
) -> bool:
    if not tenant_matches_event(event, tenant):
        return False

    def _get(key: str) -> Any:
        return getattr(event, key, None) if not isinstance(event, dict) else event.get(key)

    if kinds and _get("kind") not in kinds:
        return False

    if tags:
        row_tags = set(_get("tags") or [])
        if not row_tags.issuperset(tags):
            return False

    event_dt = event_time(event)
    since_dt = parse_time_value(since)
    until_dt = parse_time_value(until)
    if since_dt is not None and (event_dt is None or event_dt < since_dt):
        return False
    if until_dt is not None and (event_dt is None or event_dt > until_dt):
        return False

    if session_id is not None and _get("session_id") != session_id:
        return False
    if run_id is not None and _get("run_id") != run_id:
        return False
    if agent_id is not None and _get("agent_id") != agent_id:
        return False
    if client_id is not None and _get("client_id") != client_id:
        return False
    if graph_id is not None and _get("graph_id") != graph_id:
        return False
    if node_id is not None and _get("node_id") != node_id:
        return False
    if topic is not None and _get("topic") != topic:
        return False
    if tool is not None and _get("tool") != tool:
        return False

    return True


def summary_matches_filters(
    summary: dict[str, Any],
    *,
    tenant: MemoryTenantFilter | None = None,
    scope_id: str | None = None,
    summary_tag: str | None = None,
) -> bool:
    if tenant and not tenant_matches_event(summary, tenant):
        return False
    if scope_id is not None and summary.get("scope_id") != scope_id:
        return False
    if summary_tag is not None and summary.get("summary_tag") != summary_tag:
        return False
    return True

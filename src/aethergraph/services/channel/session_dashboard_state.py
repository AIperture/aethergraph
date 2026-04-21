from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from aethergraph.core.runtime.runtime_services import current_services

DASHBOARD_STATE_EVENT_KIND = "session_dashboard_state"
DASHBOARD_STATE_EVENT_TYPE = "dashboard_state.updated"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _snapshot_ts() -> float:
    return datetime.now(UTC).timestamp()


def _normalize_op(op: dict[str, Any], *, fallback_index: int) -> dict[str, Any]:
    payload = dict(op or {})
    normalized = {
        "op": str(payload.get("op") or "replace"),
        "path": str(payload.get("path") or "/"),
    }
    if "value" in payload:
        normalized["value"] = payload.get("value")
    normalized["seq"] = int(
        payload.get("seq") if payload.get("seq") is not None else fallback_index
    )
    return normalized


def _normalize_dashboard_state(
    dashboard_state: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not dashboard_state:
        return None
    raw = dict(dashboard_state)
    return {
        "dashboard_id": str(raw.get("dashboard_id") or ""),
        "dashboard_type": str(raw.get("dashboard_type") or "generic.dashboard"),
        "workflow_id": str(raw.get("workflow_id") or ""),
        "revision": int(raw.get("revision") or 0),
        "status": str(raw.get("status") or "idle"),
        "updated_at": str(raw.get("updated_at") or _utc_now_iso()),
        "data": raw.get("data") if isinstance(raw.get("data"), dict) else {},
    }


def _split_path(path: str) -> list[str]:
    if not path or path == "/":
        return []
    return [part.replace("~1", "/").replace("~0", "~") for part in path.split("/") if part]


def _ensure_container(parent: Any, key: str, next_key: str | None) -> Any:
    if isinstance(parent, dict):
        if key not in parent or not isinstance(parent[key], (dict, list)):
            parent[key] = [] if next_key == "-" or (next_key and next_key.isdigit()) else {}
        return parent[key]
    if isinstance(parent, list):
        index = len(parent) if key == "-" else int(key)
        while len(parent) <= index:
            parent.append([] if next_key == "-" or (next_key and next_key.isdigit()) else {})
        if not isinstance(parent[index], (dict, list)):
            parent[index] = [] if next_key == "-" or (next_key and next_key.isdigit()) else {}
        return parent[index]
    raise TypeError(f"Unsupported parent container for key {key!r}")


def _resolve_parent(root: Any, path: str, *, create: bool) -> tuple[Any, str | None]:
    parts = _split_path(path)
    if not parts:
        return root, None
    node = root
    for index, part in enumerate(parts[:-1]):
        next_key = parts[index + 1] if index + 1 < len(parts) else None
        if isinstance(node, dict):
            if part not in node:
                if not create:
                    raise KeyError(path)
                node[part] = [] if next_key == "-" or (next_key and next_key.isdigit()) else {}
            node = node[part]
        elif isinstance(node, list):
            position = len(node) if part == "-" else int(part)
            if position >= len(node):
                if not create:
                    raise KeyError(path)
                while len(node) <= position:
                    node.append([] if next_key == "-" or (next_key and next_key.isdigit()) else {})
            node = node[position]
        else:
            raise KeyError(path)
        if create and not isinstance(node, (dict, list)):
            node = _ensure_container(root, part, next_key)
    return node, parts[-1]


def _apply_patch_ops(data: dict[str, Any], ops: list[dict[str, Any]] | None) -> dict[str, Any]:
    snapshot = dict(data or {})
    for index, raw_op in enumerate(ops or []):
        op = _normalize_op(raw_op if isinstance(raw_op, dict) else {}, fallback_index=index)
        kind = op["op"]
        path = op["path"]
        value = op.get("value")
        if kind == "replace" and path == "/":
            snapshot = value if isinstance(value, dict) else {}
            continue
        parent, key = _resolve_parent(snapshot, path, create=kind in {"replace", "add", "append"})
        if key is None:
            continue
        if kind == "remove":
            if isinstance(parent, dict):
                parent.pop(key, None)
            elif isinstance(parent, list):
                position = int(key)
                if 0 <= position < len(parent):
                    parent.pop(position)
            continue
        if isinstance(parent, dict):
            if kind in {"replace", "add"}:
                parent[key] = value
            elif kind == "append":
                target = parent.get(key)
                if not isinstance(target, list):
                    target = []
                    parent[key] = target
                if isinstance(value, list):
                    target.extend(value)
                else:
                    target.append(value)
            continue
        if isinstance(parent, list):
            if key == "-":
                if kind == "append":
                    if isinstance(value, list):
                        parent.extend(value)
                    else:
                        parent.append(value)
                else:
                    parent.append(value)
                continue
            position = int(key)
            while len(parent) <= position:
                parent.append(None)
            if kind in {"replace", "add"}:
                parent[position] = value
            elif kind == "append":
                target = parent[position]
                if not isinstance(target, list):
                    target = []
                    parent[position] = target
                if isinstance(value, list):
                    target.extend(value)
                else:
                    target.append(value)
    return snapshot


async def _publish_row(
    *,
    session_id: str,
    op: str,
    dashboard: dict[str, Any] | None,
    patch: dict[str, Any] | None,
    meta: dict[str, Any],
) -> dict[str, Any]:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    event_hub = getattr(container, "eventhub", None)
    if event_log is None:
        raise RuntimeError("EventLog not available")

    row = {
        "id": str(uuid4()),
        "ts": _snapshot_ts(),
        "scope_id": session_id,
        "kind": DASHBOARD_STATE_EVENT_KIND,
        "payload": {
            "type": DASHBOARD_STATE_EVENT_TYPE,
            "op": op,
            "dashboard": dashboard,
            "patch": patch,
            "meta": dict(meta or {}),
        },
    }
    await event_log.append(row)
    if event_hub is not None:
        await event_hub.broadcast(row)
    return row


async def list_session_dashboard_states(session_id: str) -> list[dict[str, Any]]:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        return []
    rows = await event_log.query(
        scope_id=session_id, kinds=[DASHBOARD_STATE_EVENT_KIND], limit=1000
    )
    latest_by_id: dict[str, dict[str, Any] | None] = {}
    for row in rows:
        payload = dict((row or {}).get("payload") or {})
        dashboard = _normalize_dashboard_state(payload.get("dashboard"))
        if not dashboard:
            continue
        latest_by_id[str(dashboard.get("dashboard_id") or "")] = dashboard
    return [item for item in latest_by_id.values() if item]


async def get_session_dashboard_state(
    session_id: str,
    dashboard_id: str,
) -> dict[str, Any] | None:
    dashboards = await list_session_dashboard_states(session_id)
    for dashboard in dashboards:
        if str(dashboard.get("dashboard_id") or "") == str(dashboard_id):
            return dashboard
    return None


async def replace_session_dashboard_state(
    *,
    session_id: str,
    dashboard_state: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = _normalize_dashboard_state(dashboard_state)
    return await _publish_row(
        session_id=session_id,
        op="replace",
        dashboard=snapshot,
        patch=None,
        meta=meta or {},
    )


async def patch_session_dashboard_state(
    *,
    session_id: str,
    dashboard_id: str,
    revision: int | None = None,
    status: str | None = None,
    ops: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = await get_session_dashboard_state(session_id, dashboard_id)
    if current is None:
        current = {
            "dashboard_id": str(dashboard_id),
            "dashboard_type": "generic.dashboard",
            "workflow_id": "",
            "revision": 0,
            "status": "idle",
            "updated_at": _utc_now_iso(),
            "data": {},
        }
    current_data = dict(current.get("data") or {})
    next_revision = int(revision) if revision is not None else int(current.get("revision") or 0) + 1
    current["revision"] = next_revision
    if status is not None:
        current["status"] = str(status)
    current["updated_at"] = _utc_now_iso()
    current["data"] = _apply_patch_ops(current_data, ops)
    patch = {
        "dashboard_id": str(dashboard_id),
        "revision": next_revision,
        "status": str(current.get("status") or ""),
        "ops": [
            _normalize_op(op if isinstance(op, dict) else {}, fallback_index=index)
            for index, op in enumerate(ops or [])
        ],
    }
    return await _publish_row(
        session_id=session_id,
        op="patch",
        dashboard=current,
        patch=patch,
        meta=meta or {},
    )


async def clear_session_dashboard_state(
    *,
    session_id: str,
    dashboard_id: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return await _publish_row(
        session_id=session_id,
        op="clear",
        dashboard={
            "dashboard_id": str(dashboard_id),
            "dashboard_type": "generic.dashboard",
            "workflow_id": "",
            "revision": 0,
            "status": "cleared",
            "updated_at": _utc_now_iso(),
            "data": {},
        },
        patch=None,
        meta=meta or {},
    )

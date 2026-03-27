from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from aethergraph.core.runtime.runtime_services import current_services

WORK_STATUS_EVENT_KIND = "session_work_status"
WORK_STATUS_EVENT_TYPE = "work_status.updated"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _snapshot_ts() -> float:
    return datetime.now(UTC).timestamp()


def _normalize_item(item: dict[str, Any], *, fallback_order: int) -> dict[str, Any]:
    payload = dict(item or {})
    payload["id"] = str(payload.get("id") or "")
    payload["label"] = str(payload.get("label") or payload["id"] or "Work item")
    payload["kind"] = str(payload.get("kind") or "item")
    payload["status"] = str(payload.get("status") or "pending")
    payload["detail"] = str(payload.get("detail") or "")
    payload["order"] = int(
        payload.get("order") if payload.get("order") is not None else fallback_order
    )
    if payload.get("run_ref") is not None:
        payload["run_ref"] = dict(payload.get("run_ref") or {})
    if payload.get("artifact_ref") is not None:
        payload["artifact_ref"] = dict(payload.get("artifact_ref") or {})
    return payload


def _normalize_work_status(work_status: dict[str, Any] | None) -> dict[str, Any] | None:
    if not work_status:
        return None
    raw = dict(work_status)
    items = [
        _normalize_item(item if isinstance(item, dict) else {}, fallback_order=index)
        for index, item in enumerate(raw.get("items") or [])
    ]
    items.sort(key=lambda item: (int(item.get("order") or 0), str(item.get("id") or "")))
    return {
        "workflow_id": str(raw.get("workflow_id") or ""),
        "title": str(raw.get("title") or "Work Status"),
        "kind": str(raw.get("kind") or "workflow"),
        "status": str(raw.get("status") or "running"),
        "summary": str(raw.get("summary") or ""),
        "active_item_id": str(raw.get("active_item_id") or "") or None,
        "updated_at": str(raw.get("updated_at") or _utc_now_iso()),
        "items": items,
    }


def _apply_patch(
    current: dict[str, Any] | None,
    *,
    workflow_id: str | None,
    status: str | None,
    summary: str | None,
    active_item_id: str | None,
    item_updates: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    snapshot = _normalize_work_status(current) or {
        "workflow_id": str(workflow_id or ""),
        "title": "Work Status",
        "kind": "workflow",
        "status": "running",
        "summary": "",
        "active_item_id": None,
        "updated_at": _utc_now_iso(),
        "items": [],
    }
    if workflow_id is not None:
        snapshot["workflow_id"] = str(workflow_id)
    if status is not None:
        snapshot["status"] = str(status)
    if summary is not None:
        snapshot["summary"] = str(summary)
    if active_item_id is not None:
        snapshot["active_item_id"] = str(active_item_id) or None

    item_map = {str(item.get("id") or ""): dict(item) for item in snapshot.get("items") or []}
    for index, item_update in enumerate(item_updates or []):
        update_id = str((item_update or {}).get("id") or "")
        if not update_id:
            continue
        merged = dict(item_map.get(update_id) or {})
        merged.update(dict(item_update or {}))
        if "order" not in merged:
            merged["order"] = index
        item_map[update_id] = _normalize_item(
            merged, fallback_order=int(merged.get("order") or index)
        )
    snapshot["items"] = sorted(
        item_map.values(),
        key=lambda item: (int(item.get("order") or 0), str(item.get("id") or "")),
    )
    snapshot["updated_at"] = _utc_now_iso()
    return snapshot


async def _publish_row(
    *, session_id: str, op: str, work_status: dict[str, Any] | None, meta: dict[str, Any]
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
        "kind": WORK_STATUS_EVENT_KIND,
        "payload": {
            "type": WORK_STATUS_EVENT_TYPE,
            "op": op,
            "work_status": work_status,
            "meta": dict(meta or {}),
        },
    }
    await event_log.append(row)
    if event_hub is not None:
        await event_hub.broadcast(row)
    return row


async def get_session_work_status(session_id: str) -> dict[str, Any] | None:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        return None
    rows = await event_log.query(scope_id=session_id, kinds=[WORK_STATUS_EVENT_KIND], limit=500)
    if not rows:
        return None
    payload = dict((rows[-1] or {}).get("payload") or {})
    return _normalize_work_status(payload.get("work_status"))


async def replace_session_work_status(
    *,
    session_id: str,
    work_status: dict[str, Any],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = _normalize_work_status(work_status)
    return await _publish_row(
        session_id=session_id, op="replace", work_status=snapshot, meta=meta or {}
    )


async def patch_session_work_status(
    *,
    session_id: str,
    workflow_id: str | None = None,
    status: str | None = None,
    summary: str | None = None,
    active_item_id: str | None = None,
    item_updates: list[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = await get_session_work_status(session_id)
    snapshot = _apply_patch(
        current,
        workflow_id=workflow_id,
        status=status,
        summary=summary,
        active_item_id=active_item_id,
        item_updates=item_updates,
    )
    return await _publish_row(
        session_id=session_id, op="patch", work_status=snapshot, meta=meta or {}
    )


async def clear_session_work_status(
    *, session_id: str, meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    return await _publish_row(session_id=session_id, op="clear", work_status=None, meta=meta or {})

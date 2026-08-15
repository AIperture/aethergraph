from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aethergraph.services.state_stores.scope import scope_for_run_record


def build_error_info(
    payload: Any,
    fallback_message: str | None = None,
) -> dict[str, Any] | None:
    """Normalize a persisted run or node error without discarding detail."""
    if payload is None:
        if not fallback_message:
            return None
        payload = {"message": fallback_message}
    if isinstance(payload, dict):
        message = payload.get("message") or fallback_message
        detail = payload.get("detail")
        if not message and not detail:
            return None
        return {
            "message": message or "Run failed",
            "detail": detail,
            "kind": payload.get("kind"),
            "stage": payload.get("stage"),
            "code": payload.get("code"),
            "hints": list(payload.get("hints") or []),
            "is_traceback": bool(payload.get("is_traceback", False)),
        }
    if fallback_message or payload:
        return {
            "message": fallback_message or str(payload),
            "detail": None,
            "kind": None,
            "stage": None,
            "code": None,
            "hints": [],
            "is_traceback": False,
        }
    return None


@dataclass(frozen=True)
class RuntimeNodeDiagnostic:
    node_id: str
    tool_name: str | None
    status: str | None
    error: str | None
    error_info: dict[str, Any] | None


@dataclass(frozen=True)
class RuntimeInspection:
    record: Any
    nodes_state: dict[str, dict[str, Any]]
    snapshot_edges: tuple[dict[str, str], ...]
    run_error_info: dict[str, Any] | None
    node_diagnostics: tuple[RuntimeNodeDiagnostic, ...]


def _coerce_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except Exception:
            return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        except Exception:
            return None
    return None


def _is_terminal_node_status(value: Any) -> bool:
    return str(value or "").upper() in {
        "DONE",
        "FAILED",
        "CANCELLED",
        "CANCELED",
        "SKIPPED",
    }


def _apply_incremental_events(
    nodes_state: dict[str, dict[str, Any]],
    events: list[Any],
) -> dict[str, dict[str, Any]]:
    merged = {str(node_id): dict(state or {}) for node_id, state in nodes_state.items()}

    def _ensure_node(node_id: str) -> dict[str, Any]:
        return merged.setdefault(
            str(node_id),
            {
                "status": "PENDING",
                "started_at": None,
                "finished_at": None,
                "outputs": None,
                "error": None,
                "error_info": None,
            },
        )

    ordered = sorted(
        list(events or []),
        key=lambda event: (getattr(event, "rev", -1), getattr(event, "ts", 0.0)),
    )
    for event in ordered:
        kind = str(getattr(event, "kind", "") or "").upper()
        payload = getattr(event, "payload", None) or {}
        if not isinstance(payload, dict) or not payload.get("node_id"):
            continue
        node_state = _ensure_node(str(payload["node_id"]))

        if kind == "STATUS":
            status = payload.get("status")
            if status is None:
                continue
            node_state["status"] = status
            event_dt = _coerce_timestamp(getattr(event, "ts", None))
            event_iso = event_dt.isoformat() if event_dt is not None else None
            if str(status).upper() == "RUNNING" and not node_state.get("started_at"):
                node_state["started_at"] = event_iso
            if _is_terminal_node_status(status) and not node_state.get("finished_at"):
                node_state["finished_at"] = event_iso
        elif kind == "OUTPUT":
            node_state["outputs"] = payload.get("outputs")

    return merged


class RuntimeInspectionService:
    """Read the canonical run record and latest merged node state."""

    def __init__(self, *, run_manager: Any, state_store: Any = None):
        self.run_manager = run_manager
        self.state_store = state_store

    async def inspect(self, run_id: str) -> RuntimeInspection | None:
        record = await self.run_manager.get_record(run_id)
        if record is None:
            return None

        snapshot = None
        incremental_events: list[Any] = []
        if self.state_store is not None:
            scope = scope_for_run_record(record)
            snapshot = await self.state_store.load_latest_snapshot(scope, run_id)
            from_rev = getattr(snapshot, "rev", -1) if snapshot is not None else -1
            incremental_events = await self.state_store.load_events_since(scope, run_id, from_rev)

        nodes_state: dict[str, dict[str, Any]] = {}
        snapshot_edges: list[dict[str, str]] = []
        if snapshot is not None and isinstance(snapshot.state, dict):
            raw_nodes = snapshot.state.get("nodes") or snapshot.state.get("node_state") or {}
            if isinstance(raw_nodes, dict):
                nodes_state = {str(key): (value or {}) for key, value in raw_nodes.items()}

            raw_edges = snapshot.state.get("edges") or []
            if isinstance(raw_edges, list):
                snapshot_edges = [
                    {
                        "source": edge.get("from", edge.get("source")),
                        "target": edge.get("to", edge.get("target")),
                    }
                    for edge in raw_edges
                    if isinstance(edge, dict)
                    and (edge.get("from") or edge.get("source"))
                    and (edge.get("to") or edge.get("target"))
                ]

        if incremental_events:
            nodes_state = _apply_incremental_events(nodes_state, incremental_events)

        node_diagnostics = tuple(
            RuntimeNodeDiagnostic(
                node_id=node_id,
                tool_name=state.get("tool_name"),
                status=str(state.get("status")) if state.get("status") is not None else None,
                error=state.get("error"),
                error_info=build_error_info(state.get("error_info"), state.get("error")),
            )
            for node_id, state in nodes_state.items()
            if state.get("error") or state.get("error_info")
        )
        return RuntimeInspection(
            record=record,
            nodes_state=nodes_state,
            snapshot_edges=tuple(snapshot_edges),
            run_error_info=build_error_info(
                (record.meta or {}).get("error_info"),
                record.error,
            ),
            node_diagnostics=node_diagnostics,
        )

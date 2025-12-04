from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from aethergraph.contracts.services.state_stores import GraphSnapshot

from .serialize import state_to_json_safe


async def snapshot_from_graph(
    run_id: str,
    graph_id: str,
    rev: int,
    spec_hash: str,
    state_obj,
    *,
    artifacts=None,  # AsyncArtifactStore or None
    allow_externalize: bool = False,
    include_wait_spec: bool = True,
):
    json_state = await state_to_json_safe(
        state_obj,
        run_id=run_id,
        graph_id=graph_id,
        artifacts=artifacts,
        allow_externalize=allow_externalize,
        include_wait_spec=include_wait_spec,
    )
    snap = GraphSnapshot(
        run_id=run_id,
        graph_id=graph_id,
        rev=rev,
        created_at=datetime.utcnow().timestamp(),
        spec_hash=spec_hash,
        state=json_state,
    )
    return snap


def _status_to_str(s) -> str:
    if s is None:
        return "PENDING"
    if isinstance(s, Enum):
        return s.name
    # already a string or something printable
    return str(s)


def _enum_name_or_str(x):
    try:
        return x.name  # Enum
    except AttributeError:
        return str(x)


def _state_to_json(state_obj) -> dict[str, Any]:
    return {
        "run_id": getattr(state_obj, "run_id", None),
        "rev": getattr(state_obj, "rev", None),
        "patches": [asdict(p) if is_dataclass(p) else p for p in getattr(state_obj, "patches", [])],
        "_bound_inputs": getattr(state_obj, "_bound_inputs", None),
        "nodes": {
            nid: {
                "status": _enum_name_or_str(getattr(ns, "status", "PENDING")),
                "outputs": getattr(ns, "outputs", None),
                "error": getattr(ns, "error", None),
                "attempts": getattr(ns, "attempts", 0),
                "next_wakeup_at": getattr(ns, "next_wakeup_at", None),
                "wait_token": getattr(ns, "wait_token", None),
                # NEW: safe subset of wait_spec (avoid inline payload & tokens)
                "wait_spec": _sanitize_wait_spec(getattr(ns, "wait_spec", None)),
            }
            for nid, ns in getattr(state_obj, "nodes", {}).items()
        },
    }


def _sanitize_wait_spec(ws):
    if not ws:
        return None
    # Drop volatile/sensitive fields if present
    return {
        "kind": ws.get("kind"),
        "channel": ws.get("channel"),
        "prompt": ws.get("prompt"),
        "options": ws.get("options"),
        "meta": ws.get("meta") or {},
        # DO NOT store: token / inline_payload / resume_schema with secrets
    }

# aethergraph/runtime/recovery.py
from __future__ import annotations

import datetime
import hashlib
import time
from typing import Any

from aethergraph.contracts.services.state_stores import GraphStateStore

from ..graph.node_state import NodeStatus
from ..graph.task_graph import TaskGraph, TaskGraphSpec


def hash_spec(spec: TaskGraphSpec) -> str:
    import json

    # stable hash of the immutable parts
    raw = json.dumps(
        {
            "graph_id": spec.graph_id,
            "version": spec.version,
            "nodes": {
                nid: {
                    "type": ns.type,
                    "dependencies": ns.dependencies,
                    "logic": ns.logic if isinstance(ns.logic, str) else str(ns.logic),
                    "metadata": ns.metadata,
                }
                for nid, ns in spec.nodes.items()
            },
            "io": {
                "required": sorted(list(spec.io.required.keys())),
                "optional": sorted(list(spec.io.optional.keys())),
                "outputs": sorted(list(spec.io.outputs.keys())),
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def recover_graph_run(
    *,
    spec: TaskGraphSpec,
    run_id: str,
    store: GraphStateStore,
) -> TaskGraph:
    snap = await store.load_latest_snapshot(run_id)
    g = TaskGraph.from_spec(spec=spec, state=None)
    g.state.run_id = run_id
    # If no snapshot, we're starting fresh.
    if not snap:
        return g

    # Basic drift guard (optional: warn if different)
    want = hash_spec(spec)
    if snap.spec_hash != want:
        # Soft warning; TODO: raise if later want strictness.
        import logging

        logger = logging.getLogger("aethergraph.core.runtime.recovery")
        logger.warning(
            f"[recover_graph_run] Spec hash mismatch for run {run_id}: snapshot has {snap.spec_hash}, want {want}"
        )

    # Apply snapshot state
    _hydrate_state_from_json(g, snap.state)

    return g


def _hydrate_state_from_json(graph, j: dict[str, Any]) -> None:
    graph.state.rev = j.get("rev", 0)
    graph.state._bound_inputs = j.get("_bound_inputs")
    for nid, ns_json in j.get("nodes", {}).items():
        ns = graph.state.nodes.setdefault(nid, graph.state.nodes.get(nid))
        status_name = ns_json.get("status", "PENDING")
        status = getattr(NodeStatus, status_name, NodeStatus.PENDING)
        if status == NodeStatus.RUNNING:
            status = NodeStatus.PENDING
        ns.status = status

        outs = ns_json.get("outputs") or {}
        # Keep as-is; resume_policy already blocked non-JSON/ref earlier
        ns.outputs = outs


async def rearm_waits_if_needed(graph, env, *, ttl_s: int = 3600):
    store = env.container.cont_store
    bus = env.container.channels
    now = time.time()

    for nid, ns in graph.state.nodes.items():
        if getattr(ns, "status", None) not in (
            NodeStatus.WAITING_HUMAN,
            getattr(NodeStatus, "WAITING_EXTERNAL", "WAITING_EXTERNAL"),
        ):
            continue

        cont = await store.get(run_id=env.run_id, node_id=nid)
        # Normalize deadline to a numeric timestamp to avoid comparing datetime with float
        deadline = getattr(cont, "deadline", None)
        deadline_ts = deadline.timestamp() if isinstance(deadline, datetime.datetime) else deadline
        expired = (not cont) or (deadline_ts is not None and deadline_ts < now)

        if not expired:
            continue  # still valid

        # Rebuild OutEvent from saved wait_spec
        ws = getattr(ns, "wait_spec", None)
        if not ws:
            # No spec → safest fallback is to keep waiting but log it
            env.container.logger.for_run().warning(
                f"[rearm] missing wait_spec for {env.run_id}:{nid}; staying WAITING"
            )
            continue

        # Mint a new continuation token
        new_deadline = now + ttl_s
        token = store.mint(
            run_id=env.run_id,
            node_id=nid,
            kind=ws["kind"],
            channel=ws.get("channel"),
            deadline=new_deadline,
            meta=ws.get("meta") or {},
        )
        # Build + send OutEvent
        out = {
            "type": "session.need_input"
            if ws["kind"] == "text"
            else "session.need_approval"
            if ws["kind"] == "approval"
            else "session.need_input",  # default
            "channel": ws.get("channel"),
            "text": ws.get("prompt"),
            "buttons": [{"label": o} for o in (ws.get("options") or [])],
            "meta": ws.get("meta") or {},
        }
        payload = await bus.send(out)  # may inline-resume for console/web

        # If adapter returned a payload immediately → deliver inline
        if payload and "payload" in payload:
            # inline path (same as in _enter_wait)
            await env.container.resume_bus.deliver_inline(
                run_id=env.run_id, node_id=nid, payload=payload["payload"]
            )
        else:
            # Persist (replace/insert) the new continuation
            store.save_for_node(
                run_id=env.run_id,
                node_id=nid,
                token=token,
                kind=ws["kind"],
                channel=ws.get("channel"),
                deadline=new_deadline,
                meta=ws.get("meta") or {},
            )

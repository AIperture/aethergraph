# aethergraph/runtime/recovery.py
from __future__ import annotations

import hashlib
from typing import Any

from aethergraph.contracts.services.state_stores import GraphStateStore

from ..graph.node_state import NodeStatus
from ..graph.task_graph import TaskGraph, TaskGraphSpec


def hash_spec(spec: TaskGraphSpec) -> str:
    """Hash the immutable structure of one task-graph specification.

    Intro:
        Produces a stable SHA-256 digest used to detect graph-definition drift
        before hydrating persisted runtime state.

    Examples:
        Hash a compiled task graph:
        ```python
        digest = hash_spec(graph.spec)
        ```

        Compare a snapshot contract:
        ```python
        if snapshot.spec_hash != hash_spec(graph.spec):
            print("graph definition changed")
        ```

    Args:
        spec: Task-graph specification whose immutable structure is hashed.

    Returns:
        str: Lowercase SHA-256 hexadecimal digest.

    Notes:
        Callable node logic is represented by its stable string form because
        executable objects are not serialized into snapshots.
    """
    import json

    raw = json.dumps(
        {
            "graph_id": spec.graph_id,
            "agent_id": spec.agent_id or "",
            "app_id": spec.app_id or "",
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
    """Rehydrate a task graph from its latest durable snapshot.

    Intro:
        Materializes the supplied specification, loads the latest run snapshot,
        warns on definition drift, and applies persisted node state when present.

    Examples:
        Recover an interrupted run:
        ```python
        graph = await recover_graph_run(spec=spec, run_id="run-1", store=state_store)
        ```

        Start from a clean graph when no snapshot exists:
        ```python
        graph = await recover_graph_run(spec=spec, run_id="new-run", store=state_store)
        assert graph.state.run_id == "new-run"
        ```

    Args:
        spec: Canonical task-graph specification to materialize.
        run_id: Exact durable run identity whose snapshot is loaded.
        store: Graph-state store providing the latest snapshot.

    Returns:
        TaskGraph: Materialized graph with hydrated state when available.

    Notes:
        Persisted `RUNNING` nodes become `PENDING` so interrupted work can be
        scheduled again; specification drift currently emits a warning.
    """
    snap = await store.load_latest_snapshot(run_id)
    graph = TaskGraph.from_spec(spec=spec, state=None)
    graph.state.run_id = run_id
    if not snap:
        return graph

    expected_hash = hash_spec(spec)
    if snap.spec_hash != expected_hash:
        import logging

        logger = logging.getLogger("aethergraph.core.runtime.recovery")
        logger.warning(
            "[recover_graph_run] Spec hash mismatch for run %s: snapshot has %s..., "
            "want %s... This typically means the graph definition changed since "
            "the snapshot was taken.",
            run_id,
            snap.spec_hash[:8],
            expected_hash[:8],
        )

    try:
        _hydrate_state_from_json(graph, snap.state)
    except Exception:
        import logging

        logging.getLogger("aethergraph.core.runtime.recovery").exception(
            "[recover_graph_run] Failed to hydrate state for run %s",
            run_id,
        )

    return graph


def _hydrate_state_from_json(graph, payload: dict[str, Any]) -> None:
    graph.state.rev = payload.get("rev", 0)
    graph.state._bound_inputs = payload.get("_bound_inputs")
    for node_id, node_payload in payload.get("nodes", {}).items():
        node_state = graph.state.nodes.setdefault(node_id, graph.state.nodes.get(node_id))
        status_name = node_payload.get("status", "PENDING")
        status = getattr(NodeStatus, status_name, NodeStatus.PENDING)
        if status == NodeStatus.RUNNING:
            status = NodeStatus.PENDING
        node_state.status = status
        node_state.outputs = node_payload.get("outputs") or {}
        node_state.started_at = node_payload.get("started_at")
        node_state.finished_at = node_payload.get("finished_at")

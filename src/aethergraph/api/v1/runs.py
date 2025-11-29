# /runs

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from aethergraph.core.runtime.runtime_services import current_services

from .deps import RequestIdentity, get_identity
from .schemas import (
    NodeSnapshot,
    RunCreateRequest,
    RunCreateResponse,
    RunListResponse,
    RunSnapshot,
    RunStatus,
    RunSummary,
)

router = APIRouter(tags=["runs"])


@router.post("/graphs/{graph_id}/runs", response_model=RunCreateResponse)
async def create_run(
    graph_id: str,
    body: RunCreateRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunCreateResponse:
    container = current_services()
    rm = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    record = await rm.submit_run(
        graph_id=graph_id,
        inputs=body.inputs or {},
        run_id=body.run_id,
        tags=body.tags,
        user_id=identity.user_id,
        org_id=identity.org_id,
    )

    return RunCreateResponse(
        run_id=record.run_id,
        graph_id=record.graph_id,
        status=record.status,  # typically "running"
        outputs=None,
        has_waits=False,  # for now, we don't expose waits on submit
        continuations=[],
        started_at=record.started_at,
        finished_at=record.finished_at,
    )


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunSummary:
    """
    Get high-level summary for a run from RunStore.
    """
    container = current_services()
    rm = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    rec = await rm.get_record(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunSummary(
        run_id=rec.run_id,
        graph_id=rec.graph_id,
        status=rec.status,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        tags=rec.tags,
        user_id=rec.user_id,
        org_id=rec.org_id,
    )


# --------- TODO: list runs, get snapshot, cancel run ---------


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    graph_id: str | None = Query(None),  # noqa: B008
    status: RunStatus | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(20, ge=1, le=100),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunListResponse:
    """
    List recent runs, filterable by graph_id/status.
    Should go through container.run_manager.list_records(...) → which uses run_store under the hood.
    TODO:
      - Integrate with your run store.
      - Use `cursor`/`limit` for real pagination.
      - Filter by identity.user_id/org_id as needed.
    """
    now = datetime.utcnow()
    dummy_run = RunSummary(
        run_id="run-123",
        graph_id=graph_id or "example_graph",
        status=RunStatus("succeeded"),
        created_at=now - timedelta(minutes=5),
        started_at=now - timedelta(minutes=4),
        finished_at=now - timedelta(minutes=1),
        tags=["stub"],
        user_id=identity.user_id,
        org_id=identity.org_id,
    )
    return RunListResponse(runs=[dummy_run], next_cursor=None)


# @router.get("/runs/{run_id}/snapshot", response_model=RunSnapshot)
# async def get_run_snapshot(
#     run_id: str,
#     identity: RequestIdentity = Depends(get_identity),  # noqa: B008
# ) -> RunSnapshot:
#     """
#     Get DAG snapshot: nodes + edges with statuses.
#     Should use container.state_store
#     TODO:
#       - Wire to runtime's per-node state.
#     """
#     now = datetime.utcnow()
#     node = NodeSnapshot(
#         node_id="n1",
#         tool_name="stub_tool",
#         status=RunStatus.running,
#         started_at=now,
#         finished_at=None,
#         outputs=None,
#         error=None,
#     )
#     return RunSnapshot(
#         run_id=run_id,
#         graph_id="example_graph",
#         nodes=[node],
#         edges=[],
#     )


# @router.get("/runs/{run_id}/snapshot", response_model=RunSnapshot)
# async def get_run_snapshot(
#     run_id: str,
#     identity: RequestIdentity = Depends(get_identity),  # noqa: B008
# ) -> RunSnapshot:
#     """
#     Synthetic DAG snapshot for a run.

#     For now:
#       - Look up the Run record to get graph_id + status.
#       - Look up the TaskGraph spec (if it's a graphify graph).
#       - Build NodeSnapshot for each node in the DAG.
#       - Use the run status to fake per-node statuses.
#     """

#     container = current_services()
#     rm = getattr(container, "run_manager", None)
#     if rm is None:
#         raise HTTPException(status_code=503, detail="Run manager not configured")

#     rec = await rm.get_record(run_id)
#     if rec is None:
#         raise HTTPException(status_code=404, detail="Run not found")

#     graph_id = rec.graph_id

#     # Try to load a static TaskGraph; if not available, just return a minimal snapshot.
#     reg = container.registry  # or current_registry() if you use that
#     try:
#         graph_obj = reg.get_graph(name=graph_id, version=None)
#         spec = getattr(graph_obj, "spec", None)
#     except KeyError:
#         spec = None

#     # If we have no spec, just mirror the run status as a single pseudo-node
#     if spec is None:
#         node = NodeSnapshot(
#             node_id="run",
#             tool_name=None,
#             status=rec.status,
#             started_at=rec.started_at,
#             finished_at=rec.finished_at,
#             outputs=None,
#             error=None,
#         )
#         return RunSnapshot(
#             run_id=rec.run_id,
#             graph_id=graph_id,
#             nodes=[node],
#             edges=[],
#         )

#     # --- Build nodes + edges from TaskGraphSpec ---

#     # Simple heuristic: timing offsets so the timeline looks nice
#     base_start = rec.started_at or datetime.utcnow()
#     total_nodes = len(spec.nodes) or 1
#     dt = timedelta(seconds=3)

#     nodes: list[NodeSnapshot] = []
#     edges: list[dict[str, str]] = []

#     # Build edges from dependencies
#     edge_set: set[tuple[str, str]] = set()
#     for node_id, node_spec in spec.nodes.items():
#         for dep_id in node_spec.dependencies:
#             edge_set.add((dep_id, node_id))
#     edges = [{"from": src, "to": dst} for (src, dst) in sorted(edge_set)]

#     # Synthetic per-node status based on overall run status
#     for idx, (node_id, node_spec) in enumerate(spec.nodes.items()):
#         # naive phase-based status
#         if rec.status in (RunStatus.succeeded, RunStatus.failed, RunStatus.canceled):
#             node_status = rec.status
#         elif rec.status in (RunStatus.running, RunStatus.cancellation_requested):
#             if idx == 0:
#                 node_status = RunStatus.running
#             else:
#                 node_status = RunStatus.pending
#         else:
#             node_status = RunStatus.pending

#         started_at = base_start + dt * idx if rec.started_at else None
#         finished_at = (
#             started_at + dt if node_status in (RunStatus.succeeded, RunStatus.failed) else None
#         )

#         nodes.append(
#             NodeSnapshot(
#                 node_id=node_id,
#                 tool_name=node_spec.tool_name,
#                 status=node_status,
#                 started_at=started_at,
#                 finished_at=finished_at,
#                 outputs=None,
#                 error=None,
#             )
#         )

#     return RunSnapshot(
#         run_id=rec.run_id,
#         graph_id=graph_id,
#         nodes=nodes,
#         edges=edges,
#     )


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    """
    Request run cancellation.

    TODO:
      - Call runtime/cancellation mechanism.
    """
    return {"run_id": run_id, "status": "cancellation_requested"}


def _coerce_ts_to_dt(value: Any) -> datetime | None:
    """
    Accepts:
      - None
      - datetime
      - float / int epoch seconds
      - ISO8601 string
    Returns timezone-aware UTC datetime or None.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        # Ensure it's tz-aware; default to UTC if naive.
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    # Epoch seconds (int/float)
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None

    # ISO string
    if isinstance(value, str):
        try:
            # If you have dateutil, you can use that; otherwise fromisoformat
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None

    return None


def _coerce_node_status(value: Any, fallback: RunStatus) -> RunStatus:
    """
    Try to convert arbitrary value to RunStatus, else use fallback.
    """
    if isinstance(value, RunStatus):
        return value
    if isinstance(value, str):
        try:
            if value == "DONE":
                return RunStatus.succeeded
            if value == "FAILED":
                return RunStatus.failed
            if value == "CANCELLED":
                return RunStatus.canceled
            if value == "PENDING":
                return RunStatus.pending
            return RunStatus(value)
        except ValueError:
            # maybe uppercased, etc.
            try:
                return RunStatus(value.lower())
            except Exception:
                pass
    return fallback


@router.get("/runs/{run_id}/snapshot", response_model=RunSnapshot)
async def get_run_snapshot(
    run_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunSnapshot:
    """
    Synthetic DAG snapshot for a run.

    For now:
      - Look up the Run record to get graph_id + status.
      - Look up the TaskGraph spec (if it's a graphify graph).
      - Load latest GraphSnapshot from state_store, if available.
      - Build NodeSnapshot for each node using real per-node state where possible.
      - Fallback to heuristic behavior when we don't have detailed state.
    """
    container = current_services()

    rm = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    state_store = getattr(container, "state_store", None)

    rec = await rm.get_record(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")

    graph_id = rec.graph_id

    # --- Load static TaskGraph spec if it exists ---
    reg = getattr(container, "registry", None)
    if reg is None:
        from aethergraph.core.runtime.runtime_registry import current_registry

        reg = current_registry()

    spec = None
    try:
        graph_obj = reg.get_graph(name=graph_id, version=None)
        spec = getattr(graph_obj, "spec", None)
    except KeyError:
        spec = None

    # --- Load latest GraphSnapshot (if we have a state store) ---

    snap = None
    if state_store is not None:
        snap = await state_store.load_latest_snapshot(run_id)

    # Extract node-level state + edges from snapshot if present.
    nodes_state: dict[str, dict[str, Any]] = {}
    snapshot_edges: list[dict[str, str]] = []

    if snap is not None and isinstance(snap.state, dict):
        raw_nodes = snap.state.get("nodes") or snap.state.get("node_state") or {}
        if isinstance(raw_nodes, dict):
            # node_id -> dict
            nodes_state = {str(k): (v or {}) for k, v in raw_nodes.items()}

        raw_edges = snap.state.get("edges") or []
        if isinstance(raw_edges, list):
            snapshot_edges = [
                {"source": e.get("from"), "target": e.get("to")}
                for e in raw_edges
                if isinstance(e, dict) and "from" in e and "to" in e
            ]

    # --- Build edges ---

    edges: list[dict[str, str]] = []

    if snapshot_edges:
        # Prefer edges from snapshot for dynamic / graphfn graphs
        edges = snapshot_edges
    elif spec is not None and getattr(spec, "nodes", None):
        # Static topology from TaskGraphSpec
        edge_set: set[tuple[str, str]] = set()
        for node_id, node_spec in spec.nodes.items():
            for dep_id in getattr(node_spec, "dependencies", []):
                edge_set.add((str(dep_id), str(node_id)))
        edges = [{"source": src, "target": dst} for (src, dst) in sorted(edge_set)]

    nodes: list[NodeSnapshot] = []
    # --- Case 1: we have a TaskGraph spec (static graph) ---

    if spec is not None and getattr(spec, "nodes", None):
        for node_id, node_spec in spec.nodes.items():
            node_id_str = str(node_id)
            st = nodes_state.get(node_id_str, {})

            node_status = _coerce_node_status(st.get("status"), fallback=rec.status)

            started_at = _coerce_ts_to_dt(st.get("started_at"))
            finished_at = _coerce_ts_to_dt(st.get("finished_at"))
            outputs = st.get("outputs")
            error = st.get("error")

            nodes.append(
                NodeSnapshot(
                    node_id=node_id_str,
                    tool_name=getattr(node_spec, "tool_name", None),
                    status=node_status,
                    started_at=started_at,
                    finished_at=finished_at,
                    outputs=outputs,
                    error=error,
                )
            )

        return RunSnapshot(
            run_id=rec.run_id,
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
        )

    # --- Case 2: spec is missing, but snapshot has node info (e.g. graphfn/dynamic) ---

    if nodes_state:
        for node_id, st in nodes_state.items():
            node_status = _coerce_node_status(st.get("status"), fallback=rec.status)
            started_at = _coerce_ts_to_dt(st.get("started_at"))
            finished_at = _coerce_ts_to_dt(st.get("finished_at"))
            outputs = st.get("outputs")
            error = st.get("error")

            nodes.append(
                NodeSnapshot(
                    node_id=str(node_id),
                    tool_name=st.get("tool_name"),  # let snapshot override if present
                    status=node_status,
                    started_at=started_at,
                    finished_at=finished_at,
                    outputs=outputs,
                    error=error,
                )
            )

        return RunSnapshot(
            run_id=rec.run_id,
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,  # may be empty or from snapshot
        )

    # --- Case 3: no spec and no snapshot → fallback to pseudo-node (old behavior) ---

    node = NodeSnapshot(
        node_id="run",
        tool_name=None,
        status=rec.status,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        outputs=None,
        error=rec.error,
    )
    return RunSnapshot(
        run_id=rec.run_id,
        graph_id=graph_id,
        nodes=[node],
        edges=[],
    )

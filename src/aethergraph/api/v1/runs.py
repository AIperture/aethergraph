# /runs

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import current_services

from .deps import RequestIdentity, get_identity
from .schemas import (
    NodeSnapshot,
    RunChannelEvent,
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

    # --- Graph metadata from registry ---
    reg = getattr(container, "registry", None) or current_registry()
    flow_id: str | None = None
    entrypoint = False

    if reg is not None:
        if rec.kind == "taskgraph":
            meta = reg.get_meta(nspace="graph", name=rec.graph_id, version=None) or {}
        elif rec.kind == "graphfn":
            meta = reg.get_meta(nspace="graphfn", name=rec.graph_id, version=None) or {}
        else:
            meta = {}

        flow_id = meta.get("flow_id")
        entrypoint = bool(meta.get("entrypoint", False))

    return RunSummary(
        run_id=rec.run_id,
        graph_id=rec.graph_id,
        status=rec.status,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        tags=rec.tags,
        user_id=rec.user_id,
        org_id=rec.org_id,
        graph_kind=rec.kind,  # "taskgraph" | "graphfn" | "other"
        flow_id=flow_id,
        entrypoint=entrypoint,
    )


# --------- TODO: list runs, get snapshot, cancel run ---------


@router.get("/runs", response_model=RunListResponse)
async def list_runs(
    graph_id: str | None = Query(None),  # noqa: B008
    status: RunStatus | None = Query(None),  # noqa: B008
    flow_id: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008 (unused for now)
    limit: int = Query(20, ge=1, le=100),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunListResponse:
    """
    List recent runs, optionally filterable by graph_id, status, or flow_id.
    """
    container = current_services()
    rm = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    records = await rm.list_records(
        graph_id=graph_id,
        status=status,
        flow_id=flow_id,
        limit=limit,
    )

    reg = getattr(container, "registry", None) or current_registry()
    summaries: list[RunSummary] = []

    for rec in records:
        # Same graph metadata logic as in get_run
        flow_meta_id: str | None = None
        entrypoint = False
        if reg is not None:
            if rec.kind == "taskgraph":
                meta = reg.get_meta(nspace="graph", name=rec.graph_id, version=None) or {}
            elif rec.kind == "graphfn":
                meta = reg.get_meta(nspace="graphfn", name=rec.graph_id, version=None) or {}
            else:
                meta = {}
            flow_meta_id = meta.get("flow_id")
            entrypoint = bool(meta.get("entrypoint", False))

        # Prefer the run meta flow_id if present
        effective_flow_id = rec.meta.get("flow_id") or flow_meta_id

        summaries.append(
            RunSummary(
                run_id=rec.run_id,
                graph_id=rec.graph_id,
                status=rec.status,
                started_at=rec.started_at,
                finished_at=rec.finished_at,
                tags=rec.tags,
                user_id=rec.user_id,
                org_id=rec.org_id,
                graph_kind=rec.kind,
                flow_id=effective_flow_id,
                entrypoint=entrypoint,
            )
        )

    return RunListResponse(runs=summaries, next_cursor=None)


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
    Run snapshot for a single graph within this run.

    - Uses RunRecord for run-level status.
    - Uses registry metadata for graph_kind, flow_id, entrypoint.
    - Uses state_store (if present) for node-level state.
    - Falls back to TaskGraphSpec or a single pseudo-node.
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
    graph_kind = rec.kind

    # --- Graph metadata from registry ---
    reg = getattr(container, "registry", None) or current_registry()

    flow_id: str | None = None
    entrypoint = False
    meta = {}

    if reg is not None:
        if graph_kind == "taskgraph":
            meta = reg.get_meta(nspace="graph", name=graph_id, version=None) or {}
        elif graph_kind == "graphfn":
            meta = reg.get_meta(nspace="graphfn", name=graph_id, version=None) or {}

        flow_id = meta.get("flow_id")
        entrypoint = bool(meta.get("entrypoint", False))

    # --- Load static TaskGraph spec if it exists ---
    spec = None
    if reg is not None:
        try:
            graph_obj = reg.get_graph(name=graph_id, version=None)
            spec = getattr(graph_obj, "spec", None)
        except KeyError:
            spec = None

    # --- Load latest GraphSnapshot (if we have a state store) ---
    snap = None
    if state_store is not None:
        snap = await state_store.load_latest_snapshot(run_id)

    nodes_state: dict[str, dict[str, Any]] = {}
    snapshot_edges: list[dict[str, str]] = []

    if snap is not None and isinstance(snap.state, dict):
        raw_nodes = snap.state.get("nodes") or snap.state.get("node_state") or {}
        if isinstance(raw_nodes, dict):
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
        edges = snapshot_edges
    elif spec is not None and getattr(spec, "nodes", None):
        edge_set: set[tuple[str, str]] = set()
        for node_id, node_spec in spec.nodes.items():
            for dep_id in getattr(node_spec, "dependencies", []):
                edge_set.add((str(dep_id), str(node_id)))
        edges = [{"source": src, "target": dst} for (src, dst) in sorted(edge_set)]

    nodes: list[NodeSnapshot] = []

    # --- Case 1: TaskGraph with spec (static graph) ---
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
            graph_kind=graph_kind,
            flow_id=flow_id,
            entrypoint=entrypoint,
        )

    # --- Case 2: no spec, but snapshot has nodes (graphfn / dynamic) ---
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
                    tool_name=st.get("tool_name"),
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
            graph_kind=graph_kind,
            flow_id=flow_id,
            entrypoint=entrypoint,
        )

    # --- Case 3: no spec, no snapshot → single pseudo-node, each node is the graph itself---
    node = NodeSnapshot(
        node_id=graph_id,
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
        graph_kind=graph_kind,
        flow_id=flow_id,
        entrypoint=entrypoint,
    )


@router.get("/runs/{run_id}/channel/events", response_model=list[RunChannelEvent])
async def get_run_channel_events(
    run_id: str,
    request: Request,
    since_ts: float | None = None,
):
    """
    Fetch normalized UI channel events for a run.

    Frontend can poll with since_ts for incremental updates.
    """
    try:
        container = request.app.state.container
        event_log = container.eventlog  # eventlog for channels or equivalent

        since_dt = None
        if since_ts is not None:
            since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)

        # EventLog.query signature may differ; adapt as needed.
        # We assume it filters by scope_id and time range.
        rows = await event_log.query(
            scope_id=f"run-ui:{run_id}",
            since=since_dt,
            until=None,
        )

        # Each row is the dict we appended in the adapter; ensure fields exist
        events: list[RunChannelEvent] = []
        for row in rows:
            events.append(
                RunChannelEvent(
                    id=row.get("id"),
                    run_id=row.get("run_id", run_id),
                    type=row.get("type"),
                    text=row.get("text"),
                    buttons=row.get("buttons") or [],
                    file=row.get("file"),
                    meta=row.get("meta") or {},
                    ts=row.get("ts"),
                )
            )

        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

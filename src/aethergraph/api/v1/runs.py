# /runs

from datetime import datetime, timedelta

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
    """
    Start (or cold-resume) a run for a given graph.

    - body.run_id None → fresh run
    - body.run_id set  → run_or_resume against that run_id
    """
    container = current_services()
    rm = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")

    record, outputs, has_waits, continuations = await rm.start_run(
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
        status=record.status,
        outputs=outputs,
        has_waits=has_waits,
        continuations=continuations,
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


@router.get("/runs/{run_id}/snapshot", response_model=RunSnapshot)
async def get_run_snapshot(
    run_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunSnapshot:
    """
    Get DAG snapshot: nodes + edges with statuses.
    Should use container.state_store
    TODO:
      - Wire to runtime's per-node state.
    """
    now = datetime.utcnow()
    node = NodeSnapshot(
        node_id="n1",
        tool_name="stub_tool",
        status=RunStatus.running,
        started_at=now,
        finished_at=None,
        outputs=None,
        error=None,
    )
    return RunSnapshot(
        run_id=run_id,
        graph_id="example_graph",
        nodes=[node],
        edges=[],
    )


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

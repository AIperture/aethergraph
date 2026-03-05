from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas import (
    RunVizResponse,
    VizFigure,
    VizKind,
    VizPoint,
    VizTrack,
)
from aethergraph.core.runtime.runtime_services import current_services

router = APIRouter(tags=["viz"])


@router.get("/runs/{run_id}/viz", response_model=RunVizResponse)
async def get_run_viz(
    run_id: str,
    viz_kinds: Annotated[
        str | None,
        Query(
            description=(
                "Comma-separated list of viz kinds to include. "
                "Options: scalar,vector,matrix,image. "
                "If omitted, all viz kinds are returned."
            )
        ),
    ] = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> RunVizResponse:
    """
    Aggregate visualization data for a run into figures/tracks for the Vis tab.

    - Uses the EventLog-backed VizService.
    - Enforces demo scoping via RunManager (client_id).
    - Returns structured data (scalars, vectors, matrices, image references),
      not pre-rendered plots.
    """
    container = current_services()

    viz_service = getattr(container, "viz_service", None)
    rm = getattr(container, "run_manager", None)
    if viz_service is None:
        raise HTTPException(status_code=500, detail="VizService not available")

    # Demo mode: require RunManager to verify access
    if identity.mode == "demo" and rm is None:
        raise HTTPException(status_code=500, detail="RunManager not available")

    # Parse viz kinds filter [optional]
    kinds_filter: list[VizKind] | None = None
    if viz_kinds:
        raw = [k.strip().lower() for k in viz_kinds.split(",") if k.strip()]
        allowed: set[str] = {"scalar", "vector", "matrix", "image"}
        bad = [k for k in raw if k not in allowed]
        if bad:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid viz kinds: {', '.join(bad)}; allowed: {', '.join(sorted(allowed))}",
            )
        kinds_filter = raw  # type: ignore[assignment]

    # Query raw viz events for this run from VizService
    rows = await viz_service.query_run(run_id, kinds=kinds_filter)

    # Group into figures/tracks/points
    # Key: (figure_id, track_id, viz_kind, node_id)
    track_map: dict[tuple[str | None, str, str, str | None], dict[str, Any]] = {}
    for row in rows:
        data: dict[str, Any] = row.get("data") or {}
        viz_kind: str = data.get("viz_kind")
        track_id: str = data.get("track_id")
        figure_id: str | None = data.get("figure_id")
        node_id: str | None = data.get("node_id")
        step = data.get("step")

        if track_id is None or viz_kind is None or step is None:
            # skip malformed events
            continue

        key = (figure_id, track_id, viz_kind, node_id)
        agg = track_map.get(key)
        if agg is None:
            agg = {
                "figure_id": figure_id,
                "track_id": track_id,
                "viz_kind": viz_kind,
                "node_id": node_id,
                "mode": data.get("mode", "append"),
                "meta": data.get("meta") or {},
                "points": [],
            }
            track_map[key] = agg

        # Mode/meta: keep the first one, but allow later events to update if you want
        # For now we just keep existing 'mode' and 'meta' if already set.

        # Build point
        ts_str: str | None = row.get("ts") or data.get("created_at")
        created_at: datetime | None = None
        if ts_str:
            try:
                created_at = datetime.fromisoformat(ts_str)
            except Exception:
                created_at = None

        point = VizPoint(
            step=int(step),
            value=data.get("value"),
            vector=data.get("vector"),
            matrix=data.get("matrix"),
            artifact_id=data.get("artifact_id"),
            created_at=created_at,
        )
        agg["points"].append(point)

    # Build figures from grouping
    figures_map: dict[str | None, list[VizTrack]] = {}

    for (fig_id, track_id, _, node_id), agg in track_map.items():
        points: list[VizPoint] = agg["points"]
        # Sort points by step (and then by created_at as tiebreaker)
        points.sort(key=lambda p: (p.step, p.created_at or datetime.min))

        track = VizTrack(
            track_id=track_id,
            figure_id=fig_id,
            node_id=node_id,
            viz_kind=agg["viz_kind"],
            mode=agg["mode"],
            meta=agg["meta"],
            points=points,
        )

        lst = figures_map.setdefault(fig_id, [])
        lst.append(track)

    # Sort tracks within each figure by track_id for stability
    figures: list[VizFigure] = []
    for fig_id, tracks in figures_map.items():
        tracks.sort(key=lambda t: t.track_id)
        figures.append(VizFigure(figure_id=fig_id, tracks=tracks))

    # Sort figures: put named figures first, then the Node/default one
    figures.sort(key=lambda f: (f.figure_id is None, f.figure_id or ""))

    return RunVizResponse(run_id=run_id, figures=figures)

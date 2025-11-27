from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from aethergraph.core.runtime.runtime_services import current_services

from .deps import RequestIdentity, get_identity
from .schemas import (
    ArtifactStats,
    GraphStats,
    GraphStatsEntry,
    LLMStats,
    MemoryStats,
    StatsOverview,
)

router = APIRouter(tags=["stats"])


@router.get("/stats/overview", response_model=StatsOverview)
async def get_stats_overview(
    window: Annotated[str, Query(description="Time window for stats, e.g., '24h', '7d'")] = "24h",
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> StatsOverview:
    """
    Get an overview of usage statistics.

    - **window**: Time window for stats (e.g., "24h", "7d").
    """
    container = current_services()
    meter = getattr(container, "metering", None)
    if meter is None:
        raise HTTPException(status_code=501, detail="Metering service not available")

    raw: dict[str, int] = await meter.get_overview(
        user_id=identity.user_id if identity and identity.user_id else None,
        org_id=identity.org_id if identity and identity.org_id else None,
        window=window,
    )
    return StatsOverview(**raw)


@router.get("/stats/graphs", response_model=GraphStats)
async def get_graphs_stats(
    window: Annotated[str, Query(description="Time window for stats, e.g., '24h', '7d'")] = "24h",
    graph_id: Annotated[str | None, Query(description="Optional graph_id filter")] = None,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> GraphStats:
    """
    Get usage statistics for graphs.

    - **window**: Time window for stats (e.g., "24h", "7d").
    - **graph_id**: Optional filter; if provided, only stats for that graph are returned.
    """
    container = current_services()
    meter = getattr(container, "metering", None)
    if meter is None:
        raise HTTPException(status_code=501, detail="Metering service not available")

    raw_all: dict[str, dict[str, Any]] = await meter.get_graph_stats(
        user_id=identity.user_id if identity and identity.user_id else None,
        org_id=identity.org_id if identity and identity.org_id else None,
        window=window,
    )
    # raw_all: { "<graph_id>": {"runs":..., "succeeded":..., "failed":..., "total_duration_s":...}, ... }

    if graph_id is not None:
        # Return only the requested graph, but still in map form
        entry = raw_all.get(graph_id, {})
        filtered: dict[str, dict[str, Any]] = {
            graph_id: {
                "runs": int(entry.get("runs", 0)),
                "succeeded": int(entry.get("succeeded", 0)),
                "failed": int(entry.get("failed", 0)),
                "total_duration_s": float(entry.get("total_duration_s", 0.0)),
            }
        }
        return GraphStats(root={gid: GraphStatsEntry(**vals) for gid, vals in filtered.items()})

    # Normalize all entries to GraphStatsEntry
    normalized: dict[str, GraphStatsEntry] = {}
    for gid, vals in raw_all.items():
        normalized[gid] = GraphStatsEntry(
            runs=int(vals.get("runs", 0)),
            succeeded=int(vals.get("succeeded", 0)),
            failed=int(vals.get("failed", 0)),
            total_duration_s=float(vals.get("total_duration_s", 0.0)),
        )

    return GraphStats(root=normalized)


@router.get("/stats/memory", response_model=MemoryStats)
async def get_memory_stats(
    scope_id: Annotated[str | None, Query(description="Logical memory scope (optional)")] = None,
    window: Annotated[str, Query(description="Time window, e.g., '24h', '7d'")] = "24h",
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemoryStats:
    """
    Get memory usage statistics.

    - **scope_id**: Logical memory scope (optional).
    - **window**: Time window for stats (e.g., "24h", "7d").
    """
    container = current_services()
    meter = getattr(container, "metering", None)

    if meter is None:
        raise HTTPException(status_code=501, detail="Metering service not available")

    raw: dict[str, dict[str, int]] = await meter.get_memory_stats(
        scope_id=scope_id,
        user_id=identity.user_id if identity and identity.user_id else None,
        org_id=identity.org_id if identity and identity.org_id else None,
        window=window,
    )
    # raw: { "memory.user_msg": {"count": N}, ... }
    return MemoryStats(root=raw)


@router.get("/stats/artifacts", response_model=ArtifactStats)
async def get_artifacts_stats(
    window: Annotated[str, Query(description="Time window, e.g., '24h', '7d'")] = "24h",
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> ArtifactStats:
    """
    Aggregate artifact stats for this user/org: counts, bytes, pinned, etc.
    Backed by MeteringService.get_artifact_stats().
    """
    container = current_services()
    meter = getattr(container, "metering", None)
    if meter is None:
        raise HTTPException(status_code=501, detail="Metering service not available")

    raw: dict[str, dict[str, int]] = await meter.get_artifact_stats(
        user_id=identity.user_id if identity and identity.user_id else None,
        org_id=identity.org_id if identity and identity.org_id else None,
        window=window,
    )
    # raw: { "json": {"count":..., "bytes":..., "pinned_count":..., "pinned_bytes":...}, ... }
    return ArtifactStats(root=raw)


@router.get("/stats/llm", response_model=LLMStats)
async def get_stats_llm(
    window: Annotated[str, Query(description="Time window, e.g., '24h', '7d'")] = "24h",
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMStats:
    """
    LLM usage stats: tokens, requests, breakdown by provider/model.
    Backed by MeteringService.get_llm_stats().
    """
    container = current_services()
    meter = getattr(container, "metering", None)
    if meter is None:
        raise HTTPException(status_code=501, detail="Metering service not available")

    raw: dict[str, dict[str, int]] = await meter.get_llm_stats(
        user_id=identity.user_id if identity and identity.user_id else None,
        org_id=identity.org_id if identity and identity.org_id else None,
        window=window,
    )
    # raw: { "gpt-4o-mini": {"calls":..., "prompt_tokens":..., "completion_tokens":...}, ... }
    return LLMStats(root=raw)

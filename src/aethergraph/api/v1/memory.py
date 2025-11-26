# memory-related inspection

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query

from .deps import RequestIdentity, get_identity
from .schemas import (
    MemoryEvent,
    MemoryEventListResponse,
    MemorySearchHit,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySummaryEntry,
    MemorySummaryListResponse,
)

router = APIRouter(tags=["memory"])


@router.get("/memory/events", response_model=MemoryEventListResponse)
async def list_memory_events(
    scope_id: str,
    kinds: str | None = Query(None, description="Comma-separated list of kinds to filter"),  # noqa: B008
    tags: str | None = Query(None, description="Comma-separated list of tags to filter"),  # noqa: B008
    after: datetime | None = Query(None),  # noqa: B008
    before: datetime | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(50, ge=1, le=200),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemoryEventListResponse:
    """
    List raw memory events for a scope.

    TODO:
      - Integrate with HotLog/EventStore.
      - Implement time-based and cursor-based pagination.
    """
    now = datetime.utcnow()
    dummy_event = MemoryEvent(
        event_id="evt-1",
        scope_id=scope_id,
        kind="chat_user",
        tags=["stub"],
        created_at=now - timedelta(minutes=1),
        data={"text": "stub memory event"},
    )
    return MemoryEventListResponse(events=[dummy_event], next_cursor=None)


@router.get("/memory/summaries", response_model=MemorySummaryListResponse)
async def list_memory_summaries(
    scope_id: str = Query(...),  # noqa: B008
    summary_tag: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(50, ge=1, le=200),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySummaryListResponse:
    """
    List long-term memory summaries for a scope.

    TODO:
      - Integrate with summarizer storage (Persistence/JSONL/etc.).
    """
    now = datetime.utcnow()
    dummy_summary = MemorySummaryEntry(
        summary_id="sum-1",
        scope_id=scope_id,
        summary_tag=summary_tag or "session",
        created_at=now - timedelta(hours=1),
        time_from=now - timedelta(hours=2),
        time_to=now - timedelta(hours=1),
        text="Stub summary of recent events.",
        metadata={},
    )
    return MemorySummaryListResponse(summaries=[dummy_summary], next_cursor=None)


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    req: MemorySearchRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySearchResponse:
    """
    Semantic/keyword memory search.

    TODO:
      - Wire to memory index / RAG backend.
    """
    now = datetime.utcnow()
    dummy_event = MemoryEvent(
        event_id="evt-hit",
        scope_id=req.scope_id or "stub_scope",
        kind="chat_user",
        tags=["search_stub"],
        created_at=now,
        data={"text": f"Result for query '{req.query}'"},
    )
    hit = MemorySearchHit(score=0.99, event=dummy_event, summary=None)
    return MemorySearchResponse(hits=[hit])

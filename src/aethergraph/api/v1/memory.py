# memory-related inspection

from contextlib import suppress
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from aethergraph.contracts.services.memory import Event
from aethergraph.core.runtime.runtime_services import current_services

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

# NOTE: since hotlog is bounded in memory, it is fine to filter and rank in-memory for now.
# In future, if we need to process large volumes of memory data, we should look into changing the
# backend memory storage to support indexed queries (not changing the API contracts).

router = APIRouter(tags=["memory"])


# ------------ helpers / stubs ------------ #
def _parse_ts(ts: str) -> datetime:
    """Parse ISO8601 timestamp string to datetime."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def _event_to_api_event(evt: Event) -> MemoryEvent:
    created_at = _parse_ts(evt.ts)

    data: dict[str, Any] | None = None
    if evt.data is not None:
        data = evt.data
    elif evt.text:
        data = {"text": evt.text}

    # Fallback: if old events had no scope_id, use run_id so UI still works.
    scope = evt.scope_id or evt.run_id

    return MemoryEvent(
        event_id=evt.event_id,
        scope_id=scope,
        kind=evt.kind,
        tags=evt.tags or [],
        created_at=created_at,
        data=data or {},
    )


def _doc_to_summary_entry(doc_id: str, doc: dict[str, Any]) -> MemorySummaryEntry:
    ts_str = doc.get("ts") or doc.get("created_at") or ""
    created_at = _parse_ts(ts_str) if ts_str else datetime.utcnow()

    tw = doc.get("time_window") or {}  # expected to have 'from' and 'to'
    from_str = tw.get("from") or tw.get("start") or ""
    to_str = tw.get("to") or tw.get("end") or ""
    time_from = _parse_ts(from_str) if from_str else created_at
    time_to = _parse_ts(to_str) if to_str else created_at

    # prefer 'summary' field, fallback to text if present
    text = doc.get("summary") or doc.get("text") or ""

    # Strip out the core fields from metadata
    meta_keys = {
        "summary",
        "text",
        "scope_id",
        "run_id",
        "summary_tag",
        "ts",
        "time_window",
    }
    metadata = {k: v for k, v in doc.items() if k not in meta_keys}

    return MemorySummaryEntry(
        summary_id=doc_id,
        scope_id=doc.get("scope_id") or doc.get("run_id") or "",
        summary_tag=doc.get("summary_tag"),
        created_at=created_at,
        time_from=time_from,
        time_to=time_to,
        text=text,
        metadata=metadata,
    )


def _string_score(haystack: str, needle: str) -> float:
    """
    Very simple scoring: 0.0 if no match, 1.0 if case-insensitive substring match.
    Placeholder until a real semantic index is wired in.
    """
    if not needle:
        return 0.0
    h = haystack.lower()
    n = needle.lower()
    return 1.0 if n in h else 0.0


# ------------ API endpoints ------------ #
@router.get("/memory/events", response_model=MemoryEventListResponse)
async def list_memory_events(
    scope_id: str,
    kinds: Annotated[
        str | None, Query(description="Comma-separated list of kinds to filter")
    ] = None,  # noqa: B008
    tags: Annotated[str | None, Query(description="Comma-separated list of tags to filter")] = None,  # noqa: B008
    after: Annotated[datetime | None, Query()] = None,  # noqa: B008
    before: Annotated[datetime | None, Query()] = None,  # noqa: B008
    cursor: Annotated[str | None, Query()] = None,  # noqa: B008
    limit: Annotated[int, Query(ge=1, le=200)] = 50,  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemoryEventListResponse:
    """
    List raw memory events for a scope.

    Currently:
      - Treats `scope_id` as the underlying `run_id` used by HotLog/Persistence.
      - Reads from HotLog only (recent in-memory events).
      - Applies basic filters by kind, tag, and time.

    TODO:
      - Integrate with a long-term event store (Persistence queries).
      - Implement cursor-based pagination.
      - Optionally map scope_id → multiple runs.
      - Filter by identity.user_id / org_id when multi-tenant.
    """
    container = current_services()
    mem_factory = getattr(container, "memory_factory", None)
    if mem_factory is None:
        # No memory configured
        return MemoryEventListResponse(events=[], next_cursor=None)

    hotlog = mem_factory.hotlog

    # Parse filters
    kinds_list: list[str] | None = None
    if kinds:
        kinds_list = [k.strip() for k in kinds.split(",") if k.strip()]
    tags_list: list[str] | None = None
    if tags:
        tags_list = [t.strip() for t in tags.split(",") if t.strip()]

    # Fetch slightly more than limit to determine if there's a next page and
    # we can filter in python
    raw_events: list[Event] = await hotlog.recent(
        scope_id,
        kinds=kinds_list,
        limit=mem_factory.hot_limit + 10,
    )

    filtered: list[Event] = []
    for evt in raw_events:
        dt = _parse_ts(evt.ts)
        if after and dt <= after:
            continue
        if before and dt >= before:
            continue
        if tags_list:
            evt_tags = evt.tags or []
            if not any(t in evt_tags for t in tags_list):
                continue
        filtered.append(evt)

    # Latest events last, ehen slice to requested limit
    if len(filtered) > limit:
        filtered = filtered[-limit:]

    api_events = [_event_to_api_event(e) for e in filtered]
    return MemoryEventListResponse(events=api_events, next_cursor=None)


@router.get("/memory/summaries", response_model=MemorySummaryListResponse)
async def list_memory_summaries(
    scope_id: Annotated[str, Query()],
    summary_tag: Annotated[str | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySummaryListResponse:
    """
    List long-term memory summaries for a scope.

    Currently:
      - Scans the DocStore (memory_factory.docs) and filters docs where:
          doc["scope_id"] == scope_id
          and (summary_tag is None or doc["summary_tag"] == summary_tag)
      - Converts each summary doc into MemorySummaryEntry.

    TODO:
      - Avoid full scan for large DocStores (add indexed queries).
      - Implement cursor-based pagination.
      - Optionally filter by identity.user_id / org_id.
    """
    container = current_services()
    mem_factory = getattr(container, "memory_factory", None)
    if mem_factory is None:
        return MemorySummaryListResponse(summaries=[], next_cursor=None)

    docs = mem_factory.docs

    # DocStore.list() returns a list of doc_ids; we load and filter them
    try:
        doc_ids = await docs.list()
    except TypeError:
        # If the concrete DocStore doesn't support list(), return empty
        return MemorySummaryListResponse(summaries=[], next_cursor=None)

    entries: list[MemorySummaryEntry] = []

    for doc_id in doc_ids:
        doc = await docs.get(doc_id)
        if not doc:
            continue

        if doc.get("scope_id") != scope_id:
            continue

        if summary_tag is not None and doc.get("summary_tag") != summary_tag:
            continue

        entries.append(_doc_to_summary_entry(doc_id, doc))

    # Sort by created_at descending
    entries.sort(key=lambda e: e.created_at, reverse=True)

    if len(entries) > limit:
        entries = entries[:limit]
    return MemorySummaryListResponse(summaries=entries, next_cursor=None)


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    req: MemorySearchRequest,
    identity: RequestIdentity = Annotated[Depends(get_identity)],  # noqa: B008
) -> MemorySearchResponse:
    """
    Semantic/keyword memory search.

    Current behavior:
      - Uses a naive substring match over:
          • recent HotLog events for the given scope_id (treated as run_id)
          • all summary docs for that scope_id
      - Returns MemorySearchHit with either `event` or `summary` populated.

    TODO:
      - Plug into a real semantic index / RAG backend (mem_factory.rag_facade).
      - Support more advanced filters (kinds, tags, summary_tag) on the request.
    """
    container = current_services()
    mem_factory = getattr(container, "memory_factory", None)
    if mem_factory is None:
        return MemorySearchResponse(hits=[])

    scope_id = req.scope_id or ""
    query = req.query or ""
    top_k = getattr(req, "top_k", 10) or 10

    hotlog = mem_factory.hotlog
    docs = mem_factory.docs

    hits: list[MemorySearchHit] = []

    # 1) Search recent HotLog events
    if scope_id:
        raw_events: list[Event] = await hotlog.recent(
            scope_id,
            kinds=None,
            limit=mem_factory.hot_limit,
        )
        for evt in raw_events:
            text_parts: list[str] = []
            if evt.text:
                text_parts.append(evt.text)
            if evt.data:
                with suppress(Exception):
                    text_parts.append(str(evt.data))
            haystack = " ".join(text_parts)
            score = _string_score(haystack, query)
            if score <= 0.0:
                continue

            api_evt = _event_to_api_event(evt)
            hits.append(
                MemorySearchHit(
                    score=score,
                    event=api_evt,
                    summary=None,
                )
            )

    # 2) Search summary docs
    try:
        doc_ids = await docs.list()
    except TypeError:
        doc_ids = []

    for doc_id in doc_ids:
        doc = await docs.get(doc_id)
        if not doc:
            continue

        if scope_id and doc.get("scope_id") != scope_id:
            continue

        text_parts: list[str] = []
        if doc.get("summary"):
            text_parts.append(str(doc.get("summary")))

        if doc.get("key_facts"):
            with suppress(Exception):
                text_parts.append(" ".join(map(str, doc["key_facts"])))

        haystack = " ".join(text_parts)
        score = _string_score(haystack, query)
        if score <= 0.0:
            continue

        summary_entry = _doc_to_summary_entry(doc_id, doc)
        hits.append(
            MemorySearchHit(
                score=score,
                event=None,
                summary=summary_entry,
            )
        )

    # Sort by score (desc) and truncate
    hits.sort(key=lambda h: h.score, reverse=True)
    if len(hits) > top_k:
        hits = hits[:top_k]

    return MemorySearchResponse(hits=hits)

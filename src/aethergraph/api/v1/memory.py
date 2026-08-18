"""Bounded Memory inspection over canonical provider services."""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore

from aethergraph.contracts.services.memory import Event
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.memory.canonical_public import CanonicalPublicMemoryFacade
from aethergraph.storage.contracts import PageRequest, SearchMode, SortDirection, StorageScope

from .deps import RequestIdentity, get_identity
from .schemas.memory import (
    MemoryEvent,
    MemoryEventListResponse,
    MemorySearchHit,
    MemorySearchRequest,
    MemorySearchResponse,
    MemorySummaryEntry,
    MemorySummaryListResponse,
)

router = APIRouter(tags=["memory"])


def _parse_ts(ts: str | float | int) -> datetime:
    if isinstance(ts, int | float):
        return datetime.fromtimestamp(float(ts), tz=UTC)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    parsed = datetime.fromisoformat(ts)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def _snippet_from_event(event: Event, max_len: int = 120) -> str:
    raw: str | None = event.text
    if raw is None and isinstance(event.data, dict):
        data_text = event.data.get("text")
        if isinstance(data_text, str) and data_text.strip():
            raw = data_text
        else:
            with suppress(Exception):
                raw = json.dumps(event.data, ensure_ascii=False, sort_keys=True)
    snippet = " ".join(str(raw or "").split())
    return snippet if len(snippet) <= max_len else snippet[: max_len - 1].rstrip() + "..."


def _event_to_api_event(event: Event) -> MemoryEvent:
    data = event.data if event.data is not None else ({"text": event.text} if event.text else {})
    return MemoryEvent(
        event_id=event.event_id,
        scope_id=event.scope_id or event.run_id,
        ts=event.ts,
        session_id=event.session_id,
        agent_id=event.agent_id,
        run_id=event.run_id,
        node_id=event.node_id,
        graph_id=event.graph_id,
        kind=event.kind,
        stage=event.stage,
        topic=event.topic,
        tool=event.tool,
        tags=event.tags or [],
        severity=event.severity,
        signal=event.signal,
        created_at=_parse_ts(event.ts),
        snippet=_snippet_from_event(event),
        text=event.text,
        data=data,
        metrics=event.metrics,
        inputs=event.inputs,
        outputs=event.outputs,
    )


def _event_to_summary(event: Event, summary_tag: str) -> MemorySummaryEntry:
    payload = dict(event.data) if isinstance(event.data, dict) else {}
    time_window = payload.pop("time_window", {})
    created_at = _parse_ts(payload.pop("ts", event.ts))
    time_from = _parse_ts(time_window.get("from") or time_window.get("start") or event.ts)
    time_to = _parse_ts(time_window.get("to") or time_window.get("end") or event.ts)
    text = str(payload.pop("summary", payload.pop("text", event.text or "")))
    payload.pop("scope_id", None)
    payload.pop("summary_tag", None)
    return MemorySummaryEntry(
        summary_id=event.event_id,
        scope_id=event.scope_id,
        summary_tag=summary_tag,
        created_at=created_at,
        time_from=time_from,
        time_to=time_to,
        text=text,
        metadata=payload,
    )


def _scope_from_selector(
    *,
    identity: RequestIdentity,
    scope_id: str | None,
    session_id: str | None,
    run_id: str | None,
    agent_id: str | None,
) -> tuple[StorageScope, str]:
    selected_session = session_id
    selected_run = run_id
    logical_scope = scope_id
    if scope_id and session_id is None and run_id is None:
        if scope_id.startswith("session:"):
            selected_session = scope_id.removeprefix("session:")
        elif scope_id.startswith("run:"):
            selected_run = scope_id.removeprefix("run:")
        elif scope_id.startswith("org:") and ":user:" in scope_id:
            requested_org, requested_user = scope_id.removeprefix("org:").split(":user:", 1)
            if identity.org_id != requested_org or identity.user_id != requested_user:
                raise HTTPException(status_code=403, detail="Memory user scope is not authorized")
        elif scope_id.startswith("user:"):
            requested_user = scope_id.removeprefix("user:")
            if identity.user_id != requested_user:
                raise HTTPException(status_code=403, detail="Memory user scope is not authorized")
        elif scope_id.startswith("org:") and ":user:" not in scope_id:
            requested_org = scope_id.removeprefix("org:")
            if identity.org_id != requested_org:
                raise HTTPException(
                    status_code=403, detail="Memory organization scope is not authorized"
                )
        elif scope_id == "global":
            pass
        else:
            # The current AG UI supplies a bare run identity as its compatibility
            # scope selector. It is mapped only to the canonical run dimension.
            selected_run = scope_id
    if not any((selected_session, selected_run, identity.org_id, identity.user_id)):
        raise HTTPException(
            status_code=422,
            detail="Memory inspection requires a canonical session, run, user, or organization scope",
        )
    if logical_scope is None:
        if selected_session:
            logical_scope = f"session:{selected_session}"
        elif selected_run:
            logical_scope = f"run:{selected_run}"
        elif identity.org_id and identity.user_id:
            logical_scope = f"org:{identity.org_id}:user:{identity.user_id}"
        elif identity.user_id:
            logical_scope = f"user:{identity.user_id}"
        elif identity.org_id:
            logical_scope = f"org:{identity.org_id}"
        else:
            logical_scope = "global"
    return (
        StorageScope(
            org_id=identity.org_id,
            user_id=identity.user_id,
            session_id=selected_session,
            run_id=selected_run,
            agent_id=agent_id,
        ),
        logical_scope,
    )


def _memory_facade(
    *,
    identity: RequestIdentity,
    scope_id: str | None,
    session_id: str | None = None,
    run_id: str | None = None,
    agent_id: str | None = None,
) -> CanonicalPublicMemoryFacade:
    container = current_services()
    factory = getattr(container, "memory_factory", None)
    if factory is None:
        raise HTTPException(status_code=503, detail="Memory storage is not configured")
    scope, logical_scope = _scope_from_selector(
        identity=identity,
        scope_id=scope_id,
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
    )
    return factory.for_public_execution(scope, logical_scope_id=logical_scope)


@router.get("/memory/events", response_model=MemoryEventListResponse)
async def list_memory_events(
    scope_id: Annotated[str | None, Query(description="Deprecated logical scope selector")] = None,
    session_id: Annotated[str | None, Query(description="Canonical session filter")] = None,
    agent_id: Annotated[str | None, Query(description="Canonical Agent filter")] = None,
    run_id: Annotated[str | None, Query(description="Canonical run filter")] = None,
    kinds: Annotated[str | None, Query(description="Comma-separated exact event kinds")] = None,
    tags: Annotated[str | None, Query(description="Comma-separated required tags")] = None,
    after: Annotated[datetime | None, Query()] = None,
    before: Annotated[datetime | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemoryEventListResponse:
    memory = _memory_facade(
        identity=identity,
        scope_id=scope_id,
        session_id=session_id,
        run_id=run_id,
        agent_id=agent_id,
    )
    page = await memory.query_event_page(
        page=PageRequest(limit=limit, cursor=cursor),
        kinds=_parse_csv(kinds),
        tags=_parse_csv(tags),
        since=after,
        until=before,
        order=SortDirection.DESCENDING,
    )
    return MemoryEventListResponse(
        events=[_event_to_api_event(event) for event in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/memory/summaries", response_model=MemorySummaryListResponse)
async def list_memory_summaries(
    scope_id: Annotated[str, Query(description="Deprecated logical scope selector")],
    summary_tag: Annotated[str | None, Query()] = None,
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySummaryListResponse:
    tag = summary_tag or "session"
    memory = _memory_facade(identity=identity, scope_id=scope_id)
    page = await memory.query_event_page(
        page=PageRequest(limit=limit, cursor=cursor),
        kinds=["long_term_summary"],
        tags=["summary", tag],
        order=SortDirection.DESCENDING,
    )
    return MemorySummaryListResponse(
        summaries=[_event_to_summary(event, tag) for event in page.items],
        next_cursor=page.next_cursor,
    )


@router.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    req: MemorySearchRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> MemorySearchResponse:
    memory = _memory_facade(identity=identity, scope_id=req.scope_id)
    hits = await memory.search_events(
        query=req.query,
        mode=SearchMode.LEXICAL,
        top_k=req.top_k,
    )
    projected: list[MemorySearchHit] = []
    for hit in hits:
        if "summary" in (hit.event.tags or []):
            summary_tag = next(
                (tag for tag in hit.event.tags or [] if tag != "summary"),
                "session",
            )
            projected.append(
                MemorySearchHit(
                    score=hit.score,
                    event=None,
                    summary=_event_to_summary(hit.event, summary_tag),
                )
            )
        else:
            projected.append(
                MemorySearchHit(
                    score=hit.score,
                    event=_event_to_api_event(hit.event),
                    summary=None,
                )
            )
    return MemorySearchResponse(hits=projected)

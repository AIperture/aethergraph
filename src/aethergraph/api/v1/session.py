from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.api.v1.runs import _extract_app_id_from_tags
from aethergraph.api.v1.schemas import (
    RunSummary,
    Session,
    SessionChatEvent,
    SessionCreateRequest,
    SessionListResponse,
    SessionRunsResponse,
    SessionUpdateRequest,
)
from aethergraph.core.runtime.run_types import RunImportance, RunVisibility, SessionKind
from aethergraph.core.runtime.runtime_registry import current_registry
from aethergraph.core.runtime.runtime_services import current_services

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=Session)
async def create_session(
    body: SessionCreateRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Session:
    """
    Create a new session.
    """
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")

    sess = await ss.create(
        kind=body.kind,
        title=body.title,
        external_ref=body.external_ref,
        user_id=identity.user_id,
        org_id=identity.org_id,
        source="webui",
    )

    return sess


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    kind: SessionKind | None = Query(None, description="Filter sessions by kind"),  # noqa: B008
    limit: int = Query(50, ge=1, le=1000),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SessionListResponse:
    """
    List sessions for the current user/org, optionally filtered by kind.
    """
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")

    offset = decode_cursor(cursor)

    sessions = await ss.list_for_user(
        user_id=identity.user_id,
        org_id=identity.org_id,
        kind=kind,
        limit=limit,
        offset=offset,
    )

    next_cursor = encode_cursor(offset + limit) if len(sessions) == limit else None
    return SessionListResponse(items=sessions, next_cursor=next_cursor)


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Session:
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")

    sess = await ss.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Optional: enforce that the session belongs to the user/org
    if identity.mode != "local":
        if identity.user_id and sess.user_id is not None and sess.user_id != identity.user_id:
            raise HTTPException(status_code=403, detail="Access denied")

        if identity.org_id and sess.org_id is not None and sess.org_id != identity.org_id:
            raise HTTPException(status_code=403, detail="Access denied")
    return sess


@router.get("/sessions/{session_id}/runs", response_model=SessionRunsResponse)
async def get_session_runs(
    session_id: str,
    include_inline: bool = Query(False),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SessionRunsResponse:
    container = current_services()
    ss = getattr(container, "session_store", None)
    rm = getattr(container, "run_manager", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")
    if rm is None:
        raise HTTPException(status_code=500, detail="RunManager not available")

    # Make sure the session exists and belongs to this user/org
    sess = await ss.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if identity.mode != "local":
        if identity.user_id and sess.user_id is not None and sess.user_id != identity.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if identity.org_id and sess.org_id is not None and sess.org_id != identity.org_id:
            raise HTTPException(status_code=403, detail="Access denied")

    # For now, just scan recent runs and filter by session_id
    # Later, we need a dedicated index/query in RunStore
    records = await rm.list_records(
        graph_id=None,
        status=None,
        session_id=session_id,
        flow_id=None,
        limit=1000,
        offset=0,
    )

    # 🔹 Visibility & importance policy for session views:
    # - Always require importance == normal (ephemeral hidden for now).
    # - If include_inline is False:
    #       include only visibility == normal
    #   Else:
    #       include visibility in {normal, inline}
    visible_states = {RunVisibility.normal}
    if include_inline:
        visible_states.add(RunVisibility.inline)

    records = [
        rec
        for rec in records
        if rec.visibility in visible_states and rec.importance == RunImportance.normal
    ]

    reg = getattr(container, "registry", None) or current_registry()
    summaries: list[RunSummary] = []

    for rec in records:
        # defaults to avoid UnboundLocalError if reg is None
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

        # derive app info
        app_id = rec.meta.get("app_id") or _extract_app_id_from_tags(rec.tags)
        app_name = rec.meta.get("app_name")

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
                flow_id=flow_id,
                entrypoint=entrypoint,
                meta=rec.meta or {},
                app_id=app_id,
                app_name=app_name,
                session_id=rec.session_id,
                origin=rec.origin,
                visibility=rec.visibility,
                importance=rec.importance,
                agent_id=rec.agent_id,
            )
        )

    return SessionRunsResponse(items=summaries)


@router.get("/sessions/{session_id}/chat/events", response_model=list[SessionChatEvent])
async def get_session_chat_events(
    session_id: str,
    request: Request,
    since_ts: float | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> list[SessionChatEvent]:
    container = current_services()
    event_log = container.eventlog

    if event_log is None:
        raise HTTPException(status_code=503, detail="EventLog not available")

    since_dt: datetime | None = None
    if since_ts is not None:
        since_dt = datetime.fromtimestamp(since_ts, tz=timezone.utc)

    events = await event_log.query(
        scope_id=session_id,
        since=since_dt,
        kinds=["session_chat"],
        limit=1000,
    )
    out: list[SessionChatEvent] = []
    for ev in events:
        payload = ev.get("payload", {}) or {}
        out.append(
            SessionChatEvent(
                id=ev.get("id"),
                session_id=session_id,
                ts=ev.get("ts"),
                type=payload.get("type") or "agent.message",
                text=payload.get("text"),
                buttons=payload.get("buttons", []),
                file=payload.get("file"),  # may be None
                files=payload.get("files") or None,  # 🔹 forward list
                meta=payload.get("meta", {}) or {},
                agent_id=payload.get("agent_id"),
            )
        )
    out.sort(key=lambda e: e.ts)

    return out


@router.patch("/sessions/{session_id}", response_model=Session)
async def update_session(
    session_id: str,
    body: SessionUpdateRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Session:
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")

    existing = await ss.get(session_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Enforce ownership for non-local modes
    if identity.mode != "local":
        if (
            identity.user_id
            and existing.user_id is not None
            and existing.user_id != identity.user_id
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        if identity.org_id and existing.org_id is not None and existing.org_id != identity.org_id:
            raise HTTPException(status_code=403, detail="Access denied")

    updated = await ss.update(
        session_id,
        title=body.title,
        external_ref=body.external_ref,
    )
    if updated is None:
        # Defensive; shouldn't happen given we already fetched it
        raise HTTPException(status_code=404, detail="Session not found")

    return updated


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> None:
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")

    existing = await ss.get(session_id)
    if existing is None:
        # 204 for idempotent delete
        return

    if identity.mode != "local":
        if (
            identity.user_id
            and existing.user_id is not None
            and existing.user_id != identity.user_id
        ):
            raise HTTPException(status_code=403, detail="Access denied")
        if identity.org_id and existing.org_id is not None and existing.org_id != identity.org_id:
            raise HTTPException(status_code=403, detail="Access denied")

    await ss.delete(session_id)

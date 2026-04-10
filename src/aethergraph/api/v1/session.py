import asyncio
from contextlib import suppress
from datetime import UTC, datetime
import logging

from fastapi import (  # type: ignore
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)

from aethergraph.api.v1.deps import (
    RequestIdentity,
    ensure_identity_matches_owner,
    get_authn,
    get_identity,
)
from aethergraph.api.v1.pagination import (
    decode_cursor,
    decode_cursor_v2,
    encode_cursor,
    encode_keyset_before_cursor,
    encode_keyset_cursor,
)
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.api.v1.run_presenters import to_run_summary
from aethergraph.api.v1.schemas.session import (
    Session,
    SessionChatEvent,
    SessionChatEventListResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionRunsResponse,
    SessionUpdateRequest,
    SessionWorkStatus,
    SessionWorkStatusResponse,
)
from aethergraph.core.runtime.run_types import RunImportance, RunVisibility, SessionKind
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.channel.session_work_status import (
    WORK_STATUS_EVENT_KIND,
    get_session_work_status,
)

router = APIRouter(tags=["sessions"])
logger = logging.getLogger(__name__)


def _ensure_session_access(identity: RequestIdentity, sess: Session) -> None:
    ensure_identity_matches_owner(
        identity,
        user_id=sess.user_id,
        org_id=sess.org_id,
        missing_status=403,
        missing_detail="Access denied",
    )


async def _get_session_or_404(session_id: str):
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")
    sess = await ss.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess


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

    # Enforce identity for cloud/demo
    if identity.mode in ("cloud", "demo") and identity.user_id is None:
        raise HTTPException(status_code=403, detail="User identity required")

    sessions = await ss.list_for_user(
        user_id=identity.user_id if identity.mode in ("cloud", "demo") else identity.user_id,
        org_id=identity.org_id if identity.mode in ("cloud", "demo") else identity.org_id,
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

    sess = await _get_session_or_404(session_id)
    _ensure_session_access(identity, sess)
    return sess


@router.get("/sessions/{session_id}/runs", response_model=SessionRunsResponse)
async def get_session_runs(
    session_id: str,
    include_inline: bool = Query(False),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(50, ge=1, le=200),  # noqa: B008
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
    sess = await _get_session_or_404(session_id)
    _ensure_session_access(identity, sess)

    offset = decode_cursor(cursor)

    # Over-fetch to compensate for Python-side visibility/importance filtering
    fetch_limit = limit * 2

    records = await rm.list_records(
        graph_id=None,
        status=None,
        session_id=session_id,
        flow_id=None,
        limit=fetch_limit,
        offset=offset,
    )

    # Check if the store returned a full page (there might be more)
    store_has_more = len(records) == fetch_limit

    # Visibility & importance policy for session views
    visible_states = {RunVisibility.normal}
    if include_inline:
        visible_states.add(RunVisibility.inline)

    records = [
        rec
        for rec in records
        if rec.visibility in visible_states and rec.importance == RunImportance.normal
    ]

    # Trim to requested limit
    records = records[:limit]

    reg = scoped_registry(identity)
    summaries = [to_run_summary(rec, reg=reg) for rec in records]

    next_cursor = encode_cursor(offset + fetch_limit) if store_has_more else None

    return SessionRunsResponse(items=summaries, next_cursor=next_cursor)


def _row_to_session_chat_event(row: dict, session_id: str) -> SessionChatEvent:
    payload = row.get("payload", {}) or {}
    return SessionChatEvent(
        id=row.get("id"),
        session_id=session_id,
        ts=row.get("ts"),
        type=payload.get("type") or "agent.message",
        text=payload.get("text"),
        buttons=payload.get("buttons", []),
        file=payload.get("file"),
        files=payload.get("files") or None,
        rich=payload.get("rich") or None,
        meta=payload.get("meta", {}) or {},
        agent_id=payload.get("agent_id"),
        upsert_key=payload.get("upsert_key"),
    )


def _row_to_session_work_status(row: dict) -> SessionWorkStatus | None:
    payload = dict(row.get("payload") or {})
    raw = payload.get("work_status")
    if not raw:
        return None
    return SessionWorkStatus.model_validate(raw)


@router.websocket("/ws/sessions/{session_id}/chat")
async def ws_session_chat(websocket: WebSocket, session_id: str):
    DROP_FROM_HISTORY = {"agent.stream.start", "agent.stream.delta"}

    container = current_services()
    event_log = container.eventlog
    hub = getattr(container, "eventhub", None)
    authn = get_authn()

    if hub is None or event_log is None:
        await websocket.close(code=1011)
        return

    roles_header = websocket.headers.get("x-roles")
    roles = roles_header.split(",") if roles_header else []
    client_id = websocket.headers.get("x-client-id")
    resolved = authn.resolve(
        deploy_mode=getattr(getattr(container, "settings", None), "deploy_mode", "local"),
        session_id=websocket.cookies.get(authn.cookie_name),
        client_id=client_id,
        x_user_id=websocket.headers.get("x-user-id"),
        x_org_id=websocket.headers.get("x-org-id"),
        roles=roles,
        x_mode=websocket.headers.get("x-mode"),
    )
    identity = RequestIdentity(
        user_id=resolved.user_id,
        org_id=resolved.org_id,
        roles=resolved.roles,
        client_id=resolved.client_id,
        grant_id=resolved.session.grant_id if resolved.session else None,
        auth_source=resolved.auth_source,
        catalog_scope={
            k: v
            for k, v in {
                "apps": list(resolved.grant.allowed_apps) if resolved.grant else [],
                "agents": list(resolved.grant.allowed_agents) if resolved.grant else [],
            }.items()
            if v
        }
        or None,
        mode="cloud"
        if resolved.mode == "cloud_proxy"
        else "demo"
        if resolved.mode == "demo_guest"
        else "local",
    )
    try:
        sess = await _get_session_or_404(session_id)
        _ensure_session_access(identity, sess)
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail)[:120])
        return

    await websocket.accept()

    async def send_snapshot() -> None:
        events = await event_log.query(
            scope_id=session_id,
            kinds=["session_chat"],
            since=None,
            limit=200,
        )
        work_status_rows = await event_log.query(
            scope_id=session_id,
            kinds=[WORK_STATUS_EVENT_KIND],
            since=None,
            limit=200,
        )
        filtered = []
        for ev in events:
            payload = ev.get("payload") or {}
            t = payload.get("type") or "agent.message"
            if t in DROP_FROM_HISTORY:
                continue
            filtered.append(ev)

        filtered.sort(key=lambda ev: ev.get("ts") or 0)
        initial_payload = [
            _row_to_session_chat_event(ev, session_id).model_dump() for ev in filtered
        ]

        # Include backward pagination cursor so the frontend can load older messages via REST
        has_older = len(events) >= 200
        older_cursor: str | None = None
        if has_older and filtered:
            oldest_row_id = filtered[0].get("_row_id")
            if oldest_row_id is not None:
                older_cursor = encode_keyset_before_cursor(oldest_row_id)

        snapshot_msg: dict = {"kind": "snapshot", "events": initial_payload}
        latest_work_status = (
            _row_to_session_work_status(work_status_rows[-1]) if work_status_rows else None
        )
        snapshot_msg["work_status"] = (
            latest_work_status.model_dump() if latest_work_status else None
        )
        snapshot_msg["has_older"] = has_older
        if older_cursor is not None:
            snapshot_msg["older_cursor"] = older_cursor
        await websocket.send_json(snapshot_msg)

    async def recv_until_disconnect() -> None:
        # Blocks until disconnect; does not require the client to send meaningful messages.
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                return

    async def send_live() -> None:
        async def _send_chat() -> None:
            async for row in hub.subscribe(scope_id=session_id, kind="session_chat"):
                ev = _row_to_session_chat_event(row, session_id)
                await websocket.send_json({"kind": "event", "event": ev.model_dump()})

        async def _send_work_status() -> None:
            async for row in hub.subscribe(scope_id=session_id, kind=WORK_STATUS_EVENT_KIND):
                work_status = _row_to_session_work_status(row)
                await websocket.send_json(
                    {
                        "kind": "work_status",
                        "work_status": work_status.model_dump() if work_status else None,
                    }
                )

        chat_task = asyncio.create_task(_send_chat())
        work_status_task = asyncio.create_task(_send_work_status())
        try:
            done, pending = await asyncio.wait(
                {chat_task, work_status_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            for task in (chat_task, work_status_task):
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task

    recv_task = send_task = None
    try:
        await send_snapshot()

        recv_task = asyncio.create_task(recv_until_disconnect())
        send_task = asyncio.create_task(send_live())

        done, pending = await asyncio.wait(
            {recv_task, send_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel the other task (this is what prevents idle hangs)
        for t in pending:
            t.cancel()
            with suppress(asyncio.CancelledError):
                await t

    except WebSocketDisconnect:
        # can happen from send_json
        return
    except asyncio.CancelledError:
        # critical for uvicorn --reload
        with suppress(Exception):
            await websocket.close(code=1001)
        raise
    except Exception as e:
        with suppress(Exception):
            await websocket.close(code=1011, reason=str(e)[:120])
    finally:
        for t in (recv_task, send_task):
            if t and not t.done():
                t.cancel()
                with suppress(asyncio.CancelledError):
                    await t


@router.get("/sessions/{session_id}/chat/events", response_model=SessionChatEventListResponse)
async def get_session_chat_events(
    session_id: str,
    request: Request,
    since_ts: float | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SessionChatEventListResponse:
    DROP_FROM_HISTORY = {"agent.stream.start", "agent.stream.delta"}

    container = current_services()
    event_log = container.eventlog

    if event_log is None:
        raise HTTPException(status_code=503, detail="EventLog not available")
    sess = await _get_session_or_404(session_id)
    _ensure_session_access(identity, sess)

    since_dt: datetime | None = None
    if since_ts is not None:
        since_dt = datetime.fromtimestamp(since_ts, tz=UTC)

    # Decode cursor — supports keyset (after_id), keyset_before (before_id), and legacy offset
    cursor_info = decode_cursor_v2(cursor)
    after_id: int | None = None
    before_id: int | None = None
    query_offset: int = 0
    is_backward = False
    if cursor_info is not None:
        if cursor_info.kind == "keyset":
            after_id = cursor_info.value
        elif cursor_info.kind == "keyset_before":
            before_id = cursor_info.value
            is_backward = True
        else:
            query_offset = cursor_info.value

    # Fetch limit+1 to detect if there's a next page
    fetch_limit = limit + 1

    events = await event_log.query(
        scope_id=session_id,
        since=since_dt,
        kinds=["session_chat"],
        limit=fetch_limit,
        after_id=after_id,
        before_id=before_id,
        offset=query_offset,
    )

    if since_ts is not None and after_id is None and not is_backward:
        # make cursor exclusive -- only return events after since_ts to avoid duplicates
        events = [ev for ev in events if (ev.get("ts") or 0) > since_ts]

    # Filter legacy persisted deltas/start
    events = [ev for ev in events if (ev.get("payload") or {}).get("type") not in DROP_FROM_HISTORY]

    # Determine next_cursor before trimming
    has_more = len(events) > limit
    if is_backward:  # noqa: SIM108
        # For backward pagination, the "extra" event is the oldest one (first in list)
        # Trim from the front to keep the most recent ones
        events = events[-limit:] if has_more else events
    else:
        events = events[:limit]

    next_cursor: str | None = None
    if has_more and events:
        if is_backward:
            # Next page goes further back — cursor points before the oldest returned event
            first_row_id = events[0].get("_row_id")
            if first_row_id is not None:
                next_cursor = encode_keyset_before_cursor(first_row_id)
        else:
            last_row_id = events[-1].get("_row_id")
            if last_row_id is not None:
                next_cursor = encode_keyset_cursor(last_row_id)
            else:
                # Fallback to offset cursor
                next_cursor = encode_cursor(query_offset + limit)

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
                files=payload.get("files") or None,  # forward list
                meta=payload.get("meta", {}) or {},
                agent_id=payload.get("agent_id"),
                upsert_key=payload.get("upsert_key"),  # forward idempotent key
                rich=payload.get("rich") or None,  # forward rich content
            )
        )
    out.sort(key=lambda e: e.ts)

    return SessionChatEventListResponse(events=out, next_cursor=next_cursor)


@router.get("/sessions/{session_id}/work-status", response_model=SessionWorkStatusResponse)
async def get_session_work_status_api(
    session_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SessionWorkStatusResponse:
    sess = await _get_session_or_404(session_id)
    _ensure_session_access(identity, sess)
    work_status = await get_session_work_status(session_id)
    return SessionWorkStatusResponse(
        work_status=SessionWorkStatus.model_validate(work_status) if work_status else None
    )


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

    existing = await _get_session_or_404(session_id)
    _ensure_session_access(identity, existing)

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
    _ensure_session_access(identity, existing)

    await ss.delete(session_id)

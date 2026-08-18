import logging
import re

from fastapi import (  # type: ignore
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from aethergraph.api.v1.deps import (
    RequestIdentity,
    ensure_identity_matches_owner,
    get_identity,
)
from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.api.v1.registry_helpers import scoped_registry
from aethergraph.api.v1.run_presenters import to_run_summary
from aethergraph.api.v1.schemas.session import (
    Session,
    SessionCreateRequest,
    SessionInferTitleRequest,
    SessionInferTitleResponse,
    SessionListResponse,
    SessionRunsResponse,
    SessionUpdateRequest,
)
from aethergraph.contracts.integration import SemanticEventKind
from aethergraph.core.runtime.run_types import RunImportance, RunVisibility, SessionKind
from aethergraph.core.runtime.runtime_services import current_services

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


def _normalize_session_title(raw: str, *, max_len: int = 64) -> str:
    title = re.sub(r"\s+", " ", raw).strip()
    title = re.sub(r"^(title\s*:\s*)", "", title, flags=re.IGNORECASE)
    title = title.strip().strip("\"'`").strip()
    title = re.sub(r"\s+", " ", title).strip()
    return title[:max_len].rstrip(" .,:;!-")


def _extract_initial_title_context(
    messages: list[dict[str, str]],
) -> tuple[str | None, str | None]:
    first_user_text = next(
        (item["content"] for item in messages if item["role"] == "user"),
        None,
    )
    if first_user_text is None:
        return None, None
    user_index = next(index for index, item in enumerate(messages) if item["role"] == "user")
    first_assistant_text = next(
        (item["content"] for item in messages[user_index + 1 :] if item["role"] == "assistant"),
        None,
    )
    return first_user_text, first_assistant_text


def _extract_refresh_title_context(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    if len(messages) < 2:
        return []

    anchor = messages[:2]
    recent = messages[-6:]
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in anchor + recent:
        key = (item["role"], item["content"])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


async def _query_session_title_messages(
    session_id: str,
    *,
    limit: int = 100,
) -> list[dict[str, str]]:
    container = current_services()
    semantic_events = getattr(container, "semantic_events", None)
    manifest = getattr(container, "host_manifest", None)
    if semantic_events is None or manifest is None:
        return []
    records = await semantic_events.list_session(
        deployment_id=manifest.deployment_id,
        session_id=session_id,
    )
    messages: list[dict[str, str]] = []
    for record in records[-limit:]:
        event = record.event
        if event.kind == SemanticEventKind.INPUT_ACCEPTED:
            role = "user"
        elif event.kind == SemanticEventKind.MESSAGE_COMPLETED:
            role = "assistant"
        else:
            continue
        text = getattr(event.payload, "text", None)
        if isinstance(text, str) and text.strip():
            messages.append({"role": role, "content": text.strip()})
    return messages


async def _infer_session_title_from_events(
    session_id: str,
    *,
    mode: str = "initial",
) -> str | None:
    title_messages = await _query_session_title_messages(session_id, limit=100)

    container = current_services()
    llm_service = getattr(container, "llm", None)
    if llm_service is None:
        raise RuntimeError("LLM service not available")

    client = llm_service.get("default")
    if mode == "refresh":
        context_messages = _extract_refresh_title_context(title_messages)
        if not context_messages:
            return None
        prompt = (
            "Generate a concise workspace title for this conversation.\n"
            "Return only the title.\n"
            "Use 3 to 7 words.\n"
            "Prefer the current topic of the conversation, not generic labels.\n"
            "Do not use quotes."
        )
        messages = [
            {"role": "system", "content": "You create short, precise conversation titles."},
            {"role": "user", "content": prompt},
            *context_messages,
        ]
    else:
        user_text, assistant_text = _extract_initial_title_context(title_messages)
        if not user_text or not assistant_text:
            return None
        prompt = (
            "Generate a concise workspace title for this conversation.\n"
            "Return only the title.\n"
            "Use 3 to 7 words.\n"
            "Do not use quotes.\n\n"
            f"User: {user_text}\n"
            f"Assistant: {assistant_text}"
        )
        messages = [
            {"role": "system", "content": "You create short, precise conversation titles."},
            {"role": "user", "content": prompt},
        ]
    text, _usage = await client.chat(
        messages=messages,
        max_output_tokens=32,
        call_name="session_infer_title",
    )
    title = _normalize_session_title(text)
    return title or None


@router.post("/sessions/{session_id}/infer-title", response_model=SessionInferTitleResponse)
async def infer_session_title(
    session_id: str,
    body: SessionInferTitleRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SessionInferTitleResponse:
    container = current_services()
    ss = getattr(container, "session_store", None)
    if ss is None:
        raise HTTPException(status_code=500, detail="SessionStore not available")

    session = await _get_session_or_404(session_id)
    _ensure_session_access(identity, session)

    if session.title_source == "manual" and not body.force:
        return SessionInferTitleResponse(
            session_id=session_id,
            title=session.title,
            updated=False,
            reason="skipped_manual",
        )

    if session.title and not body.force:
        return SessionInferTitleResponse(
            session_id=session_id,
            title=session.title,
            updated=False,
            reason="skipped_has_title",
        )

    try:
        title = await _infer_session_title_from_events(session_id, mode=body.mode)
    except RuntimeError:
        return SessionInferTitleResponse(
            session_id=session_id,
            title=session.title,
            updated=False,
            reason="skipped_disabled_llm",
        )
    except Exception as exc:
        logger.exception("Failed to infer title for session %s", session_id)
        raise HTTPException(
            status_code=502, detail=f"Failed to infer session title: {exc}"
        ) from exc

    if not title:
        return SessionInferTitleResponse(
            session_id=session_id,
            title=session.title,
            updated=False,
            reason="skipped_no_context",
        )

    updated = await ss.update(session_id, title=title, title_source="auto")
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInferTitleResponse(
        session_id=session_id,
        title=updated.title,
        updated=True,
        reason="generated",
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
        title_source="manual" if body.title is not None else None,
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

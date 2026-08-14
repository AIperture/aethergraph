from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response  # type: ignore

from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.observability import (
    ActiveObservabilityScopeError,
    ObservabilityFacade,
    ObservabilityIdentity,
    ObservabilityNotFoundError,
    ObservabilityUnavailableError,
)
from aethergraph.observability.studio_translation import (
    _matches_scope,
    _paginate_rows,
)

from .deps import RequestIdentity, get_identity
from .schemas.inspect import (
    AgentEventListResponse,
    AgentEventTypeListResponse,
    AgentEventTypeRecord,
    InspectLogListResponse,
    InspectLogRecord,
    LLMCallListResponse,
    LLMCallRecord,
    LLMSummary,
    TraceEventListResponse,
    TraceSummary,
)

router = APIRouter(prefix="/inspect", tags=["inspect"])
trace_router = APIRouter(prefix="/api/trace", tags=["trace"])


def _parse_window(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _observability_identity(identity: RequestIdentity) -> ObservabilityIdentity:
    return ObservabilityIdentity(
        mode=identity.mode,
        user_id=identity.user_id,
        org_id=identity.org_id,
    )


def _observability_facade(identity: RequestIdentity) -> ObservabilityFacade:
    container = current_services()
    facade = getattr(container, "observability", None)
    if facade is None:
        raise HTTPException(status_code=503, detail="Observability not configured")
    return facade.for_identity(_observability_identity(identity))


def _observability_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ObservabilityNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ActiveObservabilityScopeError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=503, detail=str(exc))


async def _get_run_or_404(run_id: str, identity: RequestIdentity):
    container = current_services()
    rm = getattr(container, "run_manager", None)
    if rm is None:
        raise HTTPException(status_code=503, detail="Run manager not configured")
    rec = await rm.get_record(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if identity.mode in ("cloud", "demo"):
        if identity.user_id is None:
            raise HTTPException(status_code=403, detail="User identity required")
        if rec.user_id != identity.user_id:
            raise HTTPException(status_code=404, detail="Run not found")
        if identity.org_id and rec.org_id != identity.org_id:
            raise HTTPException(status_code=404, detail="Run not found")
    return rec


@router.get("/runs/{run_id}/trace", response_model=TraceEventListResponse)
async def get_run_trace(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TraceEventListResponse:
    await _get_run_or_404(run_id, identity)
    return await _observability_facade(identity).list_inspect_traces(
        since=_parse_window(from_),
        until=_parse_window(to),
        run_id=run_id,
        cursor=cursor,
        limit=limit,
    )


@router.get("/traces", response_model=TraceEventListResponse)
async def list_traces(
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    run_id: str | None = Query(None),  # noqa: B008
    session_id: str | None = Query(None),  # noqa: B008
    agent_id: str | None = Query(None),  # noqa: B008
    app_id: str | None = Query(None),  # noqa: B008
    graph_id: str | None = Query(None),  # noqa: B008
    node_id: str | None = Query(None),  # noqa: B008
    trace_id: str | None = Query(None),  # noqa: B008
    service: list[str] | None = Query(None),  # noqa: B008
    status: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TraceEventListResponse:
    try:
        return await _observability_facade(identity).list_inspect_traces(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
            trace_id=trace_id,
            service=service,
            status=status,
            cursor=cursor,
            limit=limit,
        )
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc


@router.get("/traces/{trace_id}", response_model=TraceEventListResponse)
async def get_trace_by_id(
    trace_id: str,
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TraceEventListResponse:
    return await _observability_facade(identity).list_inspect_traces(
        trace_id=trace_id, cursor=cursor, limit=limit
    )


@router.get("/runs/{run_id}/trace/summary", response_model=TraceSummary)
async def get_run_trace_summary(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TraceSummary:
    await _get_run_or_404(run_id, identity)
    events = (
        await _observability_facade(identity).list_inspect_traces(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            limit=2_147_483_647,
        )
    ).items
    trace_ids = sorted({event.trace_id for event in events if event.trace_id})
    failing_services = Counter(event.service for event in events if event.error is not None)
    latest_error_ts = max((event.ts for event in events if event.error is not None), default=None)
    return TraceSummary(
        run_id=run_id,
        trace_ids=trace_ids,
        span_count=len(events),
        error_count=sum(1 for event in events if event.error is not None),
        total_duration_ms=int(sum(int(event.duration_ms or 0) for event in events)),
        top_failing_services=dict(failing_services.most_common(5)),
        latest_error_ts=latest_error_ts,
    )


@router.get("/runs/{run_id}/llm-calls", response_model=LLMCallListResponse)
async def get_run_llm_calls(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMCallListResponse:
    await _get_run_or_404(run_id, identity)
    return await _observability_facade(identity).list_inspect_llm_calls(
        run_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        cursor=cursor,
        limit=limit,
    )


@router.get("/llm-calls", response_model=LLMCallListResponse)
async def list_llm_calls(
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    run_id: str | None = Query(None),  # noqa: B008
    session_id: str | None = Query(None),  # noqa: B008
    agent_id: str | None = Query(None),  # noqa: B008
    app_id: str | None = Query(None),  # noqa: B008
    graph_id: str | None = Query(None),  # noqa: B008
    node_id: str | None = Query(None),  # noqa: B008
    provider: str | None = Query(None),  # noqa: B008
    model: str | None = Query(None),  # noqa: B008
    call_type: str | None = Query(None),  # noqa: B008
    status: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMCallListResponse:
    try:
        return await _observability_facade(identity).list_inspect_llm_calls(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
            provider=provider,
            model=model,
            call_type=call_type,
            status=status,
            cursor=cursor,
            limit=limit,
        )
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc


@router.get("/llm-calls/{call_id}", response_model=LLMCallRecord)
async def get_llm_call(
    call_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMCallRecord:
    try:
        return await _observability_facade(identity).get_inspect_llm_call(call_id)
    except (ObservabilityUnavailableError, ObservabilityNotFoundError) as exc:
        raise _observability_http_error(exc) from exc


@router.get("/runs/{run_id}/llm-summary", response_model=LLMSummary)
async def get_run_llm_summary(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMSummary:
    await _get_run_or_404(run_id, identity)
    items = (
        await _observability_facade(identity).list_inspect_llm_calls(
            run_id=run_id,
            since=_parse_window(from_),
            until=_parse_window(to),
            limit=2_147_483_647,
        )
    ).items
    by_model: Counter[str] = Counter()
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    error_count = 0
    for item in items:
        by_model[item.model] += 1
        prompt_tokens += int(item.usage.get("prompt_tokens") or item.usage.get("input_tokens") or 0)
        completion_tokens += int(
            item.usage.get("completion_tokens") or item.usage.get("output_tokens") or 0
        )
        total_tokens += int(item.usage.get("total_tokens") or 0)
        if item.error_type:
            error_count += 1
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return LLMSummary(
        run_id=run_id,
        total_calls=len(items),
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        error_count=error_count,
        by_model=dict(by_model),
    )


@router.get("/runs/{run_id}/logs", response_model=InspectLogListResponse)
async def get_run_logs(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> InspectLogListResponse:
    await _get_run_or_404(run_id, identity)
    try:
        return await _observability_facade(identity).list_inspect_logs(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            cursor=cursor,
            limit=limit,
        )
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc


@router.get("/logs", response_model=InspectLogListResponse)
async def list_logs(
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    run_id: str | None = Query(None),  # noqa: B008
    session_id: str | None = Query(None),  # noqa: B008
    agent_id: str | None = Query(None),  # noqa: B008
    app_id: str | None = Query(None),  # noqa: B008
    graph_id: str | None = Query(None),  # noqa: B008
    node_id: str | None = Query(None),  # noqa: B008
    level: str | None = Query(None),  # noqa: B008
    logger: str | None = Query(None),  # noqa: B008
    run_status: str | None = Query(None),  # noqa: B008
    trace_status: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> InspectLogListResponse:
    try:
        return await _observability_facade(identity).list_inspect_logs(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
            level=level,
            logger=logger,
            run_status=run_status,
            trace_status=trace_status,
            cursor=cursor,
            limit=limit,
        )
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc


@router.get("/agent-event-types", response_model=AgentEventTypeListResponse)
async def list_agent_event_types(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> AgentEventTypeListResponse:
    _ = identity
    container = current_services()
    registry = getattr(container, "agent_event_registry", None)
    if registry is None:
        return AgentEventTypeListResponse(items=[])
    items = [
        AgentEventTypeRecord(
            event_type=entry.event_type,
            category=entry.category,
            display_label=entry.display_label,
            payload_schema_name=entry.payload_schema_name,
            payload_schema_version=entry.payload_schema_version,
            renderer_hint=entry.renderer_hint,
            redaction_policy=entry.redaction_policy,
        )
        for entry in sorted(registry.list(), key=lambda item: item.event_type)
    ]
    return AgentEventTypeListResponse(items=items)


@router.get("/errors", response_model=InspectLogListResponse)
async def get_errors(
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    graph_id: str | None = Query(None),  # noqa: B008
    app_id: str | None = Query(None),  # noqa: B008
    agent_id: str | None = Query(None),  # noqa: B008
    run_status: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> InspectLogListResponse:
    try:
        records = (
            await _observability_facade(identity).list_inspect_logs(
                since=_parse_window(from_),
                until=_parse_window(to),
                limit=2_147_483_647,
            )
        ).items
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc
    items: list[InspectLogRecord] = []
    for record in records:
        if record.level not in {"warning", "error", "critical"}:
            continue
        if not _matches_scope(
            record.scope,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
        ):
            continue
        if run_status and record.run_status != run_status:
            continue
        items.append(record)
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return InspectLogListResponse(items=page, next_cursor=next_cursor)


@router.get("/runs/{run_id}/agent-events", response_model=AgentEventListResponse)
async def get_run_agent_events(
    run_id: str,
    event_type: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> AgentEventListResponse:
    await _get_run_or_404(run_id, identity)
    resp = await list_agent_events(
        from_=None,
        to=None,
        run_id=run_id,
        session_id=None,
        agent_id=None,
        app_id=None,
        graph_id=None,
        node_id=None,
        event_type=event_type,
        cursor=cursor,
        limit=limit,
        identity=identity,
    )
    return resp


@router.get("/sessions/{session_id}/agent-events", response_model=AgentEventListResponse)
async def get_session_agent_events(
    session_id: str,
    event_type: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> AgentEventListResponse:
    resp = await list_agent_events(
        from_=None,
        to=None,
        run_id=None,
        session_id=session_id,
        agent_id=None,
        app_id=None,
        graph_id=None,
        node_id=None,
        event_type=event_type,
        cursor=cursor,
        limit=limit,
        identity=identity,
    )
    return resp


@router.get("/agent-events", response_model=AgentEventListResponse)
async def list_agent_events(
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    run_id: str | None = Query(None),  # noqa: B008
    session_id: str | None = Query(None),  # noqa: B008
    agent_id: str | None = Query(None),  # noqa: B008
    app_id: str | None = Query(None),  # noqa: B008
    graph_id: str | None = Query(None),  # noqa: B008
    node_id: str | None = Query(None),  # noqa: B008
    event_type: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> AgentEventListResponse:
    try:
        return await _observability_facade(identity).list_inspect_agent_events(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
            event_type=event_type,
            cursor=cursor,
            limit=limit,
        )
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc


def _required_trace(value: dict | None) -> dict:
    if value is None:
        raise HTTPException(status_code=404, detail="Trace was not found")
    return value


@trace_router.get("/sessions")
async def list_trace_sessions(
    limit: int = Query(50, ge=1, le=200),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return await _observability_facade(identity).list_trace_sessions(limit=limit, cursor=cursor)


@trace_router.delete("/sessions/{session_id}", status_code=204)
async def delete_trace_session(
    session_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> Response:
    """Delete observations for one completed Trace Explorer session.

    Intro:
        Purges AG-owned capture while retaining canonical runtime history.

    Examples:
        `DELETE /api/trace/sessions/session-1`
        `DELETE /api/trace/sessions/session-complete`

    Args:
        session_id: Exact authoritative session identity.
        identity: Authenticated request identity used for containment.

    Returns:
        Response: Empty HTTP 204 response after completed deletion.

    Notes:
        Active or resumable sessions return HTTP 409.
    """
    try:
        await _observability_facade(identity).delete_session_observations(session_id)
    except ActiveObservabilityScopeError as exc:
        raise _observability_http_error(exc) from exc
    return Response(status_code=204)


@trace_router.post("/sessions/delete")
async def delete_trace_sessions(
    payload: dict,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict[str, int]:
    """Delete observations for multiple completed sessions.

    Intro:
        Validates every session before performing the first destructive action.

    Examples:
        `POST /api/trace/sessions/delete {"session_ids": ["s-1"]}`
        `POST /api/trace/sessions/delete {"session_ids": []}`

    Args:
        payload: JSON object containing a `session_ids` list.
        identity: Authenticated request identity used for containment.

    Returns:
        dict[str, int]: Count of unique session scopes deleted.

    Notes:
        One active, resumable, or unauthorized session prevents the whole batch.
    """
    session_ids = payload.get("session_ids")
    if not isinstance(session_ids, list):
        raise HTTPException(status_code=400, detail="session_ids must be a list")
    normalized = [str(session_id).strip() for session_id in session_ids if str(session_id).strip()]
    try:
        results = await _observability_facade(identity).delete_sessions_observations(normalized)
    except ActiveObservabilityScopeError as exc:
        raise _observability_http_error(exc) from exc
    return {"deleted": len(results)}


@trace_router.get("/traces/{trace_id}")
async def get_trace_session(
    trace_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    facade = _observability_facade(identity)
    tree = _required_trace(await facade.inspect_trace(trace_id))
    run = tree["runs"][0]
    return {
        "trace_id": trace_id,
        "session_id": run["session_id"],
        "graph_id": run["graph_id"],
        "graph_topology": tree["graph_topologies_by_trace_id"][trace_id],
        "spans": tree["spans_by_trace_id"][trace_id],
        "agent_states": tree["agent_states_by_trace_id"][trace_id],
        "plans": tree["plans_by_trace_id"][trace_id],
        "started_at": run["started_at"],
        "ended_at": run.get("ended_at"),
        "status": run["status"],
    }


@trace_router.get("/traces/{trace_id}/tree")
async def get_trace_tree(
    trace_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return _required_trace(await _observability_facade(identity).inspect_trace(trace_id))


@trace_router.get("/traces/{trace_id}/graph")
async def get_trace_graph(
    trace_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return _required_trace(await _observability_facade(identity).get_trace_graph(trace_id))


@trace_router.get("/traces/{trace_id}/spans")
async def get_trace_spans(
    trace_id: str,
    kind: str | None = Query(None),  # noqa: B008
    agent_id: str | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return _required_trace(
        await _observability_facade(identity).get_trace_spans(
            trace_id, kind=kind, agent_id=agent_id
        )
    )


@trace_router.get("/traces/{trace_id}/plans")
async def get_trace_plans(
    trace_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return _required_trace(await _observability_facade(identity).get_trace_plans(trace_id))


@trace_router.get("/traces/{trace_id}/context-snapshots/{snapshot_id}")
async def get_trace_context_snapshot(
    trace_id: str,
    snapshot_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return _required_trace(
        await _observability_facade(identity).get_trace_context_snapshot(trace_id, snapshot_id)
    )


@trace_router.get("/traces/{trace_id}/agents/{agent_id}/states")
async def get_trace_agent_states(
    trace_id: str,
    agent_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> dict:
    return _required_trace(
        await _observability_facade(identity).get_trace_agent_states(trace_id, agent_id)
    )

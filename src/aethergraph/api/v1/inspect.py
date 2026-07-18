from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore

from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.services.inspect.facade import (
    InspectionFacade,
    InspectionIdentity,
    InspectionNotFoundError,
    InspectionUnavailableError,
    _matches_scope,
    _paginate_rows,
    _passes_identity_scope,
    _present_llm_row,
    _present_trace_row,
    _scope_from_mapping,
    _store_identity_scope,
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


def _parse_window(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _inspection_identity(identity: RequestIdentity) -> InspectionIdentity:
    return InspectionIdentity(
        mode=identity.mode,
        user_id=identity.user_id,
        org_id=identity.org_id,
    )


def _identity_scope(identity: RequestIdentity) -> tuple[str | None, str | None]:
    return _store_identity_scope(_inspection_identity(identity))


def _inspection_facade(identity: RequestIdentity) -> InspectionFacade:
    container = current_services()

    async def resolve_run_statuses(run_ids: set[str]) -> dict[str, str]:
        run_manager = getattr(container, "run_manager", None)
        if run_manager is None:
            return {}
        statuses: dict[str, str] = {}
        for run_id in run_ids:
            record = await run_manager.get_record(run_id)
            if record is not None:
                statuses[run_id] = (
                    record.status.value
                    if isinstance(record.status, RunStatus)
                    else str(record.status)
                )
        return statuses

    return InspectionFacade(
        event_log=getattr(container, "eventlog", None),
        observability=getattr(container, "observability", None),
        identity=_inspection_identity(identity),
        run_status_resolver=resolve_run_statuses,
    )


def _inspection_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, InspectionNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
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


async def _collect_trace_rows(
    *, run_id: str, since: datetime | None, until: datetime | None
) -> list[dict[str, Any]]:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        raise HTTPException(status_code=503, detail="Event log not configured")
    rows = await event_log.query(
        scope_id=f"trace:run/{run_id}",
        since=since,
        until=until,
        kinds=["trace"],
        limit=None,
    )
    rows.sort(key=lambda row: row.get("ts") or 0.0)
    return rows


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
    rows = await _collect_trace_rows(
        run_id=run_id, since=_parse_window(from_), until=_parse_window(to)
    )
    items = [
        _present_trace_row(row)
        for row in rows
        if _passes_identity_scope(
            _scope_from_mapping(row.get("payload") or {}),
            _inspection_identity(identity),
        )
    ]
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return TraceEventListResponse(items=page, next_cursor=next_cursor)


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
        return await _inspection_facade(identity).list_traces(
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
    except InspectionUnavailableError as exc:
        raise _inspection_http_error(exc) from exc


@router.get("/traces/{trace_id}", response_model=TraceEventListResponse)
async def get_trace_by_id(
    trace_id: str,
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TraceEventListResponse:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        raise HTTPException(status_code=503, detail="Event log not configured")
    rows = await event_log.query(kinds=["trace"], limit=None)
    items = []
    for row in rows:
        payload = row.get("payload") or {}
        if payload.get("trace_id") != trace_id:
            continue
        event = _present_trace_row(row)
        if _passes_identity_scope(event.scope, _inspection_identity(identity)):
            items.append(event)
    items.sort(key=lambda item: item.ts)
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return TraceEventListResponse(items=page, next_cursor=next_cursor)


@router.get("/runs/{run_id}/trace/summary", response_model=TraceSummary)
async def get_run_trace_summary(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> TraceSummary:
    await _get_run_or_404(run_id, identity)
    rows = await _collect_trace_rows(
        run_id=run_id, since=_parse_window(from_), until=_parse_window(to)
    )
    events = [_present_trace_row(row) for row in rows]
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
    container = current_services()
    observability = getattr(container, "observability", None)
    store = observability.store if observability is not None else None
    if store is None:
        raise HTTPException(status_code=503, detail="LLM observation store not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await store.query_llm_calls(
        run_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        user_id=user_id,
        org_id=org_id,
        limit=None,
    )
    items = [_present_llm_row(row) for row in rows]
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return LLMCallListResponse(items=page, next_cursor=next_cursor)


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
        return await _inspection_facade(identity).list_llm_calls(
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
    except InspectionUnavailableError as exc:
        raise _inspection_http_error(exc) from exc


@router.get("/llm-calls/{call_id}", response_model=LLMCallRecord)
async def get_llm_call(
    call_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMCallRecord:
    try:
        return await _inspection_facade(identity).get_llm_call(call_id)
    except (InspectionUnavailableError, InspectionNotFoundError) as exc:
        raise _inspection_http_error(exc) from exc


@router.get("/runs/{run_id}/llm-summary", response_model=LLMSummary)
async def get_run_llm_summary(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMSummary:
    await _get_run_or_404(run_id, identity)
    container = current_services()
    observability = getattr(container, "observability", None)
    store = observability.store if observability is not None else None
    if store is None:
        raise HTTPException(status_code=503, detail="LLM observation store not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await store.query_llm_calls(
        run_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        user_id=user_id,
        org_id=org_id,
        limit=None,
    )
    items = [_present_llm_row(row) for row in rows]
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
        return await _inspection_facade(identity).list_logs(
            since=_parse_window(from_),
            until=_parse_window(to),
            run_id=run_id,
            cursor=cursor,
            limit=limit,
        )
    except InspectionUnavailableError as exc:
        raise _inspection_http_error(exc) from exc


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
        return await _inspection_facade(identity).list_logs(
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
    except InspectionUnavailableError as exc:
        raise _inspection_http_error(exc) from exc


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
            await _inspection_facade(identity).list_logs(
                since=_parse_window(from_),
                until=_parse_window(to),
                limit=2_147_483_647,
            )
        ).items
    except InspectionUnavailableError as exc:
        raise _inspection_http_error(exc) from exc
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
        return await _inspection_facade(identity).list_agent_events(
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
    except InspectionUnavailableError as exc:
        raise _inspection_http_error(exc) from exc

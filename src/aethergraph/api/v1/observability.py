from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore

from aethergraph.core.runtime.runtime_services import current_services
from aethergraph.observability import (
    ObservabilityIdentity,
    ObservabilityNotFoundError,
    ObservabilityUnavailableError,
)
from aethergraph.observability.canonical_inspection import CanonicalInspectionReader
from aethergraph.observability.contracts import (
    AgentEventListResponse,
    AgentEventTypeListResponse,
    AgentEventTypeRecord,
    InspectLogListResponse,
    LLMCallListResponse,
    LLMCallRecord,
    LLMSummary,
    TraceEventListResponse,
    TraceSummary,
)

from .deps import RequestIdentity, get_identity

router = APIRouter(prefix="/inspect", tags=["inspect"])
_DEPRECATED_APP_QUERY_DESCRIPTION = (
    "Deprecated optional App compatibility metadata; not authorization or canonical scope."
)


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


def _observability_facade(identity: RequestIdentity) -> CanonicalInspectionReader:
    container = current_services()
    services = getattr(container, "storage_services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Observability not configured")
    return services.inspection(identity=_observability_identity(identity))


def _observability_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ObservabilityNotFoundError):
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
    return await _observability_facade(identity).list_traces(
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
    app_id: str | None = Query(  # noqa: B008
        None,
        deprecated=True,
        description=_DEPRECATED_APP_QUERY_DESCRIPTION,
    ),
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
        return await _observability_facade(identity).list_traces(
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
    return await _observability_facade(identity).list_traces(
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
    return await _observability_facade(identity).summarize_traces(
        run_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
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
    return await _observability_facade(identity).list_llm_calls(
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
    app_id: str | None = Query(  # noqa: B008
        None,
        deprecated=True,
        description=_DEPRECATED_APP_QUERY_DESCRIPTION,
    ),
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
        return await _observability_facade(identity).list_llm_calls(
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
        return await _observability_facade(identity).get_llm_call(call_id)
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
    return await _observability_facade(identity).summarize_llm_calls(
        run_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
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
        return await _observability_facade(identity).list_logs(
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
    app_id: str | None = Query(  # noqa: B008
        None,
        deprecated=True,
        description=_DEPRECATED_APP_QUERY_DESCRIPTION,
    ),
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
        return await _observability_facade(identity).list_logs(
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
    app_id: str | None = Query(  # noqa: B008
        None,
        deprecated=True,
        description=_DEPRECATED_APP_QUERY_DESCRIPTION,
    ),
    agent_id: str | None = Query(None),  # noqa: B008
    run_status: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> InspectLogListResponse:
    try:
        return await _observability_facade(identity).list_logs(
            since=_parse_window(from_),
            until=_parse_window(to),
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            levels=("warning", "error", "critical"),
            run_status=run_status,
            cursor=cursor,
            limit=limit,
        )
    except ObservabilityUnavailableError as exc:
        raise _observability_http_error(exc) from exc


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
    app_id: str | None = Query(  # noqa: B008
        None,
        deprecated=True,
        description=_DEPRECATED_APP_QUERY_DESCRIPTION,
    ),
    graph_id: str | None = Query(None),  # noqa: B008
    node_id: str | None = Query(None),  # noqa: B008
    event_type: str | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> AgentEventListResponse:
    try:
        return await _observability_facade(identity).list_agent_events(
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

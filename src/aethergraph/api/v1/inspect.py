from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query  # type: ignore

from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.core.runtime.run_types import RunStatus
from aethergraph.core.runtime.runtime_services import current_services

from .deps import RequestIdentity, get_identity
from .schemas.inspect import (
    AgentEventEnvelope,
    AgentEventListResponse,
    AgentEventTypeListResponse,
    AgentEventTypeRecord,
    InspectLinks,
    InspectLogError,
    InspectLogListResponse,
    InspectLogRecord,
    InspectPayloadSchema,
    InspectProducer,
    InspectScope,
    LLMCallListResponse,
    LLMCallRecord,
    LLMSummary,
    TraceErrorInfo,
    TraceEvent,
    TraceEventListResponse,
    TraceSummary,
)

router = APIRouter(prefix="/inspect", tags=["inspect"])


def _parse_window(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _identity_scope(identity: RequestIdentity) -> tuple[str | None, str | None]:
    if identity.mode in ("cloud", "demo"):
        return identity.user_id, identity.org_id
    return None, None


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


def _scope_from_mapping(data: dict[str, Any] | None = None) -> InspectScope:
    data = data or {}
    return InspectScope(
        org_id=data.get("org_id"),
        user_id=data.get("user_id"),
        client_id=data.get("client_id"),
        run_id=data.get("run_id"),
        session_id=data.get("session_id"),
        agent_id=data.get("agent_id"),
        app_id=data.get("app_id"),
        graph_id=data.get("graph_id"),
        node_id=data.get("node_id"),
        trace_id=data.get("trace_id"),
        span_id=data.get("span_id"),
    )


def _passes_identity_scope(scope: InspectScope, identity: RequestIdentity) -> bool:
    if identity.mode not in ("cloud", "demo"):
        return True
    if identity.user_id is None:
        return False
    if scope.user_id and scope.user_id != identity.user_id:
        return False
    if identity.org_id and scope.org_id and scope.org_id != identity.org_id:
        return False
    return True


def _matches_scope(
    scope: InspectScope,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    agent_id: str | None = None,
    app_id: str | None = None,
    graph_id: str | None = None,
    node_id: str | None = None,
) -> bool:
    if run_id and scope.run_id != run_id:
        return False
    if session_id and scope.session_id != session_id:
        return False
    if agent_id and scope.agent_id != agent_id:
        return False
    if app_id and scope.app_id != app_id:
        return False
    if graph_id and scope.graph_id != graph_id:
        return False
    if node_id and scope.node_id != node_id:
        return False
    return True


def _paginate_rows(
    items: list[Any], *, cursor: str | None, limit: int
) -> tuple[list[Any], str | None]:
    offset = decode_cursor(cursor)
    page = items[offset : offset + limit]
    next_cursor = encode_cursor(offset + limit) if len(items) > offset + limit else None
    return page, next_cursor


def _present_trace_row(row: dict[str, Any]) -> TraceEvent:
    payload = row.get("payload") or {}
    scope = _scope_from_mapping(payload)
    scope.trace_id = payload.get("trace_id")
    scope.span_id = payload.get("span_id")
    summary = (
        f"{payload.get('service') or 'service'}/{payload.get('operation') or 'op'} "
        f"{payload.get('phase') or 'phase'} [{payload.get('status') or 'unknown'}]"
    )
    status = str(payload.get("status") or "unknown")
    severity = "error" if payload.get("error") else ("warning" if status == "pending" else "info")
    return TraceEvent(
        id=row.get("id") or payload.get("span_id") or payload.get("trace_id"),
        ts=float(row.get("ts") or 0.0),
        summary=summary,
        severity=severity,
        status=status,
        producer=InspectProducer(family="trace", name=str(payload.get("service") or "runtime")),
        scope=scope,
        tags=list(payload.get("tags") or []),
        links=None,
        payload=payload,
        trace_id=str(payload.get("trace_id") or ""),
        span_id=str(payload.get("span_id") or ""),
        parent_span_id=payload.get("parent_span_id"),
        service=str(payload.get("service") or "unknown"),
        operation=str(payload.get("operation") or "unknown"),
        phase=str(payload.get("phase") or "unknown"),
        duration_ms=payload.get("duration_ms"),
        request_preview=payload.get("request"),
        response_preview=payload.get("response"),
        error=TraceErrorInfo(**(payload.get("error") or {})) if payload.get("error") else None,
        metrics=dict(payload.get("metrics") or {}),
    )


def _present_llm_row(row: dict[str, Any]) -> LLMCallRecord:
    scope = _scope_from_mapping(row)
    scope.trace_id = row.get("trace_id")
    scope.span_id = row.get("span_id")
    status = "error" if row.get("error_type") else "ok"
    call_name = row.get("call_name")
    summary_prefix = f"[{call_name}] " if call_name else ""
    return LLMCallRecord(
        id=str(row.get("call_id")),
        ts=_parse_llm_ts(row.get("created_at")),
        summary=f"{summary_prefix}{row.get('provider')}/{row.get('model')} {row.get('call_type')}",
        severity="error" if row.get("error_type") else "info",
        status=status,
        producer=InspectProducer(family="llm", name=str(row.get("provider") or "unknown")),
        scope=scope,
        tags=[str(row.get("call_type") or "chat"), status],
        payload={
            "max_output_tokens": row.get("max_output_tokens"),
            "schema_name": row.get("schema_name"),
            "strict_schema": row.get("strict_schema"),
            "validate_json": row.get("validate_json"),
            "extra_params": row.get("extra_params") or {},
        },
        call_id=str(row.get("call_id")),
        created_at=str(row.get("created_at")),
        call_type=str(row.get("call_type") or "chat"),
        provider=str(row.get("provider") or "unknown"),
        model=str(row.get("model") or "unknown"),
        profile_name=row.get("profile_name"),
        call_name=row.get("call_name"),
        latency_ms=row.get("latency_ms"),
        usage=dict(row.get("usage") or {}),
        reasoning_effort=row.get("reasoning_effort"),
        output_format=row.get("output_format"),
        messages_preview=row.get("messages_preview"),
        trace_payload_preview=row.get("trace_payload_preview"),
        raw_text_preview=row.get("raw_text_preview"),
        messages=row.get("messages"),
        trace_payload=row.get("trace_payload"),
        raw_text=row.get("raw_text"),
        error_type=row.get("error_type"),
        error_message=row.get("error_message"),
    )


def _parse_llm_ts(value: str | None) -> float:
    if not value:
        return 0.0
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _present_log_row(
    row: dict[str, Any], *, run_status: str | None = None, trace_status: str | None = None
) -> InspectLogRecord:
    payload = row.get("payload") or {}
    scope = _scope_from_mapping(payload.get("scope") or {})
    inner = payload.get("payload") or {}
    return InspectLogRecord(
        id=str(payload.get("id") or row.get("id")),
        ts=float(payload.get("ts") or row.get("ts") or 0.0),
        summary=str(payload.get("summary") or inner.get("message") or ""),
        severity=str(payload.get("severity") or inner.get("level") or "info"),
        status=str(payload.get("status") or inner.get("level") or "info"),
        producer=InspectProducer(
            **(payload.get("producer") or {"family": "logger", "name": "unknown"})
        ),
        scope=scope,
        tags=list(payload.get("tags") or []),
        payload=inner,
        logger=str(inner.get("logger") or "unknown"),
        level=str(inner.get("level") or "info"),
        message=str(inner.get("message") or ""),
        error=InspectLogError(**(inner.get("error") or {})) if inner.get("error") else None,
        extra=dict(inner.get("extra") or {}),
        run_status=run_status,
        trace_status=trace_status,
    )


def _present_agent_row(row: dict[str, Any]) -> AgentEventEnvelope:
    payload = row.get("payload") or {}
    return AgentEventEnvelope(
        id=str(payload.get("event_id") or row.get("id")),
        ts=float(payload.get("ts") or row.get("ts") or 0.0),
        summary=str(payload.get("summary") or payload.get("event_type") or "agent event"),
        severity="error"
        if str(payload.get("status") or "").lower() in {"error", "failed"}
        else "info",
        status=str(payload.get("status") or "info"),
        producer=InspectProducer(
            **(payload.get("producer") or {"family": "agent", "name": "unknown"})
        ),
        scope=_scope_from_mapping(payload.get("scope") or {}),
        tags=list(payload.get("tags") or []),
        links=InspectLinks(**(payload.get("links") or {})),
        payload=dict(payload.get("payload") or {}),
        event_id=str(payload.get("event_id") or row.get("id")),
        event_type=str(payload.get("event_type") or "unknown"),
        payload_schema=InspectPayloadSchema(**(payload.get("payload_schema") or {})),
    )


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


async def _get_global_trace_events(
    *,
    since: datetime | None,
    until: datetime | None,
    identity: RequestIdentity,
) -> list[TraceEvent]:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        raise HTTPException(status_code=503, detail="Event log not configured")
    rows = await event_log.query(
        since=since,
        until=until,
        kinds=["trace"],
        limit=None,
    )
    items: list[TraceEvent] = []
    for row in rows:
        event = _present_trace_row(row)
        if _passes_identity_scope(event.scope, identity):
            items.append(event)
    items.sort(key=lambda item: item.ts, reverse=True)
    return items


async def _get_run_status_map(run_ids: set[str]) -> dict[str, str]:
    if not run_ids:
        return {}
    container = current_services()
    rm = getattr(container, "run_manager", None)
    if rm is None:
        return {}
    out: dict[str, str] = {}
    for rid in run_ids:
        rec = await rm.get_record(rid)
        if rec is not None:
            out[rid] = rec.status.value if isinstance(rec.status, RunStatus) else str(rec.status)
    return out


async def _get_global_log_records(
    *,
    since: datetime | None,
    until: datetime | None,
    identity: RequestIdentity,
) -> list[InspectLogRecord]:
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        raise HTTPException(status_code=503, detail="Event log not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await event_log.query(
        since=since,
        until=until,
        kinds=["inspect_log"],
        limit=None,
        user_id=user_id,
        org_id=org_id,
    )
    run_ids = {
        (row.get("payload") or {}).get("scope", {}).get("run_id")
        for row in rows
        if (row.get("payload") or {}).get("scope", {}).get("run_id")
    }
    trace_ids = {
        (row.get("payload") or {}).get("scope", {}).get("trace_id")
        for row in rows
        if (row.get("payload") or {}).get("scope", {}).get("trace_id")
    }
    run_statuses = await _get_run_status_map(run_ids)
    trace_statuses = await _get_trace_error_statuses(trace_ids)
    items: list[InspectLogRecord] = []
    for row in rows:
        record = _present_log_row(
            row,
            run_status=run_statuses.get((row.get("payload") or {}).get("scope", {}).get("run_id")),
            trace_status=trace_statuses.get(
                (row.get("payload") or {}).get("scope", {}).get("trace_id")
            ),
        )
        if _passes_identity_scope(record.scope, identity):
            items.append(record)
    items.sort(key=lambda item: item.ts, reverse=True)
    return items


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
        if _passes_identity_scope(_scope_from_mapping(row.get("payload") or {}), identity)
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
    items = await _get_global_trace_events(
        since=_parse_window(from_), until=_parse_window(to), identity=identity
    )
    filtered = [
        item
        for item in items
        if _matches_scope(
            item.scope,
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
        )
        and (trace_id is None or item.trace_id == trace_id)
        and (service is None or item.service in service)
        and (status is None or item.status == status)
    ]
    page, next_cursor = _paginate_rows(filtered, cursor=cursor, limit=limit)
    return TraceEventListResponse(items=page, next_cursor=next_cursor)


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
        if _passes_identity_scope(event.scope, identity):
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
    store = getattr(container, "llm_observation_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="LLM observation store not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await store.query(
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
    container = current_services()
    store = getattr(container, "llm_observation_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="LLM observation store not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await store.query(
        run_id=run_id,
        session_id=session_id,
        agent_id=agent_id,
        app_id=app_id,
        graph_id=graph_id,
        node_id=node_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        user_id=user_id,
        org_id=org_id,
        limit=None,
    )
    items = [_present_llm_row(row) for row in rows]
    items = [
        item
        for item in items
        if (provider is None or item.provider == provider)
        and (model is None or item.model == model)
        and (call_type is None or item.call_type == call_type)
        and (status is None or item.status == status)
    ]
    items.sort(key=lambda item: item.ts, reverse=True)
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return LLMCallListResponse(items=page, next_cursor=next_cursor)


@router.get("/llm-calls/{call_id}", response_model=LLMCallRecord)
async def get_llm_call(
    call_id: str,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMCallRecord:
    container = current_services()
    store = getattr(container, "llm_observation_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="LLM observation store not configured")
    row = await store.get(call_id)
    if row is None:
        raise HTTPException(status_code=404, detail="LLM call not found")
    record = _present_llm_row(row)
    if not _passes_identity_scope(record.scope, identity):
        raise HTTPException(status_code=404, detail="LLM call not found")
    return record


@router.get("/runs/{run_id}/llm-summary", response_model=LLMSummary)
async def get_run_llm_summary(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> LLMSummary:
    await _get_run_or_404(run_id, identity)
    container = current_services()
    store = getattr(container, "llm_observation_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="LLM observation store not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await store.query(
        run_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        user_id=user_id,
        org_id=org_id,
        limit=None,
    )
    items = [_present_llm_row(row) for row in rows]
    by_model = Counter()
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


async def _get_trace_error_statuses(trace_ids: set[str]) -> dict[str, str]:
    if not trace_ids:
        return {}
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        return {}
    rows = await event_log.query(kinds=["trace"], limit=None)
    out: dict[str, str] = {}
    for row in rows:
        payload = row.get("payload") or {}
        trace_id = payload.get("trace_id")
        if trace_id in trace_ids and payload.get("error") is not None:
            out[trace_id] = "error"
    return out


@router.get("/runs/{run_id}/logs", response_model=InspectLogListResponse)
async def get_run_logs(
    run_id: str,
    from_: datetime | None = Query(None, alias="from"),  # noqa: B008
    to: datetime | None = Query(None),  # noqa: B008
    cursor: str | None = Query(None),  # noqa: B008
    limit: int = Query(100, ge=1, le=500),  # noqa: B008
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> InspectLogListResponse:
    rec = await _get_run_or_404(run_id, identity)
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        raise HTTPException(status_code=503, detail="Event log not configured")
    user_id, org_id = _identity_scope(identity)
    rows = await event_log.query(
        scope_id=run_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        kinds=["inspect_log"],
        limit=None,
        user_id=user_id,
        org_id=org_id,
    )
    rows.sort(key=lambda row: row.get("ts") or 0.0)
    trace_ids = {
        (row.get("payload") or {}).get("scope", {}).get("trace_id")
        for row in rows
        if (row.get("payload") or {}).get("scope", {}).get("trace_id")
    }
    trace_statuses = await _get_trace_error_statuses(trace_ids)
    items = [
        _present_log_row(
            row,
            run_status=rec.status.value if isinstance(rec.status, RunStatus) else str(rec.status),
            trace_status=trace_statuses.get(
                (row.get("payload") or {}).get("scope", {}).get("trace_id")
            ),
        )
        for row in rows
    ]
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return InspectLogListResponse(items=page, next_cursor=next_cursor)


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
    items = await _get_global_log_records(
        since=_parse_window(from_), until=_parse_window(to), identity=identity
    )
    filtered = [
        item
        for item in items
        if _matches_scope(
            item.scope,
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
        )
        and (level is None or item.level == level)
        and (logger is None or item.logger == logger)
        and (run_status is None or item.run_status == run_status)
        and (trace_status is None or item.trace_status == trace_status)
    ]
    page, next_cursor = _paginate_rows(filtered, cursor=cursor, limit=limit)
    return InspectLogListResponse(items=page, next_cursor=next_cursor)


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
    records = await _get_global_log_records(
        since=_parse_window(from_), until=_parse_window(to), identity=identity
    )
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
    container = current_services()
    event_log = getattr(container, "eventlog", None)
    if event_log is None:
        raise HTTPException(status_code=503, detail="Event log not configured")
    user_id, org_id = _identity_scope(identity)
    scope_id = run_id or (f"session:{session_id}" if session_id else None)
    rows = await event_log.query(
        scope_id=scope_id,
        since=_parse_window(from_),
        until=_parse_window(to),
        kinds=["agent_event"],
        limit=None,
        user_id=user_id,
        org_id=org_id,
    )
    rows.sort(key=lambda row: row.get("ts") or 0.0, reverse=True)
    items: list[AgentEventEnvelope] = []
    for row in rows:
        event = _present_agent_row(row)
        if not _passes_identity_scope(event.scope, identity):
            continue
        if not _matches_scope(
            event.scope,
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
        ):
            continue
        if event_type and event.event_type != event_type:
            continue
        items.append(event)
    page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
    return AgentEventListResponse(items=page, next_cursor=next_cursor)

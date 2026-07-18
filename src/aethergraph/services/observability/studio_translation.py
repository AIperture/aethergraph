from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.api.v1.schemas.inspect import (
    AgentEventEnvelope,
    AgentEventListResponse,
    InspectLinks,
    InspectLogError,
    InspectLogListResponse,
    InspectLogRecord,
    InspectPayloadSchema,
    InspectProducer,
    InspectScope,
    LLMCallListResponse,
    LLMCallRecord,
    TraceErrorInfo,
    TraceEvent,
    TraceEventListResponse,
)

from .models import ObservationFilter

RunStatusResolver = Callable[[set[str]], Awaitable[Mapping[str, str]]]


class ObservabilityUnavailableError(RuntimeError):
    """Signal that a required inspection store is unavailable."""


class ObservabilityNotFoundError(LookupError):
    """Signal that a scoped inspection record does not exist."""


class ObservabilityWorkspaceError(RuntimeError):
    """Signal that a persisted inspection workspace cannot be opened."""


@dataclass(frozen=True)
class ObservabilityIdentity:
    mode: str = "local"
    user_id: str | None = None
    org_id: str | None = None


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


def _passes_identity_scope(scope: InspectScope, identity: ObservabilityIdentity) -> bool:
    if identity.mode not in ("cloud", "demo"):
        return True
    if identity.user_id is None:
        return False
    if scope.user_id and scope.user_id != identity.user_id:
        return False
    if identity.org_id and scope.org_id and scope.org_id != identity.org_id:
        return False
    return True


def _store_identity_scope(identity: ObservabilityIdentity) -> tuple[str | None, str | None]:
    if identity.mode in ("cloud", "demo"):
        return identity.user_id, identity.org_id
    return None, None


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
    return not (
        (run_id and scope.run_id != run_id)
        or (session_id and scope.session_id != session_id)
        or (agent_id and scope.agent_id != agent_id)
        or (app_id and scope.app_id != app_id)
        or (graph_id and scope.graph_id != graph_id)
        or (node_id and scope.node_id != node_id)
    )


def _paginate_rows(
    items: list[Any], *, cursor: str | None, limit: int
) -> tuple[list[Any], str | None]:
    offset = decode_cursor(cursor)
    page = items[offset : offset + limit]
    next_cursor = encode_cursor(offset + limit) if len(items) > offset + limit else None
    return page, next_cursor


def _present_trace_row(row: dict[str, Any]) -> TraceEvent:
    payload = dict(row.get("attributes") or {})
    scope = _scope_from_mapping(row)
    trace_id = str(row.get("trace_id") or row.get("run_id") or "")
    span_id = str(row.get("observation_id") or "")
    scope.trace_id = trace_id
    scope.span_id = span_id
    status = str(row.get("status") or "unknown")
    service = str(payload.get("service") or row.get("name") or "runtime")
    operation = str(payload.get("operation") or row.get("name") or "observation")
    error = payload.get("error")
    return TraceEvent(
        id=span_id,
        ts=_parse_llm_ts(row.get("occurred_at")),
        summary=str(row.get("summary") or f"{service}/{operation} [{status}]"),
        severity=str(row.get("severity") or "info"),
        status=status,
        producer=InspectProducer(family="trace", name=service),
        scope=scope,
        tags=[str(row.get("category") or "trace")],
        links=InspectLinks(
            parent_event_id=row.get("parent_observation_id"),
            caused_by_event_id=row.get("caused_by_observation_id"),
        ),
        payload=payload,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=row.get("parent_observation_id"),
        service=service,
        operation=operation,
        phase=str(payload.get("phase") or "event"),
        duration_ms=payload.get("duration_ms"),
        request_preview=payload.get("request"),
        response_preview=payload.get("response"),
        error=TraceErrorInfo(**error) if isinstance(error, dict) else None,
        metrics=dict(payload.get("metrics") or {}),
    )


def _parse_llm_ts(value: str | int | float | datetime | None) -> float:
    if not value:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _present_llm_row(row: dict[str, Any]) -> LLMCallRecord:
    scope = _scope_from_mapping(row)
    scope.trace_id = row.get("trace_id")
    scope.span_id = row.get("span_id")
    status = "error" if row.get("error_type") else "ok"
    call_name = row.get("call_name")
    return LLMCallRecord(
        id=str(row.get("call_id")),
        ts=_parse_llm_ts(row.get("created_at")),
        summary=f"{f'[{call_name}] ' if call_name else ''}{row.get('provider')}/{row.get('model')} {row.get('call_type')}",
        severity="error" if row.get("error_type") else "info",
        status=status,
        producer=InspectProducer(family="llm", name=str(row.get("provider") or "unknown")),
        scope=scope,
        tags=[str(row.get("call_type") or "chat"), status],
        payload={},
        call_id=str(row.get("call_id")),
        created_at=str(row.get("created_at")),
        call_type=str(row.get("call_type") or "chat"),
        provider=str(row.get("provider") or "unknown"),
        model=str(row.get("model") or "unknown"),
        profile_name=row.get("profile_name"),
        call_name=call_name,
        latency_ms=row.get("latency_ms"),
        usage=dict(row.get("usage") or {}),
        reasoning_effort=row.get("reasoning_effort"),
        output_format=row.get("output_format"),
        request_args=dict(row.get("request_args") or {}),
        provider_request_args=dict(row.get("provider_request_args") or {}),
        compatibility_notes=[str(item) for item in list(row.get("compatibility_notes") or [])],
        messages_preview=row.get("messages_preview"),
        trace_payload_preview=row.get("trace_payload_preview"),
        raw_text_preview=row.get("raw_text_preview"),
        messages=row.get("messages"),
        trace_payload=row.get("trace_payload"),
        raw_text=row.get("raw_text"),
        error_type=row.get("error_type"),
        error_message=row.get("error_message"),
    )


def _present_log_row(
    row: dict[str, Any], *, run_status: str | None = None, trace_status: str | None = None
) -> InspectLogRecord:
    inner = row.get("attributes") or {}
    scope = _scope_from_mapping(row)
    return InspectLogRecord(
        id=str(row.get("observation_id")),
        ts=_parse_llm_ts(row.get("occurred_at")),
        summary=str(row.get("summary") or inner.get("message") or ""),
        severity=str(row.get("severity") or inner.get("level") or "info"),
        status=str(row.get("status") or "ok"),
        producer=InspectProducer(family="logger", name=str(inner.get("logger") or "unknown")),
        scope=scope,
        tags=[str(inner.get("level") or "info")],
        payload=inner,
        logger=str(inner.get("logger") or "unknown"),
        level=str(inner.get("level") or "info"),
        message=str(inner.get("message") or ""),
        error=InspectLogError(**inner["error"]) if inner.get("error") else None,
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


def _present_engine_event_row(row: dict[str, Any]) -> AgentEventEnvelope:
    data = dict(row.get("data") or {})
    event_type = str(data.get("event_kind") or row.get("kind") or "agent_engine.unknown")
    scope = _scope_from_mapping({**row, **data})
    scope.trace_id = str(row.get("run_id") or "") or None
    links = InspectLinks(
        parent_event_id=data.get("parent_event_id"),
        caused_by_event_id=data.get("caused_by_event_id"),
    )
    status = str(data.get("status") or "info")
    event_id = str(row.get("event_id") or row.get("id") or "")
    return AgentEventEnvelope(
        id=event_id,
        ts=_parse_llm_ts(row.get("ts")),
        summary=str(row.get("text") or data.get("summary") or event_type),
        severity="error" if status in {"error", "failed"} else "info",
        status=status,
        producer=InspectProducer(family="engine", name="agent_engine"),
        scope=scope,
        tags=[str(tag) for tag in row.get("tags") or []],
        links=links,
        payload=data,
        event_id=event_id,
        event_type=event_type,
        payload_schema=InspectPayloadSchema(name="agent_engine", version=2),
    )


@dataclass
class StudioTranslationPresenter:
    event_log: Any | None
    store: Any | None
    run_store: Any | None = None
    identity: ObservabilityIdentity = ObservabilityIdentity()
    run_status_resolver: RunStatusResolver | None = None

    async def list_traces(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        trace_id: str | None = None,
        service: list[str] | None = None,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> TraceEventListResponse:
        """List explicitly captured service-operation observations.

        The query is read-only and applies identity and runtime scope before pagination.

        Examples:
            List one run:
                ```python
                page = await facade.list_traces(run_id="run-1")
                ```
            Filter services:
                ```python
                page = await facade.list_traces(service=["runner"], status="error")
                ```

        Args:
            since: Optional inclusive start time.
            until: Optional inclusive end time.
            run_id: Optional exact run scope.
            session_id: Optional exact session scope.
            agent_id: Optional exact agent scope.
            app_id: Optional exact application scope.
            graph_id: Optional exact graph scope.
            node_id: Optional exact node scope.
            trace_id: Optional exact operational trace identifier.
            service: Optional accepted service names.
            status: Optional exact trace status.
            cursor: Optional pagination cursor.
            limit: Maximum records returned.

        Returns:
            TraceEventListResponse: Scoped trace page and next cursor.

        Notes:
            Raises `InspectionUnavailableError` when no event log is configured.
        """
        user_id, org_id = _store_identity_scope(self.identity)
        rows = await self._require_llm_store().list_observations(
            ObservationFilter(
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                app_id=app_id,
                graph_id=graph_id,
                node_id=node_id,
                trace_id=trace_id,
                user_id=user_id,
                org_id=org_id,
            )
        )
        items = [
            _present_trace_row(row)
            for row in rows
            if row.get("category") in {"service_operation", "trace"}
            and (since is None or _parse_llm_ts(row.get("occurred_at")) >= since.timestamp())
            and (until is None or _parse_llm_ts(row.get("occurred_at")) <= until.timestamp())
        ]
        items = [
            item
            for item in items
            if _passes_identity_scope(item.scope, self.identity)
            and _matches_scope(
                item.scope,
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                app_id=app_id,
                graph_id=graph_id,
                node_id=node_id,
            )
            and (service is None or item.service in service)
            and (status is None or item.status == status)
        ]
        items.sort(key=lambda item: item.ts, reverse=True)
        page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
        return TraceEventListResponse(items=page, next_cursor=next_cursor)

    async def list_llm_calls(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        call_type: str | None = None,
        status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> LLMCallListResponse:
        """List sanitized LLM observation records from the configured store.

        Full prompts and raw responses remain available only through `get_llm_call`.

        Examples:
            List one run:
                ```python
                page = await facade.list_llm_calls(run_id="run-1")
                ```
            Filter failures:
                ```python
                page = await facade.list_llm_calls(status="error", provider="openai")
                ```

        Args:
            since: Optional inclusive start time.
            until: Optional inclusive end time.
            run_id: Optional exact run scope.
            session_id: Optional exact session scope.
            agent_id: Optional exact agent scope.
            app_id: Optional exact application scope.
            graph_id: Optional exact graph scope.
            node_id: Optional exact node scope.
            provider: Optional exact provider filter.
            model: Optional exact model filter.
            call_type: Optional exact call type.
            status: Optional `ok` or `error` filter.
            cursor: Optional pagination cursor.
            limit: Maximum records returned.

        Returns:
            LLMCallListResponse: Sanitized observation page and next cursor.

        Notes:
            Raises `InspectionUnavailableError` when no LLM store is configured.
        """
        store = self._require_llm_store()
        user_id, org_id = _store_identity_scope(self.identity)
        rows = await store.query_llm_calls(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            graph_id=graph_id,
            node_id=node_id,
            since=since,
            until=until,
            user_id=user_id,
            org_id=org_id,
            limit=None,
        )
        items = [_present_llm_row(row) for row in rows]
        items = [
            item
            for item in items
            if _passes_identity_scope(item.scope, self.identity)
            and (provider is None or item.provider == provider)
            and (model is None or item.model == model)
            and (call_type is None or item.call_type == call_type)
            and (status is None or item.status == status)
        ]
        items.sort(key=lambda item: item.ts, reverse=True)
        page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
        return LLMCallListResponse(items=page, next_cursor=next_cursor)

    async def get_llm_call(
        self, call_id: str, *, required_run_id: str | None = None
    ) -> LLMCallRecord:
        """Read one full LLM observation with optional run ownership enforcement.

        The detail lookup applies identity scope and hides cross-run records as missing.

        Examples:
            Read a connected record:
                ```python
                item = await facade.get_llm_call("call-1")
                ```
            Enforce Studio run ownership:
                ```python
                item = await facade.get_llm_call("call-1", required_run_id="run-1")
                ```

        Args:
            call_id: Exact LLM observation identifier.
            required_run_id: Optional run that must own the record.

        Returns:
            LLMCallRecord: Full scoped observation record.

        Notes:
            Raises `InspectionNotFoundError` for missing or out-of-scope records.
        """
        row = await self._require_llm_store().get_llm_call(call_id)
        if row is None or (required_run_id and row.get("run_id") != required_run_id):
            raise ObservabilityNotFoundError("LLM call not found")
        record = _present_llm_row(row)
        if not _passes_identity_scope(record.scope, self.identity):
            raise ObservabilityNotFoundError("LLM call not found")
        return record

    async def list_logs(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        level: str | None = None,
        logger: str | None = None,
        run_status: str | None = None,
        trace_status: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> InspectLogListResponse:
        """List structured inspection logs with run and trace status enrichment.

        The query reads only persisted event-log records and applies scope before pagination.

        Examples:
            List one run:
                ```python
                page = await facade.list_logs(run_id="run-1")
                ```
            Filter errors:
                ```python
                page = await facade.list_logs(level="error", trace_status="error")
                ```

        Args:
            since: Optional inclusive start time.
            until: Optional inclusive end time.
            run_id: Optional exact run scope.
            session_id: Optional exact session scope.
            agent_id: Optional exact agent scope.
            app_id: Optional exact application scope.
            graph_id: Optional exact graph scope.
            node_id: Optional exact node scope.
            level: Optional exact log level.
            logger: Optional exact logger name.
            run_status: Optional enriched run status.
            trace_status: Optional enriched trace status.
            cursor: Optional pagination cursor.
            limit: Maximum records returned.

        Returns:
            InspectLogListResponse: Scoped log page and next cursor.

        Notes:
            Run status is omitted when no resolver is configured.
        """
        user_id, org_id = _store_identity_scope(self.identity)
        rows = await self._require_llm_store().list_observations(
            ObservationFilter(
                category="log",
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                app_id=app_id,
                graph_id=graph_id,
                node_id=node_id,
                user_id=user_id,
                org_id=org_id,
            )
        )
        raw_run_ids = {row.get("run_id") for row in rows if row.get("run_id")}
        run_statuses = await self._resolve_run_statuses(raw_run_ids)
        trace_statuses = {
            str(row["trace_id"]): "error"
            for row in rows
            if row.get("trace_id") and str(row.get("status")) == "error"
        }
        items = [
            _present_log_row(
                row,
                run_status=run_statuses.get(row.get("run_id")),
                trace_status=trace_statuses.get(row.get("trace_id")),
            )
            for row in rows
            if (since is None or _parse_llm_ts(row.get("occurred_at")) >= since.timestamp())
            and (until is None or _parse_llm_ts(row.get("occurred_at")) <= until.timestamp())
        ]
        items = [
            item
            for item in items
            if _passes_identity_scope(item.scope, self.identity)
            and _matches_scope(
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
        items.sort(key=lambda item: item.ts, reverse=True)
        page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
        return InspectLogListResponse(items=page, next_cursor=next_cursor)

    async def list_agent_events(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        event_type: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
    ) -> AgentEventListResponse:
        """List structured agent events from the configured event log.

        Run and session scopes use the persisted event partition while other scopes are filtered.

        Examples:
            List one run:
                ```python
                page = await facade.list_agent_events(run_id="run-1")
                ```
            Filter event type:
                ```python
                page = await facade.list_agent_events(event_type="planning.started")
                ```

        Args:
            since: Optional inclusive start time.
            until: Optional inclusive end time.
            run_id: Optional exact run scope.
            session_id: Optional exact session scope.
            agent_id: Optional exact agent scope.
            app_id: Optional exact application scope.
            graph_id: Optional exact graph scope.
            node_id: Optional exact node scope.
            event_type: Optional exact event type.
            cursor: Optional pagination cursor.
            limit: Maximum records returned.

        Returns:
            AgentEventListResponse: Scoped agent-event page and next cursor.

        Notes:
            Raises `InspectionUnavailableError` when no event log is configured.
        """
        event_log = self._require_event_log()
        user_id, org_id = _store_identity_scope(self.identity)
        engine_rows = await event_log.query(
            since=since,
            until=until,
            tags=["agent_engine"],
            limit=None,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            graph_id=graph_id,
            node_id=node_id,
        )
        custom_rows = await event_log.query(
            scope_id=run_id or (f"session:{session_id}" if session_id else None),
            since=since,
            until=until,
            kinds=["agent_event"],
            limit=None,
            user_id=user_id,
            org_id=org_id,
        )
        items = [
            *(_present_engine_event_row(row) for row in engine_rows),
            *(_present_agent_row(row) for row in custom_rows),
        ]
        items = [
            item
            for item in items
            if _passes_identity_scope(item.scope, self.identity)
            and _matches_scope(
                item.scope,
                run_id=run_id,
                session_id=session_id,
                agent_id=agent_id,
                app_id=app_id,
                graph_id=graph_id,
                node_id=node_id,
            )
            and (event_type is None or item.event_type == event_type)
        ]
        items.sort(key=lambda item: item.ts, reverse=True)
        page, next_cursor = _paginate_rows(items, cursor=cursor, limit=limit)
        return AgentEventListResponse(items=page, next_cursor=next_cursor)

    async def _list_trace_sessions(
        self, *, limit: int = 50, cursor: str | None = None
    ) -> dict[str, Any]:
        runs = await self._list_runs()
        events = await self._engine_rows()
        events_by_run: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            events_by_run.setdefault(str(event.get("run_id") or ""), []).append(event)
        groups: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            run_id = _run_text(run, "run_id")
            session_id = _run_text(run, "session_id") or run_id
            groups.setdefault(session_id, []).append(
                self._run_summary(run, events_by_run.get(run_id, []))
            )
        items: list[dict[str, Any]] = []
        for session_id, turns in groups.items():
            turns.sort(key=lambda item: item["started_at"], reverse=True)
            latest = turns[0]
            items.append(
                {
                    "session_id": session_id,
                    "latest_trace_id": latest["trace_id"],
                    "latest_started_at": latest["started_at"],
                    "latest_status": latest["status"],
                    "turn_count": len(turns),
                    "turns": turns,
                }
            )
        items.sort(key=lambda item: item["latest_started_at"], reverse=True)
        try:
            offset = max(0, int(cursor or "0"))
        except ValueError:
            offset = 0
        page = items[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "items": page,
            "next_cursor": str(next_offset) if next_offset < len(items) else None,
            "has_more": next_offset < len(items),
        }

    async def _inspect_trace(self, trace_id: str) -> dict[str, Any] | None:
        run = await self._get_run(trace_id)
        if run is None:
            return None
        events = await self._engine_rows(run_id=trace_id)
        summary = self._run_summary(run, events)
        return {
            "root_trace_id": trace_id,
            "selected_trace_id": trace_id,
            "runs": [self._run_tree_summary(run, summary)],
            "spans_by_trace_id": {trace_id: self._trace_spans(trace_id, events)},
            "plans_by_trace_id": {trace_id: self._trace_plans(events)},
            "agent_states_index": {},
            "graph_topologies_by_trace_id": {trace_id: self._trace_graph(run, events)},
            "agent_states_by_trace_id": {trace_id: {}},
            "context_snapshots": [],
        }

    async def _get_trace_graph(self, trace_id: str) -> dict[str, Any] | None:
        run = await self._get_run(trace_id)
        if run is None:
            return None
        return self._trace_graph(run, await self._engine_rows(run_id=trace_id))

    async def _get_trace_spans(
        self,
        trace_id: str,
        *,
        kind: str | None = None,
        agent_id: str | None = None,
    ) -> dict[str, Any] | None:
        if await self._get_run(trace_id) is None:
            return None
        items = self._trace_spans(trace_id, await self._engine_rows(run_id=trace_id))
        return {
            "items": [
                item
                for item in items
                if (kind is None or item["kind"] == kind)
                and (agent_id is None or item["agent_instance_id"] == agent_id)
            ]
        }

    async def _get_trace_plans(self, trace_id: str) -> dict[str, Any] | None:
        if await self._get_run(trace_id) is None:
            return None
        return {"items": self._trace_plans(await self._engine_rows(run_id=trace_id))}

    async def _get_context_snapshot(self, trace_id: str, snapshot_id: str) -> dict[str, Any] | None:
        events = await self._engine_rows(run_id=trace_id)
        decision = next(
            (
                row
                for row in events
                if str((row.get("data") or {}).get("prompt_manifest_id") or "") == snapshot_id
            ),
            None,
        )
        if decision is None:
            return None
        manifest = await self._require_llm_store().hydrate_prompt_manifest(snapshot_id)
        if manifest is None:
            return None
        data = dict(decision.get("data") or {})
        parts = list(manifest.get("parts") or [])
        sections = [
            {
                "key": str(part.get("semantic_kind") or f"part:{index}"),
                "present": True,
                "value_type": str(part.get("content_kind") or "json"),
                "char_count": int(part.get("byte_count") or 0),
                "hash": str(part.get("fragment_id") or ""),
                "preview": "",
                "preview_truncated": False,
                "body_truncated": False,
                "omitted": manifest.get("provider_request") is None,
                "truncated": False,
                "omission_reason": manifest.get("omission_reason"),
            }
            for index, part in enumerate(parts)
        ]
        provider_request = manifest.get("provider_request")
        body_sections: list[dict[str, Any]] = []
        if isinstance(provider_request, dict):
            for index, message in enumerate(provider_request.get("messages") or []):
                body_sections.append(
                    {
                        "key": f"message:{index}:{message.get('role') or 'unknown'}",
                        "value": message,
                        "omitted": False,
                        "truncated": False,
                    }
                )
        return {
            "snapshot_id": snapshot_id,
            "trace_id": trace_id,
            "span_id": str(decision.get("event_id") or ""),
            "session_id": str(decision.get("session_id") or ""),
            "agent_instance_id": str(data.get("agent_instance_id") or ""),
            "step_index": int(data.get("step_index") or 0),
            "capture_mode": str(manifest.get("capture_mode") or "metadata"),
            "created_at": str(decision.get("ts") or ""),
            "summary": {
                "version": 2,
                "capture_mode": manifest.get("capture_mode"),
                "step_index": data.get("step_index"),
                "section_count": len(sections),
                "total_chars": int(manifest.get("total_chars") or 0),
                "body_truncated": False,
                "sections": sections,
            },
            "body": {
                "version": 2,
                "capture_mode": manifest.get("capture_mode"),
                "step_index": data.get("step_index"),
                "sections": body_sections,
                "omission_reason": manifest.get("omission_reason"),
            },
        }

    async def _get_agent_states(self, trace_id: str, agent_id: str) -> dict[str, Any] | None:
        del agent_id
        if await self._get_run(trace_id) is None:
            return None
        return {"items": []}

    async def _engine_rows(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        rows = await self._require_event_log().query(
            tags=["agent_engine"],
            run_id=run_id,
            limit=None,
            order_dir="asc",
        )
        return [
            row for row in rows if _passes_identity_scope(_scope_from_mapping(row), self.identity)
        ]

    async def _list_runs(self) -> list[Any]:
        if self.run_store is None:
            raise ObservabilityUnavailableError("Run store not configured")
        return list(await self.run_store.list(limit=10_000, offset=0))

    async def _get_run(self, run_id: str) -> Any | None:
        if self.run_store is None:
            raise ObservabilityUnavailableError("Run store not configured")
        run = await self.run_store.get(run_id)
        if run is None:
            return None
        scope = _scope_from_mapping(_run_mapping(run))
        return run if _passes_identity_scope(scope, self.identity) else None

    def _run_summary(self, run: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
        kinds = [str(row.get("kind") or "") for row in events]
        agents = {
            str((row.get("data") or {}).get("agent_instance_id") or "")
            for row in events
            if (row.get("data") or {}).get("agent_instance_id")
        }
        return {
            "trace_id": _run_text(run, "run_id"),
            "session_id": _run_text(run, "session_id") or _run_text(run, "run_id"),
            "graph_id": _run_text(run, "graph_id"),
            "started_at": _run_time(run, "started_at"),
            "ended_at": _run_time(run, "finished_at") or None,
            "status": _run_status(run),
            "span_count": len(self._trace_spans(_run_text(run, "run_id"), events)),
            "agent_count": len(agents),
            "plan_count": sum(".plan_" in kind for kind in kinds),
        }

    def _run_tree_summary(self, run: Any, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            **summary,
            "root_trace_id": summary["trace_id"],
            "parent_trace_id": "",
            "parent_span_id": "",
            "run_id": summary["trace_id"],
            "parent_run_id": "",
            "trace_role": "root",
            "dispatch_mode": "",
            "agent_name": _run_text(run, "agent_id"),
            "agent_instance_id": _run_text(run, "agent_id"),
        }

    def _trace_spans(self, trace_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result_by_cause = {
            str((row.get("data") or {}).get("caused_by_event_id") or ""): row
            for row in events
            if row.get("kind") == "agent_engine.tool_result"
        }
        returned_by_token = {
            str((row.get("data") or {}).get("dispatch_token") or ""): row
            for row in events
            if row.get("kind") == "agent_engine.dispatch_returned"
        }
        spans: list[dict[str, Any]] = []
        for sequence_no, row in enumerate(events):
            event_kind = str(row.get("kind") or "")
            if (
                event_kind
                in {
                    "agent_engine.tool_result",
                    "agent_engine.dispatch_returned",
                }
                or ".plan_" in event_kind
            ):
                continue
            data = dict(row.get("data") or {})
            kind = _legacy_span_kind(event_kind)
            if kind is None:
                continue
            event_id = str(row.get("event_id") or row.get("id") or "")
            paired = None
            if event_kind == "agent_engine.tool_call":
                paired = result_by_cause.get(event_id)
            elif event_kind == "agent_engine.dispatch_entered":
                paired = returned_by_token.get(str(data.get("dispatch_token") or ""))
            paired_data = dict((paired or {}).get("data") or {})
            payload = {**data}
            if paired is not None:
                payload["result_summary"] = str(paired.get("text") or "")
                payload["result"] = paired_data.get("result")
                payload["resource_links"] = paired_data.get("resource_links") or []
                payload["result_status"] = paired_data.get("status")
            if event_kind == "agent_engine.decision":
                payload["context_snapshot_id"] = data.get("prompt_manifest_id") or ""
            status = str(paired_data.get("status") or data.get("status") or "info")
            spans.append(
                {
                    "span_id": event_id,
                    "parent_span_id": str(
                        data.get("parent_event_id") or data.get("caused_by_event_id") or ""
                    ),
                    "trace_id": trace_id,
                    "session_id": str(row.get("session_id") or ""),
                    "kind": kind,
                    "agent_instance_id": str(data.get("agent_instance_id") or ""),
                    "started_at": str(row.get("ts") or ""),
                    "ended_at": str((paired or row).get("ts") or ""),
                    "status": status,
                    "summary": str(row.get("text") or event_kind),
                    "sequence_no": sequence_no,
                    "payload": payload,
                    "run_trace_id": trace_id,
                    "root_trace_id": trace_id,
                    "trace_role": "root",
                    "dispatch_mode": str(data.get("dispatch_mode") or ""),
                    "agent_name": str(data.get("agent_id") or ""),
                    "run_started_at": "",
                }
            )
            manifest_id = str(data.get("prompt_manifest_id") or "")
            if event_kind == "agent_engine.decision" and manifest_id:
                spans.append(
                    {
                        **spans[-1],
                        "span_id": f"{event_id}:context",
                        "parent_span_id": event_id,
                        "kind": "context_composition",
                        "summary": "Prompt context composition",
                        "payload": {
                            "context_snapshot_id": manifest_id,
                            "prompt_manifest_id": manifest_id,
                            "new_context_entry_ids": data.get("new_context_entry_ids") or [],
                            "dynamic_context_summary": data.get("dynamic_context_summary") or {},
                        },
                    }
                )
        return spans

    def _trace_plans(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for row in events:
            if ".plan_" not in str(row.get("kind") or ""):
                continue
            data = dict(row.get("data") or {})
            plan = data.get("plan")
            if not isinstance(plan, dict):
                continue
            snapshots.append(
                {
                    "captured_at": str(row.get("ts") or ""),
                    "run_trace_id": str(row.get("run_id") or ""),
                    "trace_role": "root",
                    "agent_name": str(data.get("agent_id") or ""),
                    "plan": dict(plan),
                }
            )
        return snapshots

    def _trace_graph(self, run: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}
        for row in events:
            data = dict(row.get("data") or {})
            agent_id = str(data.get("agent_instance_id") or data.get("agent_id") or "")
            if agent_id:
                nodes.setdefault(
                    agent_id,
                    {
                        "node_id": agent_id,
                        "node_kind": "agent",
                        "target_agent_instance_id": agent_id,
                        "entry": not nodes,
                        "agent_name": str(data.get("agent_id") or agent_id),
                    },
                )
            if row.get("kind") != "agent_engine.dispatch_entered":
                continue
            source = str(data.get("source_agent_instance_id") or "")
            target = str(data.get("target_agent_instance_id") or "")
            token = str(data.get("dispatch_token") or row.get("event_id") or "")
            for node_id in (source, target):
                if node_id:
                    nodes.setdefault(
                        node_id,
                        {
                            "node_id": node_id,
                            "node_kind": "agent",
                            "target_agent_instance_id": node_id,
                            "entry": not nodes,
                            "agent_name": node_id,
                        },
                    )
            if source and target:
                edges[token] = {
                    "edge_id": token,
                    "source_node_id": source,
                    "target_node_id": target,
                    "dispatch_mode": str(data.get("dispatch_mode") or ""),
                }
        return {"graph_id": _run_text(run, "graph_id"), "nodes": nodes, "edges": edges}

    def _require_event_log(self) -> Any:
        if self.event_log is None:
            raise ObservabilityUnavailableError("Event log not configured")
        return self.event_log

    def _require_llm_store(self) -> Any:
        if self.store is None:
            raise ObservabilityUnavailableError("Observation store not configured")
        return self.store

    async def _resolve_run_statuses(self, run_ids: set[str]) -> Mapping[str, str]:
        if not run_ids or self.run_status_resolver is None:
            return {}
        return await self.run_status_resolver(run_ids)


def _run_mapping(run: Any) -> dict[str, Any]:
    if isinstance(run, dict):
        return dict(run)
    return {
        name: getattr(run, name, None)
        for name in (
            "run_id",
            "session_id",
            "graph_id",
            "agent_id",
            "app_id",
            "user_id",
            "org_id",
            "started_at",
            "finished_at",
            "status",
        )
    }


def _run_text(run: Any, name: str) -> str:
    return str(_run_mapping(run).get(name) or "")


def _run_time(run: Any, name: str) -> str:
    value = _run_mapping(run).get(name)
    return value.isoformat() if isinstance(value, datetime) else str(value or "")


def _run_status(run: Any) -> str:
    value = _run_mapping(run).get("status")
    return str(getattr(value, "value", value) or "unknown")


def _legacy_span_kind(event_kind: str) -> str | None:
    return {
        "agent_engine.user_request": "graph_turn",
        "agent_engine.agent_entered": "agent_dispatch",
        "agent_engine.agent_exited": "agent_dispatch",
        "agent_engine.dispatch_entered": "agent_dispatch",
        "agent_engine.decision": "react_cycle",
        "agent_engine.tool_call": "tool_call",
        "agent_engine.action_validation_failed": "action_validation_failed",
        "agent_engine.runtime_error": "runtime_error",
        "agent_engine.interaction_waited": "interaction",
        "agent_engine.interaction_resumed": "interaction",
    }.get(event_kind)


__all__ = [
    "ObservabilityIdentity",
    "ObservabilityNotFoundError",
    "ObservabilityUnavailableError",
    "ObservabilityWorkspaceError",
    "StudioTranslationPresenter",
]

"""Generic inspection presentation over raw AetherGraph observations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aethergraph.api.v1.pagination import decode_cursor, encode_cursor
from aethergraph.observability.contracts import (
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
        attempt_count=int(row.get("attempt_count") or 0),
        retry_count=int(row.get("retry_count") or 0),
        total_retry_wait_ms=int(row.get("total_retry_wait_ms") or 0),
        attempts=list(row.get("attempts") or []),
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
class InspectionPresenter:
    event_log: Any | None
    engine_event_log: Any | None
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
        engine_event_log = self._require_engine_event_log()
        event_log = self._require_event_log()
        user_id, org_id = _store_identity_scope(self.identity)
        engine_rows = await engine_event_log.query(
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

    def _require_event_log(self) -> Any:
        if self.event_log is None:
            raise ObservabilityUnavailableError("Event log not configured")
        return self.event_log

    def _require_engine_event_log(self) -> Any:
        if self.engine_event_log is None:
            raise ObservabilityUnavailableError("Canonical engine event log not configured")
        return self.engine_event_log

    def _require_llm_store(self) -> Any:
        if self.store is None:
            raise ObservabilityUnavailableError("Observation store not configured")
        return self.store

    async def _resolve_run_statuses(self, run_ids: set[str]) -> Mapping[str, str]:
        if not run_ids or self.run_status_resolver is None:
            return {}
        return await self.run_status_resolver(run_ids)


__all__ = [
    "InspectionPresenter",
    "ObservabilityIdentity",
    "ObservabilityNotFoundError",
    "ObservabilityUnavailableError",
    "ObservabilityWorkspaceError",
]

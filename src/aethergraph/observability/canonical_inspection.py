"""Inactive bounded Inspect projection over canonical observation storage."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Any

from aethergraph.services.canonical_storage_scope import merge_storage_scope
from aethergraph.storage.contracts import (
    LLMCallDetail,
    LLMCallQuery,
    LLMCallRecord as CanonicalLLMCallRecord,
    ObservationQuery,
    ObservationRecord as CanonicalObservationRecord,
    ObservationSeverity,
    ObservationStatus,
    PageRequest,
    StorageScope,
)

from .canonical_service import CanonicalObservationService
from .contracts import (
    InspectLinks,
    InspectLogError,
    InspectLogListResponse,
    InspectLogRecord,
    InspectProducer,
    InspectScope,
    LLMCallAttempt,
    LLMCallListResponse,
    LLMCallRecord,
    TraceErrorInfo,
    TraceEvent,
    TraceEventListResponse,
)
from .inspection import (
    ObservabilityIdentity,
    ObservabilityNotFoundError,
)

RunStatusResolver = Callable[[set[str]], Awaitable[Mapping[str, str]]]
_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_LOCAL_IDENTITY = ObservabilityIdentity()


class CanonicalInspectionReader:
    """Project bounded canonical observation pages into current Inspect DTOs."""

    def __init__(
        self,
        service: CanonicalObservationService,
        *,
        identity: ObservabilityIdentity = _LOCAL_IDENTITY,
        run_status_resolver: RunStatusResolver | None = None,
    ) -> None:
        """Bind Inspect reads to one canonical observation service and identity.

        Construction performs no repository I/O. Every later list call uses one
        provider cursor page and exact promoted filters before presentation.

        Examples:
            Bind local inspection:
                ```python
                reader = CanonicalInspectionReader(service)
                ```

            Bind cloud identity and run enrichment:
                ```python
                reader = CanonicalInspectionReader(
                    service,
                    identity=ObservabilityIdentity(
                        mode="cloud",
                        user_id="user-1",
                        org_id="org-1",
                    ),
                    run_status_resolver=resolve_statuses,
                )
                ```

        Args:
            service: Canonical observation service and trusted owner scope.
            identity: Read identity merged into every canonical storage query.
            run_status_resolver: Optional bounded batch resolver for visible run IDs.

        Returns:
            None: The inactive-until-S9 reader is ready.

        Notes:
            The reader neither owns nor closes the service or its storage bundle.
        """
        self.service = service
        self.identity = identity
        self.run_status_resolver = run_status_resolver

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
        """List one provider-cursor page of canonical trace observations.

        Scope, category, producer, status, trace, and time predicates execute in the
        provider before bounded pagination; presentation never performs a broad scan.

        Examples:
            List one run:
                ```python
                page = await reader.list_traces(run_id="run-1")
                ```

            Continue filtered services:
                ```python
                page = await reader.list_traces(
                    service=["runner"],
                    cursor=previous.next_cursor,
                    limit=50,
                )
                ```

        Args:
            since: Optional inclusive occurrence lower bound.
            until: Optional inclusive occurrence upper bound.
            run_id: Optional exact canonical run scope.
            session_id: Optional exact canonical session scope.
            agent_id: Optional exact canonical agent scope.
            app_id: Deprecated optional compatibility-metadata filter.
            graph_id: Optional exact canonical graph scope.
            node_id: Optional exact canonical node scope.
            trace_id: Optional exact observation trace identity.
            service: Optional exact promoted producer names.
            status: Optional exact canonical observation status.
            cursor: Optional provider-authored continuation cursor.
            limit: Maximum records in the provider page.

        Returns:
            TraceEventListResponse: Presented page with the provider continuation cursor.

        Notes:
            Deprecated `app_id` is never a provider query or authorization dimension;
            it may only remove rows from the already-bounded page.
        """
        scope = self._scope(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            graph_id=graph_id,
            node_id=node_id,
        )
        if scope is None:
            return TraceEventListResponse(items=[], next_cursor=None)
        statuses = (ObservationStatus(status),) if status is not None else ()
        page = await self.service.repository.query(
            ObservationQuery(
                scope=scope,
                page=PageRequest(limit=limit, cursor=cursor),
                categories=("service_operation", "trace"),
                producers=tuple(service or ()),
                statuses=statuses,
                trace_id=trace_id,
                occurred_at_or_after=since,
                occurred_at_or_before=until,
            )
        )
        items = [
            _trace(record)
            for record in page.items
            if app_id is None or _deprecated_app_id(record) == app_id
        ]
        return TraceEventListResponse(items=items, next_cursor=page.next_cursor)

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
        """List one metadata-only provider-cursor page of canonical LLM calls.

        Promoted scope, provider, model, call type, status, and time predicates are
        applied before pagination. Captured prompt and response bodies stay excluded.

        Examples:
            List failed calls:
                ```python
                page = await reader.list_llm_calls(status="error")
                ```

            Continue one provider/model page:
                ```python
                page = await reader.list_llm_calls(
                    provider="openai",
                    model="gpt-test",
                    cursor=previous.next_cursor,
                )
                ```

        Args:
            since: Optional inclusive occurrence lower bound.
            until: Optional inclusive occurrence upper bound.
            run_id: Optional exact canonical run scope.
            session_id: Optional exact canonical session scope.
            agent_id: Optional exact canonical agent scope.
            app_id: Deprecated optional compatibility-metadata filter.
            graph_id: Optional exact canonical graph scope.
            node_id: Optional exact canonical node scope.
            provider: Optional exact provider name.
            model: Optional exact model name.
            call_type: Optional exact logical call type.
            status: Optional exact canonical observation status.
            cursor: Optional provider-authored continuation cursor.
            limit: Maximum metadata records in the provider page.

        Returns:
            LLMCallListResponse: Metadata-only page and provider continuation cursor.

        Notes:
            Deprecated `app_id` may only post-filter the bounded page. Exact content
            hydration is available solely through `get_llm_call`.
        """
        scope = self._scope(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            graph_id=graph_id,
            node_id=node_id,
        )
        if scope is None:
            return LLMCallListResponse(items=[], next_cursor=None)
        page = await self.service.repository.query_llm_calls(
            LLMCallQuery(
                scope=scope,
                page=PageRequest(limit=limit, cursor=cursor),
                providers=(provider,) if provider is not None else (),
                models=(model,) if model is not None else (),
                call_types=(call_type,) if call_type is not None else (),
                statuses=(ObservationStatus(status),) if status is not None else (),
                occurred_at_or_after=since,
                occurred_at_or_before=until,
            )
        )
        items = [
            _llm(record)
            for record in page.items
            if app_id is None or _deprecated_app_id(record.observation) == app_id
        ]
        return LLMCallListResponse(items=items, next_cursor=page.next_cursor)

    async def get_llm_call(
        self,
        call_id: str,
        *,
        required_run_id: str | None = None,
    ) -> LLMCallRecord:
        """Hydrate one exact canonical LLM call after scoped authorization.

        The repository performs one exact scoped detail read. Missing, cross-owner,
        and wrong-run records all project to the same not-found result.

        Examples:
            Read one call:
                ```python
                call = await reader.get_llm_call("call-1")
                ```

            Require Studio run ownership:
                ```python
                call = await reader.get_llm_call(
                    "call-1",
                    required_run_id="run-1",
                )
                ```

        Args:
            call_id: Exact canonical LLM call identity.
            required_run_id: Optional exact run that must own the call.

        Returns:
            LLMCallRecord: Presented call with policy-retained detail content.

        Notes:
            No manifest/path hydration or historical fallback is attempted after a miss.
        """
        scope = self._scope(run_id=required_run_id)
        if scope is None:
            raise ObservabilityNotFoundError("LLM call not found")
        detail = await self.service.repository.get_llm_call(scope, call_id)
        if detail is None:
            raise ObservabilityNotFoundError("LLM call not found")
        return _llm(detail.record, detail=detail)

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
        """List one provider-cursor page of canonical structured logs.

        Scope, log category, promoted logger producer, severity, and time filters run
        before pagination; run and trace status enrichment is page-bounded.

        Examples:
            List error logs:
                ```python
                page = await reader.list_logs(level="error")
                ```

            Continue one logger page:
                ```python
                page = await reader.list_logs(
                    logger="aethergraph.runner",
                    cursor=previous.next_cursor,
                )
                ```

        Args:
            since: Optional inclusive occurrence lower bound.
            until: Optional inclusive occurrence upper bound.
            run_id: Optional exact canonical run scope.
            session_id: Optional exact canonical session scope.
            agent_id: Optional exact canonical agent scope.
            app_id: Deprecated optional compatibility-metadata filter.
            graph_id: Optional exact canonical graph scope.
            node_id: Optional exact canonical node scope.
            level: Optional exact canonical log severity.
            logger: Optional exact promoted logger producer.
            run_status: Optional page-bounded enriched run-status filter.
            trace_status: Optional page-bounded trace-status filter.
            cursor: Optional provider-authored continuation cursor.
            limit: Maximum records in the provider page.

        Returns:
            InspectLogListResponse: Presented page with provider continuation cursor.

        Notes:
            Deprecated App and enriched status filters may produce a sparse page; the
            provider cursor remains authoritative and can continue the query safely.
        """
        scope = self._scope(
            run_id=run_id,
            session_id=session_id,
            agent_id=agent_id,
            graph_id=graph_id,
            node_id=node_id,
        )
        if scope is None:
            return InspectLogListResponse(items=[], next_cursor=None)
        severities = (ObservationSeverity(level),) if level is not None else ()
        page = await self.service.repository.query(
            ObservationQuery(
                scope=scope,
                page=PageRequest(limit=limit, cursor=cursor),
                categories=("log",),
                producers=(logger,) if logger is not None else (),
                severities=severities,
                occurred_at_or_after=since,
                occurred_at_or_before=until,
            )
        )
        run_ids = {record.scope.run_id for record in page.items if record.scope.run_id}
        run_statuses = await self._run_statuses(run_ids)
        trace_statuses = {
            record.trace_id: ObservationStatus.ERROR.value
            for record in page.items
            if record.trace_id and record.status is ObservationStatus.ERROR
        }
        items = []
        for record in page.items:
            resolved_run_status = run_statuses.get(record.scope.run_id)
            resolved_trace_status = trace_statuses.get(record.trace_id)
            if app_id is not None and _deprecated_app_id(record) != app_id:
                continue
            if run_status is not None and resolved_run_status != run_status:
                continue
            if trace_status is not None and resolved_trace_status != trace_status:
                continue
            items.append(
                _log(
                    record,
                    run_status=resolved_run_status,
                    trace_status=resolved_trace_status,
                )
            )
        return InspectLogListResponse(items=items, next_cursor=page.next_cursor)

    def _scope(self, **dimensions: str | None) -> StorageScope | None:
        values = {name: value for name, value in dimensions.items() if value is not None}
        if self.identity.mode in {"cloud", "demo"}:
            if self.identity.user_id is None:
                return None
            values["user_id"] = self.identity.user_id
            if self.identity.org_id is not None:
                values["org_id"] = self.identity.org_id
        try:
            return merge_storage_scope(self.service.owner_scope, **values)
        except ValueError:
            return None

    async def _run_statuses(self, run_ids: set[str]) -> Mapping[str, str]:
        if not run_ids or self.run_status_resolver is None:
            return {}
        return await self.run_status_resolver(run_ids)


def _attributes(record: CanonicalObservationRecord) -> dict[str, Any]:
    return {
        key: value for key, value in record.attributes.items() if key != _COMPATIBILITY_METADATA
    }


def _deprecated_app_id(record: CanonicalObservationRecord) -> str | None:
    envelope = record.attributes.get(_COMPATIBILITY_METADATA)
    if not isinstance(envelope, Mapping):
        return None
    app = envelope.get(_DEPRECATED_APP_ID)
    if not isinstance(app, Mapping):
        return None
    if app.get("deprecated") is not True or app.get("compatibility_only") is not True:
        return None
    value = app.get("value")
    return value if isinstance(value, str) and value else None


def _scope(record: CanonicalObservationRecord) -> InspectScope:
    return InspectScope(
        org_id=record.scope.org_id,
        user_id=record.scope.user_id,
        run_id=record.scope.run_id,
        session_id=record.scope.session_id,
        agent_id=record.scope.agent_id,
        app_id=_deprecated_app_id(record),
        graph_id=record.scope.graph_id,
        node_id=record.scope.node_id,
        trace_id=record.trace_id,
        span_id=record.observation_id,
    )


def _trace(record: CanonicalObservationRecord) -> TraceEvent:
    payload = _attributes(record)
    service = record.producer or str(payload.get("service") or "runtime")
    operation = str(payload.get("operation") or record.name)
    error = payload.get("error")
    return TraceEvent(
        id=record.observation_id,
        ts=record.occurred_at.timestamp(),
        summary=record.summary,
        severity=record.severity.value,
        status=record.status.value,
        producer=InspectProducer(family="trace", name=service),
        scope=_scope(record),
        tags=[record.category],
        links=InspectLinks(
            parent_event_id=record.parent_observation_id,
            caused_by_event_id=record.caused_by_observation_id,
        ),
        payload=payload,
        trace_id=record.trace_id or record.scope.run_id or "",
        span_id=record.observation_id,
        parent_span_id=record.parent_observation_id,
        service=service,
        operation=operation,
        phase=str(payload.get("phase") or "event"),
        duration_ms=payload.get("duration_ms"),
        request_preview=payload.get("request"),
        response_preview=payload.get("response"),
        error=TraceErrorInfo(**error) if isinstance(error, Mapping) else None,
        metrics=dict(payload.get("metrics") or {}),
    )


def _llm(
    record: CanonicalLLMCallRecord,
    *,
    detail: LLMCallDetail | None = None,
) -> LLMCallRecord:
    options = dict(record.request_options)
    attempts = [
        LLMCallAttempt(
            attempt_number=attempt.attempt_number,
            elapsed_ms=attempt.elapsed_ms,
            outcome=attempt.outcome,
            retryable=attempt.retryable,
            status_code=attempt.status_code,
            error_code=attempt.error_code,
            request_id=attempt.request_id,
            provider_delay_ms=attempt.provider_delay_ms,
            scheduled_delay_ms=attempt.scheduled_delay_ms,
            rate_limits=list(attempt.rate_limits),
        )
        for attempt in record.attempts
    ]
    captured_request = detail.captured_request if detail is not None else None
    captured_response = detail.captured_response if detail is not None else None
    messages = captured_request.get("messages") if isinstance(captured_request, Mapping) else None
    raw_text = captured_response.get("text") if isinstance(captured_response, Mapping) else None
    status = record.observation.status.value
    return LLMCallRecord(
        id=record.llm_call_id,
        ts=record.observation.occurred_at.timestamp(),
        summary=record.observation.summary,
        severity=record.observation.severity.value,
        status=status,
        producer=InspectProducer(family="llm", name=record.provider),
        scope=_scope(record.observation),
        tags=[record.call_type, status],
        payload={},
        call_id=record.llm_call_id,
        created_at=record.observation.occurred_at.isoformat(),
        call_type=record.call_type,
        provider=record.provider,
        model=record.model,
        profile_name=record.profile_name,
        call_name=record.call_name,
        latency_ms=record.latency_ms,
        usage=dict(record.usage),
        reasoning_effort=options.get("reasoning_effort"),
        output_format=options.get("output_format"),
        request_args=dict(options.get("request_args") or {}),
        provider_request_args=dict(options.get("provider_request_args") or {}),
        compatibility_notes=[str(item) for item in options.get("compatibility_notes") or ()],
        messages_preview=record.request_preview,
        trace_payload_preview=record.trace_payload_preview,
        raw_text_preview=record.response_preview,
        messages=messages,
        trace_payload=detail.trace_payload if detail is not None else None,
        raw_text=raw_text,
        error_type=record.error_type,
        error_message=record.error_message,
        attempt_count=len(attempts),
        retry_count=max(0, len(attempts) - 1),
        total_retry_wait_ms=sum(attempt.scheduled_delay_ms or 0 for attempt in attempts),
        attempts=attempts,
    )


def _log(
    record: CanonicalObservationRecord,
    *,
    run_status: str | None,
    trace_status: str | None,
) -> InspectLogRecord:
    payload = _attributes(record)
    logger = record.producer or str(payload.get("logger") or "unknown")
    level = str(payload.get("level") or record.severity.value)
    error = payload.get("error")
    return InspectLogRecord(
        id=record.observation_id,
        ts=record.occurred_at.timestamp(),
        summary=record.summary,
        severity=record.severity.value,
        status=record.status.value,
        producer=InspectProducer(family="logger", name=logger),
        scope=_scope(record),
        tags=[level],
        payload=payload,
        logger=logger,
        level=level,
        message=str(payload.get("message") or ""),
        error=InspectLogError(**error) if isinstance(error, Mapping) else None,
        extra=dict(payload.get("extra") or {}),
        run_status=run_status,
        trace_status=trace_status,
    )


__all__ = ["CanonicalInspectionReader"]

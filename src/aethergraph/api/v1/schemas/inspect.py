from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field  # type: ignore


class InspectProducer(BaseModel):
    family: str
    name: str
    version: str | None = None


class InspectScope(BaseModel):
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None


class InspectLinks(BaseModel):
    parent_event_id: str | None = None
    caused_by_event_id: str | None = None


class InspectPayloadSchema(BaseModel):
    name: str | None = None
    version: int | None = None


class TraceErrorInfo(BaseModel):
    type: str | None = None
    message: str | None = None


class TraceEvent(BaseModel):
    id: str
    ts: float
    kind: str = "trace"
    summary: str
    severity: str
    status: str
    producer: InspectProducer
    scope: InspectScope
    tags: list[str] = Field(default_factory=list)
    links: InspectLinks | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    service: str
    operation: str
    phase: str
    duration_ms: int | float | None = None
    request_preview: dict[str, Any] | None = None
    response_preview: dict[str, Any] | None = None
    error: TraceErrorInfo | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)


class TraceEventListResponse(BaseModel):
    items: list[TraceEvent]
    next_cursor: str | None = None


class TraceSummary(BaseModel):
    run_id: str
    trace_ids: list[str] = Field(default_factory=list)
    span_count: int = 0
    error_count: int = 0
    total_duration_ms: int = 0
    top_failing_services: dict[str, int] = Field(default_factory=dict)
    latest_error_ts: float | None = None


class LLMCallRecord(BaseModel):
    id: str
    ts: float
    kind: str = "llm_call"
    summary: str
    severity: str
    status: str
    producer: InspectProducer
    scope: InspectScope
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    call_id: str
    created_at: str
    call_type: str
    provider: str
    model: str
    profile_name: str | None = None
    call_name: str | None = None
    latency_ms: int | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    reasoning_effort: str | None = None
    output_format: str | None = None
    messages_preview: dict[str, Any] | None = None
    trace_payload_preview: dict[str, Any] | None = None
    raw_text_preview: dict[str, Any] | None = None
    messages: list[dict[str, Any]] | None = None
    trace_payload: dict[str, Any] | None = None
    raw_text: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class LLMCallListResponse(BaseModel):
    items: list[LLMCallRecord]
    next_cursor: str | None = None


class LLMSummary(BaseModel):
    run_id: str
    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    error_count: int = 0
    by_model: dict[str, int] = Field(default_factory=dict)


class InspectLogError(BaseModel):
    type: str | None = None
    message: str | None = None
    detail: str | None = None


class InspectLogRecord(BaseModel):
    id: str
    ts: float
    kind: str = "inspect_log"
    summary: str
    severity: str
    status: str
    producer: InspectProducer
    scope: InspectScope
    tags: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
    logger: str
    level: str
    message: str
    error: InspectLogError | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    run_status: str | None = None
    trace_status: str | None = None


class InspectLogListResponse(BaseModel):
    items: list[InspectLogRecord]
    next_cursor: str | None = None


class AgentEventEnvelope(BaseModel):
    id: str
    ts: float
    kind: str = "agent_event"
    summary: str
    severity: str
    status: str
    producer: InspectProducer
    scope: InspectScope
    tags: list[str] = Field(default_factory=list)
    links: InspectLinks | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    event_id: str
    event_type: str
    payload_schema: InspectPayloadSchema | None = None


class AgentEventTypeRecord(BaseModel):
    event_type: str
    category: str
    display_label: str
    payload_schema_name: str | None = None
    payload_schema_version: int | None = None
    renderer_hint: str | None = None
    redaction_policy: str | None = None


class AgentEventListResponse(BaseModel):
    items: list[AgentEventEnvelope]
    next_cursor: str | None = None


class AgentEventTypeListResponse(BaseModel):
    items: list[AgentEventTypeRecord]

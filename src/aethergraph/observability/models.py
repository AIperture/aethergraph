from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
import uuid

from aethergraph.services.llm.provider_transport.models import ProviderTransportAttempt

CaptureMode = Literal["off", "metadata", "manifest", "full"]
ObservationStatus = Literal["ok", "error", "pending", "unknown"]
_DEPRECATED_APP_FIELD_METADATA = {
    "deprecated": True,
    "compatibility_only": True,
    "description": (
        "Deprecated optional App compatibility metadata; not authorization or canonical scope."
    ),
}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ObservationScope:
    tenant_id: str | None = None
    project_id: str | None = None
    org_id: str | None = None
    user_id: str | None = None
    app_id: str | None = field(default=None, metadata=_DEPRECATED_APP_FIELD_METADATA)
    session_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    agent_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    turn_id: str | None = None

    @classmethod
    def from_dimensions(cls, dimensions: dict[str, Any]) -> ObservationScope:
        return cls(**{name: dimensions.get(name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ObservationRecord:
    category: str
    name: str
    summary: str
    scope: ObservationScope = ObservationScope()
    observation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    occurred_at: str = field(default_factory=utc_now_iso)
    status: ObservationStatus = "ok"
    severity: str = "info"
    attributes: dict[str, Any] = field(default_factory=dict)
    parent_observation_id: str | None = None
    caused_by_observation_id: str | None = None
    source_event_id: str | None = None
    llm_call_id: str | None = None
    payload_fragment_id: str | None = None
    retention_class: str = "standard"
    expires_at: str | None = None


@dataclass
class LLMObservationRecord:
    llm_call_id: str
    created_at: str
    call_type: str
    provider: str
    model: str
    scope: ObservationScope
    profile_name: str | None = None
    call_name: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    reasoning_effort: str | None = None
    max_output_tokens: int | None = None
    output_format: str | None = None
    json_schema: dict[str, Any] | None = None
    schema_name: str | None = None
    strict_schema: bool | None = None
    validate_json: bool | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    request_args: dict[str, Any] = field(default_factory=dict)
    provider_request_args: dict[str, Any] = field(default_factory=dict)
    compatibility_notes: list[str] = field(default_factory=list)
    trace_payload: dict[str, Any] | None = None
    raw_text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    prompt_manifest_id: str | None = None
    attempts: tuple[ProviderTransportAttempt, ...] = ()
    tool_surface: dict[str, Any] | None = None
    request_items: list[dict[str, Any]] = field(default_factory=list)
    response_items: list[dict[str, Any]] = field(default_factory=list)
    tool_definitions: list[dict[str, Any]] = field(default_factory=list)
    lifecycle_status: str = "in_progress"

    @classmethod
    def new(
        cls,
        *,
        call_type: str,
        provider: str,
        model: str,
        dimensions: dict[str, Any],
        messages: list[dict[str, Any]],
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: str,
        json_schema: dict[str, Any] | None,
        schema_name: str | None,
        strict_schema: bool | None,
        validate_json: bool | None,
        extra_params: dict[str, Any],
        request_args: dict[str, Any] | None,
        provider_request_args: dict[str, Any] | None,
        compatibility_notes: list[str] | None,
        trace_payload: dict[str, Any] | None,
        profile_name: str | None = None,
        call_name: str | None = None,
        tool_surface: dict[str, Any] | None = None,
        request_items: list[dict[str, Any]] | None = None,
        tool_definitions: list[dict[str, Any]] | None = None,
    ) -> LLMObservationRecord:
        return cls(
            llm_call_id=str(uuid.uuid4()),
            created_at=utc_now_iso(),
            call_type=call_type,
            provider=provider,
            model=model,
            scope=ObservationScope.from_dimensions(dimensions),
            profile_name=profile_name,
            call_name=call_name,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            extra_params=extra_params,
            request_args=request_args or {},
            provider_request_args=provider_request_args or {},
            compatibility_notes=compatibility_notes or [],
            trace_payload=trace_payload,
            tool_surface=tool_surface,
            request_items=request_items or [],
            tool_definitions=tool_definitions or [],
        )


@dataclass(frozen=True)
class ObservationFilter:
    tenant_id: str | None = None
    project_id: str | None = None
    org_id: str | None = None
    user_id: str | None = None
    app_id: str | None = field(default=None, metadata=_DEPRECATED_APP_FIELD_METADATA)
    session_id: str | None = None
    run_id: str | None = None
    trace_id: str | None = None
    agent_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    category: str | None = None
    capture_mode: CaptureMode | None = None
    retention_class: str | None = None
    exclude_severity: str | None = None
    created_before: str | None = None
    expired_before: str | None = None
    pinned: bool | None = None
    target_reclaimed_bytes: int | None = None
    limit: int | None = None


@dataclass(frozen=True)
class StorageStats:
    observations: int
    llm_calls: int
    manifests: int
    fragments: int
    fragment_bytes: int
    logical_bytes: int
    database_bytes: int
    wal_bytes: int
    shm_bytes: int
    physical_bytes: int


@dataclass(frozen=True)
class PurgeResult:
    dry_run: bool
    matching_traces: int
    matching_observations: int
    matching_manifests: int
    exclusive_fragment_bytes: int
    shared_fragment_bytes_retained: int
    estimated_reclaimed_bytes: int
    deleted_observations: int = 0
    deleted_manifests: int = 0
    deleted_fragments: int = 0

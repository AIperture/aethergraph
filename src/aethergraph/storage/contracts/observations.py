"""Canonical observation, LLM-call, and retention repository contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Protocol, cast

from .pagination import Page, PageRequest
from .records import (
    FrozenJson,
    _freeze_json,
    _freeze_mapping,
    _nonempty,
    _optional_nonempty,
    _utc,
)
from .scope import StorageScope


class ObservationStatus(StrEnum):
    """Canonical lifecycle status for an observation."""

    OK = "ok"
    ERROR = "error"
    PENDING = "pending"
    UNKNOWN = "unknown"


class ObservationSeverity(StrEnum):
    """Canonical severity used for indexed observation filtering."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ObservationCaptureMode(StrEnum):
    """Canonical retained-content level for one LLM call."""

    OFF = "off"
    METADATA = "metadata"
    MANIFEST = "manifest"
    FULL = "full"


class LLMCallLifecycleStatus(StrEnum):
    """Canonical persistence lifecycle for one real provider call."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObservationUsageDimension(StrEnum):
    """Canonical logical scope dimension used for bounded usage accounting."""

    TRACE = "trace"
    RUN = "run"


class ObservationResourceRelation(StrEnum):
    """Canonical relationship from an observation to an external resource."""

    INPUT = "input"
    OUTPUT = "output"
    READ = "read"
    CREATED = "created"
    UPDATED = "updated"
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    INVALIDATES = "invalidates"
    MENTIONS = "mentions"


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationResourceLink:
    """Immutable indexed link from an observation to a canonical resource."""

    resource_key: str
    relation: ObservationResourceRelation
    resource_revision: str | None = None
    content_hash: str | None = None
    slot_key: str | None = None

    def __post_init__(self) -> None:
        _nonempty("resource_key", self.resource_key)
        if not isinstance(self.relation, ObservationResourceRelation):
            raise TypeError("relation must be an ObservationResourceRelation")
        for name in ("resource_revision", "content_hash", "slot_key"):
            _optional_nonempty(name, getattr(self, name))


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationDraft:
    """Canonical observation content before an ordered cursor is assigned."""

    observation_id: str
    category: str
    name: str
    summary: str
    occurred_at: datetime
    scope: StorageScope
    status: ObservationStatus = ObservationStatus.OK
    severity: ObservationSeverity = ObservationSeverity.INFO
    producer: str | None = None
    trace_id: str | None = None
    turn_id: str | None = None
    parent_observation_id: str | None = None
    caused_by_observation_id: str | None = None
    source_event_id: str | None = None
    attributes: Mapping[str, FrozenJson] = field(default_factory=dict)
    resource_links: tuple[ObservationResourceLink, ...] = ()
    payload_fragment_id: str | None = None
    retention_class: str = "standard"
    expires_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_observation(
            observation_id=self.observation_id,
            category=self.category,
            name=self.name,
            summary=self.summary,
            occurred_at=self.occurred_at,
            status=self.status,
            severity=self.severity,
            producer=self.producer,
            trace_id=self.trace_id,
            turn_id=self.turn_id,
            parent_observation_id=self.parent_observation_id,
            caused_by_observation_id=self.caused_by_observation_id,
            source_event_id=self.source_event_id,
            resource_links=self.resource_links,
            payload_fragment_id=self.payload_fragment_id,
            retention_class=self.retention_class,
            expires_at=self.expires_at,
            schema_version=self.schema_version,
        )
        object.__setattr__(
            self,
            "attributes",
            _freeze_mapping(self.attributes, path="attributes"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationRecord:
    """Committed canonical observation with provider-assigned opaque cursor."""

    observation_id: str
    category: str
    name: str
    summary: str
    occurred_at: datetime
    scope: StorageScope
    cursor: str
    status: ObservationStatus = ObservationStatus.OK
    severity: ObservationSeverity = ObservationSeverity.INFO
    producer: str | None = None
    trace_id: str | None = None
    turn_id: str | None = None
    parent_observation_id: str | None = None
    caused_by_observation_id: str | None = None
    source_event_id: str | None = None
    attributes: Mapping[str, FrozenJson] = field(default_factory=dict)
    resource_links: tuple[ObservationResourceLink, ...] = ()
    payload_fragment_id: str | None = None
    retention_class: str = "standard"
    expires_at: datetime | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_observation(
            observation_id=self.observation_id,
            category=self.category,
            name=self.name,
            summary=self.summary,
            occurred_at=self.occurred_at,
            status=self.status,
            severity=self.severity,
            producer=self.producer,
            trace_id=self.trace_id,
            turn_id=self.turn_id,
            parent_observation_id=self.parent_observation_id,
            caused_by_observation_id=self.caused_by_observation_id,
            source_event_id=self.source_event_id,
            resource_links=self.resource_links,
            payload_fragment_id=self.payload_fragment_id,
            retention_class=self.retention_class,
            expires_at=self.expires_at,
            schema_version=self.schema_version,
        )
        _nonempty("cursor", self.cursor)
        object.__setattr__(
            self,
            "attributes",
            _freeze_mapping(self.attributes, path="attributes"),
        )


def _validate_observation(
    *,
    observation_id: str,
    category: str,
    name: str,
    summary: str,
    occurred_at: datetime,
    status: ObservationStatus,
    severity: ObservationSeverity,
    producer: str | None,
    trace_id: str | None,
    turn_id: str | None,
    parent_observation_id: str | None,
    caused_by_observation_id: str | None,
    source_event_id: str | None,
    resource_links: tuple[ObservationResourceLink, ...],
    payload_fragment_id: str | None,
    retention_class: str,
    expires_at: datetime | None,
    schema_version: int,
) -> None:
    for field_name, value in (
        ("observation_id", observation_id),
        ("category", category),
        ("name", name),
        ("summary", summary),
        ("retention_class", retention_class),
    ):
        _nonempty(field_name, value)
    _utc("occurred_at", occurred_at)
    if not isinstance(status, ObservationStatus):
        raise TypeError("status must be an ObservationStatus")
    if not isinstance(severity, ObservationSeverity):
        raise TypeError("severity must be an ObservationSeverity")
    _optional_nonempty("producer", producer)
    for field_name, value in (
        ("trace_id", trace_id),
        ("turn_id", turn_id),
        ("parent_observation_id", parent_observation_id),
        ("caused_by_observation_id", caused_by_observation_id),
        ("source_event_id", source_event_id),
        ("payload_fragment_id", payload_fragment_id),
    ):
        _optional_nonempty(field_name, value)
    if not isinstance(resource_links, tuple):
        raise TypeError("resource_links must be an immutable tuple")
    if any(not isinstance(link, ObservationResourceLink) for link in resource_links):
        raise TypeError("resource_links must contain ObservationResourceLink values")
    identities = tuple((link.resource_key, link.relation) for link in resource_links)
    if len(set(identities)) != len(identities):
        raise ValueError("resource_links must not contain duplicate identities")
    if expires_at is not None:
        _utc("expires_at", expires_at)
        if expires_at <= occurred_at:
            raise ValueError("expires_at must be after occurred_at")
    if isinstance(schema_version, bool) or schema_version < 1:
        raise ValueError("schema_version must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationQuery:
    """Bounded indexed observation query with stable opaque pagination."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    categories: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    producers: tuple[str, ...] = ()
    statuses: tuple[ObservationStatus, ...] = ()
    severities: tuple[ObservationSeverity, ...] = ()
    trace_id: str | None = None
    turn_id: str | None = None
    resource_key: str | None = None
    resource_relation: ObservationResourceRelation | None = None
    occurred_at_or_after: datetime | None = None
    occurred_at_or_before: datetime | None = None

    def __post_init__(self) -> None:
        for name, values in (
            ("categories", self.categories),
            ("names", self.names),
            ("producers", self.producers),
            ("statuses", self.statuses),
            ("severities", self.severities),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        for name, values in (
            ("categories", self.categories),
            ("names", self.names),
            ("producers", self.producers),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if any(not isinstance(value, ObservationStatus) for value in self.statuses):
            raise TypeError("statuses must contain ObservationStatus values")
        if any(not isinstance(value, ObservationSeverity) for value in self.severities):
            raise TypeError("severities must contain ObservationSeverity values")
        for name in ("trace_id", "turn_id", "resource_key"):
            _optional_nonempty(name, getattr(self, name))
        if self.resource_relation is not None and self.resource_key is None:
            raise ValueError("resource_relation requires resource_key")
        if self.resource_relation is not None and not isinstance(
            self.resource_relation, ObservationResourceRelation
        ):
            raise TypeError("resource_relation must be an ObservationResourceRelation")
        for name in ("occurred_at_or_after", "occurred_at_or_before"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if (
            self.occurred_at_or_after is not None
            and self.occurred_at_or_before is not None
            and self.occurred_at_or_after > self.occurred_at_or_before
        ):
            raise ValueError("observation time bounds are reversed")


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCallAttempt:
    """Immutable provider-transport attempt attached to one LLM call."""

    attempt_number: int
    elapsed_ms: int
    outcome: str
    retryable: bool
    status_code: int | None = None
    error_code: str | None = None
    request_id: str | None = None
    provider_delay_ms: int | None = None
    scheduled_delay_ms: int | None = None
    rate_limits: tuple[Mapping[str, FrozenJson], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.attempt_number, bool) or self.attempt_number < 1:
            raise ValueError("attempt_number must be a positive integer")
        if isinstance(self.elapsed_ms, bool) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be a non-negative integer")
        _nonempty("outcome", self.outcome)
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        if self.status_code is not None and (
            isinstance(self.status_code, bool) or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be an HTTP status when supplied")
        for name in ("error_code", "request_id"):
            _optional_nonempty(name, getattr(self, name))
        for name in ("provider_delay_ms", "scheduled_delay_ms"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be non-negative when supplied")
        if not isinstance(self.rate_limits, tuple):
            raise TypeError("rate_limits must be an immutable tuple")
        object.__setattr__(
            self,
            "rate_limits",
            tuple(
                _freeze_mapping(value, path=f"rate_limits[{index}]")
                for index, value in enumerate(self.rate_limits)
            ),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCallDraft:
    """Canonical prepared LLM observation and optional captured content."""

    llm_call_id: str
    observation: ObservationDraft
    call_type: str
    provider: str
    model: str
    capture_mode: ObservationCaptureMode
    lifecycle_status: LLMCallLifecycleStatus = LLMCallLifecycleStatus.COMPLETED
    profile_name: str | None = None
    call_name: str | None = None
    request_options: Mapping[str, FrozenJson] = field(default_factory=dict)
    usage: Mapping[str, FrozenJson] = field(default_factory=dict)
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    prompt_manifest_id: str | None = None
    request_preview: FrozenJson = None
    response_preview: FrozenJson = None
    captured_request: FrozenJson = None
    captured_response: FrozenJson = None
    trace_payload: FrozenJson = None
    attempts: tuple[LLMCallAttempt, ...] = ()
    tool_surface: FrozenJson = None
    request_items: FrozenJson = None
    response_items: FrozenJson = None
    provider_request_facts: FrozenJson = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_llm_call(
            llm_call_id=self.llm_call_id,
            observation=self.observation,
            call_type=self.call_type,
            provider=self.provider,
            model=self.model,
            capture_mode=self.capture_mode,
            lifecycle_status=self.lifecycle_status,
            profile_name=self.profile_name,
            call_name=self.call_name,
            latency_ms=self.latency_ms,
            error_type=self.error_type,
            error_message=self.error_message,
            prompt_manifest_id=self.prompt_manifest_id,
            attempts=self.attempts,
            schema_version=self.schema_version,
        )
        if self.capture_mode in {
            ObservationCaptureMode.OFF,
            ObservationCaptureMode.METADATA,
        } and any(
            value is not None
            for value in (self.captured_request, self.captured_response, self.trace_payload)
        ):
            raise ValueError("off and metadata capture must not retain content")
        _freeze_llm_fields(self)


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCallRecord:
    """Committed LLM-call metadata with captured content excluded."""

    llm_call_id: str
    observation: ObservationRecord
    call_type: str
    provider: str
    model: str
    capture_mode: ObservationCaptureMode
    lifecycle_status: LLMCallLifecycleStatus = LLMCallLifecycleStatus.COMPLETED
    profile_name: str | None = None
    call_name: str | None = None
    request_options: Mapping[str, FrozenJson] = field(default_factory=dict)
    usage: Mapping[str, FrozenJson] = field(default_factory=dict)
    latency_ms: int | None = None
    error_type: str | None = None
    error_message: str | None = None
    prompt_manifest_id: str | None = None
    request_preview: FrozenJson = None
    response_preview: FrozenJson = None
    trace_payload_preview: FrozenJson = None
    attempts: tuple[LLMCallAttempt, ...] = ()
    tool_surface: FrozenJson = None
    request_items: FrozenJson = None
    response_items: FrozenJson = None
    provider_request_facts: FrozenJson = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_llm_call(
            llm_call_id=self.llm_call_id,
            observation=self.observation,
            call_type=self.call_type,
            provider=self.provider,
            model=self.model,
            capture_mode=self.capture_mode,
            lifecycle_status=self.lifecycle_status,
            profile_name=self.profile_name,
            call_name=self.call_name,
            latency_ms=self.latency_ms,
            error_type=self.error_type,
            error_message=self.error_message,
            prompt_manifest_id=self.prompt_manifest_id,
            attempts=self.attempts,
            schema_version=self.schema_version,
        )
        _freeze_llm_fields(self)


def _validate_llm_call(
    *,
    llm_call_id: str,
    observation: ObservationDraft | ObservationRecord,
    call_type: str,
    provider: str,
    model: str,
    capture_mode: ObservationCaptureMode,
    lifecycle_status: LLMCallLifecycleStatus,
    profile_name: str | None,
    call_name: str | None,
    latency_ms: int | None,
    error_type: str | None,
    error_message: str | None,
    prompt_manifest_id: str | None,
    attempts: tuple[LLMCallAttempt, ...],
    schema_version: int,
) -> None:
    for name, value in (
        ("llm_call_id", llm_call_id),
        ("call_type", call_type),
        ("provider", provider),
        ("model", model),
    ):
        _nonempty(name, value)
    if observation.category != "llm":
        raise ValueError("LLM calls require an observation in category 'llm'")
    if not isinstance(capture_mode, ObservationCaptureMode):
        raise TypeError("capture_mode must be an ObservationCaptureMode")
    if not isinstance(lifecycle_status, LLMCallLifecycleStatus):
        raise TypeError("lifecycle_status must be an LLMCallLifecycleStatus")
    for name, value in (
        ("profile_name", profile_name),
        ("call_name", call_name),
        ("error_type", error_type),
        ("error_message", error_message),
        ("prompt_manifest_id", prompt_manifest_id),
    ):
        _optional_nonempty(name, value)
    if (error_type is None) != (error_message is None):
        raise ValueError("error_type and error_message must be supplied together")
    if capture_mode is ObservationCaptureMode.OFF and prompt_manifest_id is not None:
        raise ValueError("off capture must not have prompt_manifest_id")
    if capture_mode is not ObservationCaptureMode.OFF and prompt_manifest_id is None:
        raise ValueError("non-off capture requires prompt_manifest_id")
    if error_type is None and observation.status is ObservationStatus.ERROR:
        raise ValueError("error observations require LLM error details")
    if error_type is not None and observation.status is not ObservationStatus.ERROR:
        raise ValueError("LLM error details require error observation status")
    expected_status = {
        LLMCallLifecycleStatus.IN_PROGRESS: ObservationStatus.PENDING,
        LLMCallLifecycleStatus.COMPLETED: ObservationStatus.OK,
        LLMCallLifecycleStatus.FAILED: ObservationStatus.ERROR,
        LLMCallLifecycleStatus.CANCELLED: ObservationStatus.ERROR,
    }[lifecycle_status]
    if observation.status is not expected_status:
        raise ValueError("LLM lifecycle status does not match observation status")
    if latency_ms is not None and (isinstance(latency_ms, bool) or latency_ms < 0):
        raise ValueError("latency_ms must be non-negative when supplied")
    if not isinstance(attempts, tuple):
        raise TypeError("attempts must be an immutable tuple")
    if any(not isinstance(attempt, LLMCallAttempt) for attempt in attempts):
        raise TypeError("attempts must contain LLMCallAttempt values")
    numbers = tuple(attempt.attempt_number for attempt in attempts)
    if numbers != tuple(range(1, len(attempts) + 1)):
        raise ValueError("attempt numbers must be contiguous starting at one")
    if isinstance(schema_version, bool) or schema_version < 1:
        raise ValueError("schema_version must be a positive integer")


def _freeze_llm_fields(value: LLMCallDraft | LLMCallRecord) -> None:
    object.__setattr__(
        value,
        "request_options",
        _freeze_mapping(value.request_options, path="request_options"),
    )
    object.__setattr__(value, "usage", _freeze_mapping(value.usage, path="usage"))
    for name in ("request_preview", "response_preview"):
        object.__setattr__(value, name, _freeze_json(getattr(value, name), path=name))
    for name in (
        "tool_surface",
        "request_items",
        "response_items",
        "provider_request_facts",
    ):
        object.__setattr__(value, name, _freeze_json(getattr(value, name), path=name))
    if isinstance(value, LLMCallDraft):
        for name in ("captured_request", "captured_response", "trace_payload"):
            object.__setattr__(value, name, _freeze_json(getattr(value, name), path=name))
    else:
        object.__setattr__(
            value,
            "trace_payload_preview",
            _freeze_json(value.trace_payload_preview, path="trace_payload_preview"),
        )


@dataclass(frozen=True, slots=True)
class LLMCallDetail:
    """Exact LLM call plus retained request, response, and trace content."""

    record: LLMCallRecord
    captured_request: FrozenJson = None
    captured_response: FrozenJson = None
    trace_payload: FrozenJson = None

    def __post_init__(self) -> None:
        for name in ("captured_request", "captured_response", "trace_payload"):
            object.__setattr__(self, name, _freeze_json(getattr(self, name), path=name))
        if self.record.capture_mode not in {
            ObservationCaptureMode.MANIFEST,
            ObservationCaptureMode.FULL,
        } and any(
            value is not None
            for value in (self.captured_request, self.captured_response, self.trace_payload)
        ):
            raise ValueError("only manifest or full capture may expose retained content")


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMCallQuery:
    """Bounded indexed LLM-call query returning metadata-only records."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    trace_id: str | None = None
    providers: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    call_types: tuple[str, ...] = ()
    prompt_manifest_ids: tuple[str, ...] = ()
    statuses: tuple[ObservationStatus, ...] = ()
    occurred_at_or_after: datetime | None = None
    occurred_at_or_before: datetime | None = None

    def __post_init__(self) -> None:
        _optional_nonempty("trace_id", self.trace_id)
        for name, values in (
            ("providers", self.providers),
            ("models", self.models),
            ("call_types", self.call_types),
            ("prompt_manifest_ids", self.prompt_manifest_ids),
            ("statuses", self.statuses),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        for name, values in (
            ("providers", self.providers),
            ("models", self.models),
            ("call_types", self.call_types),
            ("prompt_manifest_ids", self.prompt_manifest_ids),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if any(not isinstance(value, ObservationStatus) for value in self.statuses):
            raise TypeError("statuses must contain ObservationStatus values")
        for name in ("occurred_at_or_after", "occurred_at_or_before"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if (
            self.occurred_at_or_after is not None
            and self.occurred_at_or_before is not None
            and self.occurred_at_or_after > self.occurred_at_or_before
        ):
            raise ValueError("LLM call time bounds are reversed")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationTraceSummaryQuery:
    """Bound one provider-side trace aggregation to exact canonical scope and time."""

    scope: StorageScope
    occurred_at_or_after: datetime | None = None
    occurred_at_or_before: datetime | None = None
    trace_id_limit: int = 100
    failing_service_limit: int = 5

    def __post_init__(self) -> None:
        _summary_bounds(self.occurred_at_or_after, self.occurred_at_or_before)
        _summary_limit("trace_id_limit", self.trace_id_limit)
        _summary_limit("failing_service_limit", self.failing_service_limit)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationTraceSummaryRecord:
    """Bounded aggregate-only trace statistics for one exact canonical query."""

    span_count: int
    error_count: int
    total_duration_ms: int
    trace_id_count: int
    trace_ids: tuple[str, ...] = ()
    trace_ids_truncated: bool = False
    top_failing_services: Mapping[str, int] = field(default_factory=dict)
    latest_error_at: datetime | None = None

    def __post_init__(self) -> None:
        for name in ("span_count", "error_count", "total_duration_ms", "trace_id_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.error_count > self.span_count:
            raise ValueError("error_count must not exceed span_count")
        if not isinstance(self.trace_ids, tuple):
            raise TypeError("trace_ids must be an immutable tuple")
        if len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError("trace_ids must not contain duplicates")
        if any(not isinstance(value, str) or not value.strip() for value in self.trace_ids):
            raise ValueError("trace_ids must contain non-empty strings")
        if self.trace_id_count < len(self.trace_ids):
            raise ValueError("trace_id_count must cover returned trace_ids")
        if self.trace_ids_truncated is not (self.trace_id_count > len(self.trace_ids)):
            raise ValueError("trace_ids_truncated must match trace_id_count")
        if self.latest_error_at is not None:
            _utc("latest_error_at", self.latest_error_at)
        if (self.error_count == 0) is not (self.latest_error_at is None):
            raise ValueError("latest_error_at must match error_count")
        failures = _positive_counts(
            "top_failing_services",
            self.top_failing_services,
        )
        if sum(failures.values()) > self.error_count:
            raise ValueError("top_failing_services must not exceed error_count")
        object.__setattr__(self, "top_failing_services", failures)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationLLMSummaryQuery:
    """Bound one provider-side LLM aggregation to exact canonical scope and time."""

    scope: StorageScope
    occurred_at_or_after: datetime | None = None
    occurred_at_or_before: datetime | None = None
    model_limit: int = 100

    def __post_init__(self) -> None:
        _summary_bounds(self.occurred_at_or_after, self.occurred_at_or_before)
        _summary_limit("model_limit", self.model_limit)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationLLMSummaryRecord:
    """Bounded aggregate-only LLM statistics for one exact canonical query."""

    total_calls: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    error_count: int
    model_count: int
    by_model: Mapping[str, int] = field(default_factory=dict)
    by_model_truncated: bool = False

    def __post_init__(self) -> None:
        for name in (
            "total_calls",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "error_count",
            "model_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.error_count > self.total_calls:
            raise ValueError("error_count must not exceed total_calls")
        models = _positive_counts("by_model", self.by_model)
        if self.model_count < len(models):
            raise ValueError("model_count must cover returned by_model entries")
        if self.by_model_truncated is not (self.model_count > len(models)):
            raise ValueError("by_model_truncated must match model_count")
        if sum(models.values()) > self.total_calls:
            raise ValueError("by_model counts must not exceed total_calls")
        object.__setattr__(self, "by_model", models)


def _summary_bounds(after: datetime | None, before: datetime | None) -> None:
    if after is not None:
        _utc("occurred_at_or_after", after)
    if before is not None:
        _utc("occurred_at_or_before", before)
    if after is not None and before is not None and after > before:
        raise ValueError("summary time bounds are reversed")


def _summary_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 500:
        raise ValueError(f"{name} must be between 1 and 500")


def _positive_counts(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    normalized: dict[str, int] = {}
    for key, value in values.items():
        _nonempty(f"{name} key", key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} values must be positive integers")
        normalized[key] = value
    return cast(Mapping[str, int], _freeze_mapping(normalized, path=name))


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationPurgeRequest:
    """Bounded retention deletion request with explicit preview behavior."""

    scope: StorageScope
    dry_run: bool = True
    categories: tuple[str, ...] = ()
    capture_modes: tuple[ObservationCaptureMode, ...] = ()
    retention_classes: tuple[str, ...] = ()
    severities: tuple[ObservationSeverity, ...] = ()
    excluded_severities: tuple[ObservationSeverity, ...] = ()
    trace_id: str | None = None
    occurred_before: datetime | None = None
    expired_before: datetime | None = None
    max_observations: int = 1000
    target_reclaimed_bytes: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.dry_run, bool):
            raise TypeError("dry_run must be a boolean")
        for name, values in (
            ("categories", self.categories),
            ("capture_modes", self.capture_modes),
            ("retention_classes", self.retention_classes),
            ("severities", self.severities),
            ("excluded_severities", self.excluded_severities),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        for name, values in (
            ("categories", self.categories),
            ("retention_classes", self.retention_classes),
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
        if any(not isinstance(value, ObservationCaptureMode) for value in self.capture_modes):
            raise TypeError("capture_modes must contain ObservationCaptureMode values")
        for name, values in (
            ("severities", self.severities),
            ("excluded_severities", self.excluded_severities),
        ):
            if any(not isinstance(value, ObservationSeverity) for value in values):
                raise TypeError(f"{name} must contain ObservationSeverity values")
        if set(self.severities) & set(self.excluded_severities):
            raise ValueError("included and excluded severities must not overlap")
        _optional_nonempty("trace_id", self.trace_id)
        for name in ("occurred_before", "expired_before"):
            value = getattr(self, name)
            if value is not None:
                _utc(name, value)
        if isinstance(self.max_observations, bool) or not 1 <= self.max_observations <= 10_000:
            raise ValueError("max_observations must be between 1 and 10000")
        if self.target_reclaimed_bytes is not None and (
            isinstance(self.target_reclaimed_bytes, bool) or self.target_reclaimed_bytes < 1
        ):
            raise ValueError("target_reclaimed_bytes must be positive when supplied")


@dataclass(frozen=True, slots=True)
class ObservationPurgeResult:
    """Provider-neutral preview or result of bounded observation retention."""

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

    def __post_init__(self) -> None:
        if not isinstance(self.dry_run, bool):
            raise TypeError("dry_run must be a boolean")
        for name in (
            "matching_traces",
            "matching_observations",
            "matching_manifests",
            "exclusive_fragment_bytes",
            "shared_fragment_bytes_retained",
            "estimated_reclaimed_bytes",
            "deleted_observations",
            "deleted_manifests",
            "deleted_fragments",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.dry_run and any(
            value > 0
            for value in (
                self.deleted_observations,
                self.deleted_manifests,
                self.deleted_fragments,
            )
        ):
            raise ValueError("dry-run purge results must not report deletions")


@dataclass(frozen=True, slots=True)
class ObservationStorageStats:
    """Provider-neutral logical observation storage accounting."""

    observations: int
    llm_calls: int
    manifests: int
    fragments: int
    fragment_bytes: int
    logical_bytes: int
    provider_metrics: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "observations",
            "llm_calls",
            "manifests",
            "fragments",
            "fragment_bytes",
            "logical_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        frozen: dict[str, int] = {}
        for name, value in self.provider_metrics.items():
            _nonempty("provider_metrics key", name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("provider_metrics values must be non-negative integers")
            frozen[name] = value
        object.__setattr__(
            self, "provider_metrics", _freeze_mapping(frozen, path="provider_metrics")
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationScopeUsageRecord:
    """Logical observation usage for one trace or run scope."""

    dimension: ObservationUsageDimension
    scope_id: str
    latest_at: datetime
    observation_count: int
    logical_bytes: int
    pinned: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ObservationUsageDimension):
            raise TypeError("dimension must be an ObservationUsageDimension")
        _nonempty("scope_id", self.scope_id)
        _utc("latest_at", self.latest_at)
        for name in ("observation_count", "logical_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.pinned, bool):
            raise TypeError("pinned must be a boolean")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationScopeUsageQuery:
    """Bounded logical usage query for one trace or run dimension."""

    scope: StorageScope
    dimension: ObservationUsageDimension
    page: PageRequest = PageRequest()

    def __post_init__(self) -> None:
        if not isinstance(self.dimension, ObservationUsageDimension):
            raise TypeError("dimension must be an ObservationUsageDimension")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationScopeManagementRecord:
    """Revisioned retention and visibility policy for one logical scope key."""

    scope_key: str
    scope: StorageScope
    revision: int
    updated_at: datetime
    trace_id: str | None = None
    pinned: bool = False
    hidden: bool = False
    deleted: bool = False
    label: str | None = None
    tags: tuple[str, ...] = ()
    retention_class: str = "standard"
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _nonempty("scope_key", self.scope_key)
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("updated_at", self.updated_at)
        _optional_nonempty("trace_id", self.trace_id)
        _optional_nonempty("label", self.label)
        _nonempty("retention_class", self.retention_class)
        for name in ("pinned", "hidden", "deleted"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        if not isinstance(self.tags, tuple):
            raise TypeError("tags must be an immutable tuple")
        if any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("tags must contain non-empty strings")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags must not contain duplicates")
        if self.expires_at is not None:
            _utc("expires_at", self.expires_at)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationScopeManagementQuery:
    """Bounded indexed query for explicit observation scope-management records."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    trace_id: str | None = None
    pinned: bool | None = None
    hidden: bool | None = None
    deleted: bool | None = None
    retention_classes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _optional_nonempty("trace_id", self.trace_id)
        for name in ("pinned", "hidden", "deleted"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean when supplied")
        if not isinstance(self.retention_classes, tuple):
            raise TypeError("retention_classes must be an immutable tuple")
        if len(set(self.retention_classes)) != len(self.retention_classes):
            raise ValueError("retention_classes must not contain duplicates")
        if any(not isinstance(value, str) or not value.strip() for value in self.retention_classes):
            raise ValueError("retention_classes must contain non-empty strings")


class ObservationRepository(Protocol):
    """Ordered observation, LLM detail, and retention repository."""

    async def append_many(
        self,
        observations: tuple[ObservationDraft, ...],
    ) -> tuple[ObservationRecord, ...]:
        """Atomically append ordered observations and resource links.

        Provider cursors are assigned in input order. Repeating an observation ID with
        identical content is idempotent; conflicting reuse fails the entire batch.

        Examples:
            Append one log observation:
                ```python
                stored, = await observations.append_many((draft,))
                ```

            Append a span batch:
                ```python
                stored = await observations.append_many((started, finished))
                ```

        Args:
            observations: Non-empty immutable batch in required append order.

        Returns:
            tuple[ObservationRecord, ...]: Committed records in input order.

        Notes:
            Resource links commit with their observations. Partial append and
            per-observation fallback are forbidden.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        observation_id: str,
    ) -> ObservationRecord | None:
        """Read one canonical observation by exact scoped identity.

        The provider returns the observation and its indexed resource links without
        hydrating retained LLM prompt content.

        Examples:
            Read an observation:
                ```python
                observation = await observations.get(scope, "obs-1")
                ```

            Detect absence:
                ```python
                assert await observations.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner/execution scope constraining access.
            observation_id: Exact stable observation identity.

        Returns:
            ObservationRecord | None: Stored observation or `None` when absent.

        Notes:
            Deprecated App identity is neither authorization nor a lookup dimension.
        """
        ...

    async def query(self, query: ObservationQuery) -> Page[ObservationRecord]:
        """Query a bounded cursor page using promoted observation indexes.

        Canonical scope and category/status/severity/trace/resource/time filters apply
        before ordering and pagination inside the provider.

        Examples:
            List error logs:
                ```python
                page = await observations.query(ObservationQuery(scope=scope, categories=("log",)))
                ```

            Follow one resource:
                ```python
                page = await observations.query(ObservationQuery(scope=scope, resource_key=key))
                ```

        Args:
            query: Exact canonical scope, indexed filters, and opaque page request.

        Returns:
            Page[ObservationRecord]: Matching records and continuation cursor.

        Notes:
            Offset pagination, unbounded resource lists, and service-side broad scans
            are absent from this protocol.
        """
        ...

    async def begin_llm_call(self, call: LLMCallDraft) -> LLMCallRecord:
        """Atomically persist one prepared in-progress LLM call.

        AG applies capture/redaction policy before this call. The provider persists the
        prepared draft as one authority and returns a metadata-only list record.

        Examples:
            Store a metadata-only call:
                ```python
                record = await observations.begin_llm_call(call)
                ```

            Retry the same begin identity:
                ```python
                assert await observations.begin_llm_call(call) == record
                ```

        Args:
            call: Prepared canonical LLM call and optional policy-approved content.

        Returns:
            LLMCallRecord: Committed metadata-only LLM record.

        Notes:
            The draft lifecycle must be `in_progress`. Conflicting identity raises
            `StorageIntegrityError`; request capture commits transactionally.
        """
        ...

    async def finish_llm_call(
        self,
        llm_call_id: str,
        outcome: LLMCallDraft,
    ) -> LLMCallRecord:
        """Atomically finish one previously begun LLM call.

        The outcome carries the terminal response, usage, attempts, and exact
        `completed`, `failed`, or `cancelled` lifecycle. Missing begins and
        conflicting terminal retries raise `StorageIntegrityError`.

        Examples:
            Finish a completed call:
                ```python
                record = await observations.finish_llm_call("call-1", completed)
                ```

            Finish a cancelled call:
                ```python
                record = await observations.finish_llm_call("call-2", cancelled)
                ```

        Args:
            llm_call_id: Exact identity previously passed to `begin_llm_call`.
            outcome: Terminal canonical LLM call draft.

        Returns:
            LLMCallRecord: Authoritative terminal metadata-only record.

        Notes:
            Finish never inserts a missing call and terminal conflicts fail closed.
        """
        ...

    async def get_llm_call(
        self,
        scope: StorageScope,
        llm_call_id: str,
    ) -> LLMCallDetail | None:
        """Read one exact LLM call and its policy-retained content.

        Full captured content is available only from this exact scoped detail method;
        ordinary list records expose previews and metadata only.

        Examples:
            Hydrate an inspected call:
                ```python
                detail = await observations.get_llm_call(scope, "call-1")
                ```

            Detect absent content:
                ```python
                assert await observations.get_llm_call(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner/execution scope constraining access.
            llm_call_id: Exact stable LLM call identity.

        Returns:
            LLMCallDetail | None: Scoped call detail or `None` when absent.

        Notes:
            Providers never hydrate prompt bodies during query/list operations.
        """
        ...

    async def query_llm_calls(self, query: LLMCallQuery) -> Page[LLMCallRecord]:
        """Query bounded metadata-only LLM records using promoted indexes.

        Scope, trace, provider, model, call type, prompt-manifest identity, status,
        and time filters execute in the provider before stable cursor pagination.

        Examples:
            List failed calls:
                ```python
                page = await observations.query_llm_calls(LLMCallQuery(scope=scope, statuses=(status,)))
                ```

            Continue one provider page:
                ```python
                page = await observations.query_llm_calls(replace(query, page=next_page))
                ```

        Args:
            query: Exact canonical scope, LLM filters, and opaque page request.

        Returns:
            Page[LLMCallRecord]: Metadata-only records and continuation cursor.

        Notes:
            Attempts are loaded without per-row correlated queries; retained prompt
            and response content is excluded. Manifest identity is an indexed
            correlation filter, not a direct fragment-hydration API.
        """
        ...

    async def summarize_traces(
        self,
        query: ObservationTraceSummaryQuery,
    ) -> ObservationTraceSummaryRecord:
        """Aggregate trace observations inside the provider with bounded breakdowns.

        Intro:
            Exact scope and time predicates apply before aggregate counts, duration,
            error statistics, and deterministic capped breakdown selection.

        Examples:
            Summarize one run:
                ```python
                summary = await observations.summarize_traces(
                    ObservationTraceSummaryQuery(scope=run_scope)
                )
                ```

            Apply a time window and smaller identity cap:
                ```python
                summary = await observations.summarize_traces(
                    ObservationTraceSummaryQuery(
                        scope=run_scope,
                        occurred_at_or_after=start,
                        trace_id_limit=25,
                    )
                )
                ```

        Args:
            query: Exact canonical scope, time bounds, and breakdown caps.

        Returns:
            ObservationTraceSummaryRecord: Aggregate-only counts and bounded breakdowns.

        Notes:
            Captured content, provider locators, deprecated App identity, raw rows,
            and unbounded identity collections are absent from this operation.
        """
        ...

    async def summarize_llm_calls(
        self,
        query: ObservationLLMSummaryQuery,
    ) -> ObservationLLMSummaryRecord:
        """Aggregate LLM call metadata inside the provider with bounded models.

        Intro:
            Exact scope and time predicates apply before call, token, error, and
            deterministic capped model-count aggregation.

        Examples:
            Summarize one run:
                ```python
                summary = await observations.summarize_llm_calls(
                    ObservationLLMSummaryQuery(scope=run_scope)
                )
                ```

            Apply a time window and smaller model cap:
                ```python
                summary = await observations.summarize_llm_calls(
                    ObservationLLMSummaryQuery(
                        scope=run_scope,
                        occurred_at_or_before=end,
                        model_limit=25,
                    )
                )
                ```

        Args:
            query: Exact canonical scope, time bounds, and model cap.

        Returns:
            ObservationLLMSummaryRecord: Aggregate-only usage and bounded model counts.

        Notes:
            Prompt, response, trace content, provider locators, deprecated App
            identity, and unbounded model collections are absent from this operation.
        """
        ...

    async def purge(self, request: ObservationPurgeRequest) -> ObservationPurgeResult:
        """Preview or execute one bounded retention purge transaction.

        Selection, shared-fragment accounting, deletion, and orphan collection use the
        same provider authority and explicit maximum observation count.

        Examples:
            Preview old rows:
                ```python
                preview = await observations.purge(request)
                ```

            Execute an approved purge:
                ```python
                result = await observations.purge(replace(request, dry_run=False))
                ```

        Args:
            request: Exact scope, retention filters, safety bound, and dry-run mode.

        Returns:
            ObservationPurgeResult: Matching, retained, reclaimed, and deletion counts.

        Notes:
            Read-only bundles reject execution with `StorageReadOnlyError`; preview is
            still permitted.
        """
        ...

    async def storage_stats(self, scope: StorageScope) -> ObservationStorageStats:
        """Return logical observation accounting for one canonical scope.

        Provider-specific non-sensitive capacity counters may appear in the metrics
        mapping without exposing paths, handles, or schema details.

        Examples:
            Inspect workspace usage:
                ```python
                stats = await observations.storage_stats(workspace_scope)
                ```

            Read fragment usage:
                ```python
                print(stats.fragment_bytes)
                ```

        Args:
            scope: Canonical scope constraining logical accounting.

        Returns:
            ObservationStorageStats: Logical counts, bytes, and bounded provider metrics.

        Notes:
            Physical filenames and SQLite WAL/SHM concepts are not canonical fields.
        """
        ...

    async def query_scope_usage(
        self,
        query: ObservationScopeUsageQuery,
    ) -> Page[ObservationScopeUsageRecord]:
        """Query bounded logical observation usage by trace or run.

        The provider computes logical bytes and observation counts for the requested
        dimension before ordering by latest activity and applying cursor pagination.

        Examples:
            List the newest trace usage:
                ```python
                page = await observations.query_scope_usage(
                    ObservationScopeUsageQuery(
                        scope=scope,
                        dimension=ObservationUsageDimension.TRACE,
                    )
                )
                ```

            Continue a run-usage page:
                ```python
                page = await observations.query_scope_usage(
                    replace(query, page=PageRequest(limit=50, cursor=cursor))
                )
                ```

        Args:
            query: Canonical scope, trace-or-run dimension, and opaque page request.

        Returns:
            Page[ObservationScopeUsageRecord]: Logical usage records and continuation cursor.

        Notes:
            Results are logical provider accounting, not physical database allocation.
            Unbounded scope lists and service-side broad scans are forbidden.
        """
        ...

    async def query_scope_management(
        self,
        query: ObservationScopeManagementQuery,
    ) -> Page[ObservationScopeManagementRecord]:
        """Query a bounded page of explicit scope-management records.

        Visibility, deletion, pin, trace, and retention-class predicates execute in
        the provider before stable revision/update-time pagination.

        Examples:
            List suppressed scopes:
                ```python
                page = await observations.query_scope_management(
                    ObservationScopeManagementQuery(scope=scope, deleted=True)
                )
                ```

            List pinned forensic scopes:
                ```python
                page = await observations.query_scope_management(
                    ObservationScopeManagementQuery(
                        scope=scope,
                        pinned=True,
                        retention_classes=("forensic",),
                    )
                )
                ```

        Args:
            query: Canonical scope, promoted management filters, and page request.

        Returns:
            Page[ObservationScopeManagementRecord]: Explicit records and continuation cursor.

        Notes:
            Missing policy remains absent; providers do not synthesize inherited or
            default management records.
        """
        ...

    async def get_scope_management(
        self,
        scope: StorageScope,
        scope_key: str,
    ) -> ObservationScopeManagementRecord | None:
        """Read retention and visibility management for one logical scope key.

        The exact scoped lookup has no inheritance or alternate-key fallback.

        Examples:
            Read trace policy:
                ```python
                policy = await observations.get_scope_management(scope, "trace:trace-1")
                ```

            Detect default policy:
                ```python
                assert await observations.get_scope_management(scope, "trace:new") is None
                ```

        Args:
            scope: Canonical owner/execution scope constraining access.
            scope_key: Exact opaque logical management identity.

        Returns:
            ObservationScopeManagementRecord | None: Current record or `None`.

        Notes:
            Missing records mean AG policy defaults; providers do not synthesize rows.
        """
        ...

    async def compare_and_set_scope_management(
        self,
        record: ObservationScopeManagementRecord,
        expected_revision: int,
    ) -> ObservationScopeManagementRecord:
        """Atomically create or advance one scope-management record.

        Revision zero creates the first record. Pin, visibility, deletion marker,
        labels, tags, and retention fields commit together.

        Examples:
            Pin one trace:
                ```python
                stored = await observations.compare_and_set_scope_management(record, 0)
                ```

            Mark a scope deleted:
                ```python
                stored = await observations.compare_and_set_scope_management(updated, current.revision)
                ```

        Args:
            record: Complete canonical next management revision.
            expected_revision: Current revision required, or zero for creation.

        Returns:
            ObservationScopeManagementRecord: Newly committed authoritative revision.

        Notes:
            Stale expectations raise `StorageConflictError`; tombstones never delete
            authoritative run or session records.
        """
        ...

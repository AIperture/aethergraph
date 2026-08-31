"""Canonical observation service projection over provider storage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from aethergraph.server.security.redaction import canonical_json, sanitize_content
from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.services.llm.correlation import complete_llm_call_correlation
from aethergraph.storage.contracts import (
    LLMCallAttempt,
    LLMCallDraft,
    LLMCallLifecycleStatus,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationRepository,
    ObservationResourceLink,
    ObservationResourceRelation,
    ObservationSeverity,
    ObservationStatus,
    StorageBundle,
    StorageScope,
)

from .models import LLMObservationRecord, ObservationRecord, ObservationScope
from .policy import ObservationPolicy
from .prompt_store import PromptStore

_COMPATIBILITY_METADATA = "compatibility_metadata"
_DEPRECATED_APP_ID = "app_id"
_CANONICAL_SCOPE_NAMES = (
    "tenant_id",
    "project_id",
    "org_id",
    "user_id",
    "session_id",
    "run_id",
    "graph_id",
    "node_id",
    "agent_id",
)


class CanonicalObservationService(Protocol):
    """Provider-neutral observation service used by the active runtime."""

    repository: ObservationRepository
    owner_scope: StorageScope
    policy: ObservationPolicy

    async def append_observation(
        self,
        record: ObservationRecord,
        *,
        resource_links: Iterable[dict[str, Any]] = (),
    ) -> str:
        """Append one current AG observation through the canonical repository.

        The service maps runtime scope, producer, resources, and compatibility
        metadata once before the provider performs its atomic append.

        Examples:
            Append an operation observation:
                ```python
                observation_id = await service.append_observation(record)
                ```

            Append linked artifact evidence:
                ```python
                observation_id = await service.append_observation(
                    record,
                    resource_links=({"resource_key": "artifact:a1", "relation": "output"},),
                )
                ```

        Args:
            record: Current AG observation record to project.
            resource_links: Immutable or iterable canonical resource-link mappings.

        Returns:
            str: Stable appended observation identity.

        Notes:
            Deprecated App identity is stored only in a marked non-indexed
            compatibility envelope. No fallback or secondary write occurs.
        """
        ...

    async def begin_llm_call(self, record: LLMObservationRecord, *, capture_mode: str) -> None:
        """Persist one prepared request before quota or provider transport.

        The configured capture policy bounds request evidence before the canonical
        repository creates one in-progress identity.

        Examples:
            Begin manifest capture:
                ```python
                await service.begin_llm_call(record, capture_mode="manifest")
                ```

            Begin metadata capture:
                ```python
                await service.begin_llm_call(record, capture_mode="metadata")
                ```

        Args:
            record: Prepared provider-neutral LLM observation.
            capture_mode: Exact configured capture mode.

        Returns:
            None: The in-progress call is committed.

        Notes:
            No provider request is issued by this persistence method.
        """
        ...

    async def finish_llm_call(self, record: LLMObservationRecord, *, capture_mode: str) -> None:
        """Persist one terminal outcome for a previously begun provider call.

        Capture/redaction policy is applied before the provider receives one prepared
        atomic LLM draft and its observation metadata.

        Examples:
            Store a completed call:
                ```python
                await service.finish_llm_call(call, capture_mode="manifest")
                ```

            Store metadata-only evidence:
                ```python
                await metadata_service.finish_llm_call(call, capture_mode="metadata")
                ```

        Args:
            record: Current completed AG LLM observation.
            capture_mode: Exact configured capture mode for mismatch detection.

        Returns:
            None: The canonical call is committed and correlation is completed.

        Notes:
            A matching begin is mandatory; there is no insert-on-finish path.
        """
        ...


class ProviderObservationService:
    """Project current AG observation writes onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: ObservationRepository,
        owner_scope: StorageScope,
        policy: ObservationPolicy,
    ) -> None:
        """Bind one canonical observation repository without opening storage.

        Construction validates the trusted owner and capture policy but performs no
        provider selection, I/O, lifecycle operation, fallback, or schema work.

        Examples:
            Bind an open bundle repository:
                ```python
                service = ProviderObservationService(
                    repository=bundle.observations,
                    owner_scope=StorageScope(project_id="project-1"),
                    policy=ObservationPolicy(),
                )
                ```

            Bind a deterministic fake repository:
                ```python
                service = ProviderObservationService(
                    repository=fake_observations,
                    owner_scope=owner_scope,
                    policy=ObservationPolicy(capture_mode="metadata"),
                )
                ```

        Args:
            repository: Exact canonical observation repository.
            owner_scope: Trusted provider ownership scope without execution identity.
            policy: AG capture, redaction, preview, and retention policy.

        Returns:
            None: The provider-backed service is ready for use.

        Notes:
            The owning `StorageBundle` retains repository lifecycle responsibility.
        """
        validate_storage_owner_scope(owner_scope)
        policy.validate()
        self.repository = repository
        self.owner_scope = owner_scope
        self.policy = policy
        self._prompt_store = PromptStore(policy)

    async def append_observation(
        self,
        record: ObservationRecord,
        *,
        resource_links: Iterable[dict[str, Any]] = (),
    ) -> str:
        """Append one current AG observation through the canonical repository.

        Runtime identity and indexed producer fields are projected once, then one
        canonical atomic append supplies the only persistence authority.

        Examples:
            Append a log observation:
                ```python
                observation_id = await service.append_observation(record)
                ```

            Append resource-linked evidence:
                ```python
                observation_id = await service.append_observation(
                    record,
                    resource_links=links,
                )
                ```

        Args:
            record: Current AG observation record to project.
            resource_links: Canonical resource-link mappings in authored order.

        Returns:
            str: Stable committed observation identity.

        Notes:
            App compatibility metadata is never added to canonical scope or indexes.
        """
        links = tuple(_resource_link(link) for link in resource_links)
        draft = _observation_draft(record, owner_scope=self.owner_scope, links=links)
        (stored,) = await self.repository.append_many((draft,))
        return stored.observation_id

    async def begin_llm_call(self, record: LLMObservationRecord, *, capture_mode: str) -> None:
        """Persist one in-progress AG LLM call before execution starts.

        Capture policy is applied to the prepared request before the provider-owned
        repository creates the sole durable call identity.

        Examples:
            Begin a manifest call:
                ```python
                await service.begin_llm_call(record, capture_mode="manifest")
                ```

            Begin a metadata-only call:
                ```python
                await service.begin_llm_call(record, capture_mode="metadata")
                ```

        Args:
            record: Prepared provider-neutral LLM observation.
            capture_mode: Exact configured capture mode for mismatch detection.

        Returns:
            None: The in-progress call and request evidence are committed.

        Notes:
            This method never performs provider transport or terminal persistence.
        """
        self._validate_capture_mode(capture_mode)
        record.lifecycle_status = LLMCallLifecycleStatus.IN_PROGRESS.value
        draft = _llm_draft(
            record,
            owner_scope=self.owner_scope,
            policy=self.policy,
            prompt_store=self._prompt_store,
            lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
        )
        stored = await self.repository.begin_llm_call(draft)
        record.prompt_manifest_id = stored.prompt_manifest_id

    async def finish_llm_call(self, record: LLMObservationRecord, *, capture_mode: str) -> None:
        """Persist one completed, failed, or cancelled AG LLM call outcome.

        The service prepares bounded sanitized content and delegates one atomic call
        append to the selected provider.

        Examples:
            Store a successful call:
                ```python
                await service.finish_llm_call(call, capture_mode="manifest")
                ```

            Store a failed metadata call:
                ```python
                await metadata_service.finish_llm_call(failed_call, capture_mode="metadata")
                ```

        Args:
            record: Current completed AG LLM observation.
            capture_mode: Exact configured capture mode for mismatch detection.

        Returns:
            None: Persistence and correlation completion have finished.

        Notes:
            No `emit` alias or insert-on-finish path exists and no dual write is attempted.
        """
        self._validate_capture_mode(capture_mode)
        lifecycle_status = LLMCallLifecycleStatus(record.lifecycle_status)
        if lifecycle_status is LLMCallLifecycleStatus.IN_PROGRESS:
            raise ValueError("terminal LLM observation cannot remain in_progress")
        draft = _llm_draft(
            record,
            owner_scope=self.owner_scope,
            policy=self.policy,
            prompt_store=self._prompt_store,
            lifecycle_status=lifecycle_status,
        )
        stored = await self.repository.finish_llm_call(record.llm_call_id, draft)
        record.prompt_manifest_id = stored.prompt_manifest_id
        complete_llm_call_correlation(
            record.llm_call_id,
            prompt_manifest_id=stored.prompt_manifest_id,
        )

    def _validate_capture_mode(self, capture_mode: str) -> None:
        if capture_mode != self.policy.capture_mode:
            raise ValueError("LLM client capture mode does not match observation policy")


def bind_canonical_observation_service(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    policy: ObservationPolicy,
) -> CanonicalObservationService:
    """Bind AG observation writes to the bundle's exact observation field.

    The binding projects the active observation service from the already-open
    provider bundle without taking over lifecycle ownership.

    Examples:
        Bind production composition inputs:
            ```python
            service = bind_canonical_observation_service(
                bundle=bundle,
                owner_scope=owner_scope,
                policy=policy,
            )
            ```

        Bind an external fake bundle:
            ```python
            service = bind_canonical_observation_service(
                bundle=fake_bundle,
                owner_scope=StorageScope(project_id="project-1"),
                policy=ObservationPolicy(capture_mode="off"),
            )
            ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        policy: AG observation capture and retention policy.

    Returns:
        CanonicalObservationService: Provider-neutral active service projection.

    Notes:
        The binding performs no selection, I/O, fallback, close, or secondary write.
    """
    return ProviderObservationService(
        repository=bundle.observations,
        owner_scope=owner_scope,
        policy=policy,
    )


def _observation_draft(
    record: ObservationRecord,
    *,
    owner_scope: StorageScope,
    links: tuple[ObservationResourceLink, ...],
) -> ObservationDraft:
    occurred_at = _utc(record.occurred_at)
    attributes = _attributes_with_compatibility(record.attributes, record.scope.app_id)
    return ObservationDraft(
        observation_id=record.observation_id,
        category=record.category,
        name=record.name,
        summary=record.summary,
        occurred_at=occurred_at,
        scope=_storage_scope(record.scope, owner_scope=owner_scope),
        status=ObservationStatus(record.status),
        severity=ObservationSeverity(record.severity),
        producer=_producer(record),
        trace_id=record.scope.trace_id,
        turn_id=record.scope.turn_id,
        parent_observation_id=record.parent_observation_id,
        caused_by_observation_id=record.caused_by_observation_id,
        source_event_id=record.source_event_id,
        attributes=attributes,
        resource_links=links,
        payload_fragment_id=record.payload_fragment_id,
        retention_class=record.retention_class,
        expires_at=_optional_utc(record.expires_at),
    )


def _llm_draft(
    record: LLMObservationRecord,
    *,
    owner_scope: StorageScope,
    policy: ObservationPolicy,
    prompt_store: PromptStore,
    lifecycle_status: LLMCallLifecycleStatus,
) -> LLMCallDraft:
    prepared = prompt_store.prepare(record)
    mode = ObservationCaptureMode(policy.capture_mode)
    occurred_at = _utc(record.created_at)
    manifest_id = (
        None if mode is ObservationCaptureMode.OFF else f"llm-manifest:{record.llm_call_id}"
    )
    captured_request = None
    captured_response = None
    captured_trace = None
    if mode in {ObservationCaptureMode.MANIFEST, ObservationCaptureMode.FULL}:
        captured_request = _bounded_capture(
            {
                "messages": record.messages,
                "effective_messages": record.effective_messages,
                "provider_request_args": record.provider_request_args,
                "tools": record.tool_definitions,
                "continuation_inputs": record.continuation_inputs,
            },
            policy=policy,
        )
        captured_response = _bounded_capture(
            {"text": record.raw_text} if record.raw_text is not None else None,
            policy=policy,
        )
        captured_trace = _bounded_capture(record.trace_payload, policy=policy)
    status = {
        LLMCallLifecycleStatus.IN_PROGRESS: ObservationStatus.PENDING,
        LLMCallLifecycleStatus.COMPLETED: ObservationStatus.OK,
        LLMCallLifecycleStatus.FAILED: ObservationStatus.ERROR,
        LLMCallLifecycleStatus.CANCELLED: ObservationStatus.ERROR,
    }[lifecycle_status]
    observation = ObservationDraft(
        observation_id=f"llm:{record.llm_call_id}",
        category="llm",
        name=record.call_type,
        summary=f"{record.provider}/{record.model} {record.call_type}",
        occurred_at=occurred_at,
        scope=_storage_scope(record.scope, owner_scope=owner_scope),
        status=status,
        severity=(
            ObservationSeverity.ERROR
            if lifecycle_status in {LLMCallLifecycleStatus.FAILED, LLMCallLifecycleStatus.CANCELLED}
            else ObservationSeverity.INFO
        ),
        producer="aethergraph.llm",
        trace_id=record.scope.trace_id,
        turn_id=record.scope.turn_id,
        attributes=_attributes_with_compatibility(
            {
                "capture_mode": policy.capture_mode,
                "prompt_roles": prepared.roles,
                "prompt_message_count": len(record.messages),
                "prompt_chars": prepared.total_chars,
                "prompt_bytes": prepared.total_bytes,
                "assembled_request_hash": prepared.assembled_request_hash,
                "request_hash_version": prepared.request_hash_version,
                "omission_reason": prepared.omission_reason,
            },
            record.scope.app_id,
        ),
        retention_class="forensic" if mode is ObservationCaptureMode.FULL else "standard",
        expires_at=(
            occurred_at + timedelta(days=policy.full_prompt_ttl_days)
            if mode is ObservationCaptureMode.FULL
            else None
        ),
    )
    return LLMCallDraft(
        llm_call_id=record.llm_call_id,
        observation=observation,
        call_type=record.call_type,
        provider=record.provider,
        model=record.model,
        capture_mode=mode,
        lifecycle_status=lifecycle_status,
        profile_name=record.profile_name,
        call_name=record.call_name,
        request_options={
            "reasoning_effort": record.reasoning_effort,
            "max_output_tokens": record.max_output_tokens,
            "output_format": record.output_format,
            "json_schema": record.json_schema,
            "schema_name": record.schema_name,
            "strict_schema": record.strict_schema,
            "validate_json": record.validate_json,
            "extra_params": record.extra_params,
            "request_args": record.request_args,
            "provider_request_args": record.provider_request_args,
            "compatibility_notes": record.compatibility_notes,
        },
        usage=record.usage,
        latency_ms=record.latency_ms,
        error_type=record.error_type,
        error_message=record.error_message,
        prompt_manifest_id=manifest_id,
        request_preview={
            "message_count": len(record.messages),
            "roles": prepared.roles,
            "chars": prepared.total_chars,
            "bytes": prepared.total_bytes,
            "hash": prepared.assembled_request_hash,
            "request_hash_version": prepared.request_hash_version,
            "omission_reason": prepared.omission_reason,
        },
        response_preview=(
            {"text_chars": len(record.raw_text)} if record.raw_text is not None else None
        ),
        captured_request=captured_request,
        captured_response=captured_response,
        trace_payload=captured_trace,
        attempts=tuple(_attempt(attempt) for attempt in record.attempts),
        tool_surface=record.tool_surface,
        request_items=record.request_items,
        response_items=record.response_items,
        provider_request_facts=record.provider_request_facts,
    )


def _storage_scope(scope: ObservationScope, *, owner_scope: StorageScope) -> StorageScope:
    dimensions = {
        name: value
        for name in _CANONICAL_SCOPE_NAMES
        if (value := getattr(scope, name)) is not None
    }
    return merge_storage_scope(owner_scope, **dimensions)


def _attributes_with_compatibility(
    attributes: Mapping[str, Any], app_id: str | None
) -> dict[str, Any]:
    projected = dict(attributes)
    if _COMPATIBILITY_METADATA in projected:
        raise ValueError("observation attributes reserve compatibility_metadata")
    if app_id is not None:
        if not isinstance(app_id, str) or not app_id.strip():
            raise ValueError("deprecated app_id must be a non-empty string when supplied")
        projected[_COMPATIBILITY_METADATA] = {
            _DEPRECATED_APP_ID: {
                "value": app_id,
                "deprecated": True,
                "compatibility_only": True,
            }
        }
    return projected


def _producer(record: ObservationRecord) -> str | None:
    if record.category in {"service_operation", "trace"}:
        value = record.attributes.get("service")
    elif record.category == "log":
        value = record.attributes.get("logger")
    else:
        value = record.attributes.get("producer")
    return str(value) if isinstance(value, str) and value.strip() else None


def _resource_link(value: Mapping[str, Any]) -> ObservationResourceLink:
    return ObservationResourceLink(
        resource_key=str(value["resource_key"]),
        relation=ObservationResourceRelation(str(value["relation"])),
        resource_revision=value.get("resource_revision", value.get("revision")),
        content_hash=value.get("content_hash"),
        slot_key=value.get("slot_key"),
    )


def _attempt(value: Any) -> LLMCallAttempt:
    return LLMCallAttempt(
        attempt_number=value.attempt_number,
        elapsed_ms=round(value.elapsed_s * 1_000),
        outcome=value.outcome,
        retryable=value.retryable,
        status_code=value.status_code,
        error_code=value.error_code,
        request_id=value.request_id,
        provider_delay_ms=(
            round(value.provider_delay_s * 1_000) if value.provider_delay_s is not None else None
        ),
        scheduled_delay_ms=(
            round(value.scheduled_delay_s * 1_000) if value.scheduled_delay_s is not None else None
        ),
        rate_limits=tuple(asdict(snapshot) for snapshot in value.rate_limits),
    )


def _bounded_capture(value: Any, *, policy: ObservationPolicy) -> Any:
    if value is None:
        return None
    sanitized = sanitize_content(value)
    if len(canonical_json(sanitized).encode()) > policy.max_fragment_bytes:
        return None
    return sanitized


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _optional_utc(value: str | None) -> datetime | None:
    return _utc(value) if value is not None else None


__all__ = [
    "CanonicalObservationService",
    "ProviderObservationService",
    "bind_canonical_observation_service",
]

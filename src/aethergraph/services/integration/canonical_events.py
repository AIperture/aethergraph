"""One-way Host projections for canonical inbound and semantic repositories."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from aethergraph.contracts.integration import (
    ExternalSessionBinding,
    IngressEnvelope,
    IntegrationRoute,
    SemanticEvent,
    SemanticEventKind,
)
from aethergraph.services.canonical_storage_scope import (
    merge_storage_scope,
    validate_storage_owner_scope,
)
from aethergraph.services.channel.resources import InputResource
from aethergraph.storage.contracts import (
    InboundEventDraft,
    InboundEventRepository,
    PageRequest,
    SemanticEventDraft,
    SemanticEventKind as StorageSemanticEventKind,
    SemanticEventQuery,
    SemanticEventRecord,
    SemanticEventRepository,
    StorageIntegrityError,
    StorageScope,
)

from .event_contracts import (
    PersistedInboundEvent,
    PersistedSemanticEvent,
    SemanticEventStoreError,
)

_MAX_PAGE_SIZE = 1_000
_DEFAULT_MAX_HISTORY_EVENTS = 10_000


class CanonicalInboundEventStore:
    """Project validated Host ingress onto one canonical inbound repository."""

    def __init__(
        self,
        *,
        repository: InboundEventRepository,
        owner_scope: StorageScope,
    ) -> None:
        """Bind inbound persistence to one provider-authoritative Host owner.

        Host DTOs are normalized into canonical identity fields, JSON payload, and
        materialized artifact keys before repository access. The projection selects
        no provider and retains no physical path.

        Examples:
            Bind an opened bundle repository:
                ```python
                store = CanonicalInboundEventStore(
                    repository=bundle.inbound_events,
                    owner_scope=owner_scope,
                )
                ```

            Bind a deterministic fake repository:
                ```python
                store = CanonicalInboundEventStore(
                    repository=fake_repository,
                    owner_scope=StorageScope(project_id="project-1"),
                )
                ```

        Args:
            repository: Canonical inbound repository from one coherent bundle.
            owner_scope: Exact trusted Host ownership scope.

        Returns:
            None: The provider-backed service projection is ready without I/O.

        Notes:
            Deprecated App/client metadata is not accepted as scope or identity.
        """
        validate_storage_owner_scope(owner_scope)
        self._repository = repository
        self._owner_scope = owner_scope

    async def append(
        self,
        *,
        deployment_id: str,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        envelope: IngressEnvelope,
        resources: tuple[InputResource, ...],
    ) -> PersistedInboundEvent:
        """Normalize and append one accepted Host ingress event.

        The AG session is merged into trusted owner scope. Every resource must already
        be materialized to an Artifact identity; paths, URI aliases, raw bytes, and
        mutable resource objects never cross the canonical repository boundary.

        Examples:
            Persist text ingress:
                ```python
                stored = await store.append(
                    deployment_id="deployment-1",
                    route=route,
                    binding=binding,
                    envelope=envelope,
                    resources=(),
                )
                ```

            Persist Artifact-backed attachments:
                ```python
                stored = await store.append(
                    deployment_id=deployment_id,
                    route=route,
                    binding=binding,
                    envelope=envelope,
                    resources=resources,
                )
                ```

        Args:
            deployment_id: Exact Host deployment identity.
            route: Exact immutable manifest route.
            binding: Durable external-to-AG session binding.
            envelope: Validated immutable ingress command.
            resources: Materialized artifact-backed input resources.

        Returns:
            PersistedInboundEvent: Stable event identity and delivery cursor.

        Notes:
            Repository identity conflicts fail directly; no alternate event log or
            generated replacement identity is attempted after an append failure.
        """
        scope = merge_storage_scope(self._owner_scope, session_id=binding.ag_session_id)
        resource_payloads = tuple(_canonical_resource(resource) for resource in resources)
        resource_keys = tuple(f"artifact:{payload['artifact_id']}" for payload in resource_payloads)
        event_id = f"ingress-{uuid4().hex}"
        try:
            record = await self._repository.append(
                InboundEventDraft(
                    event_id=event_id,
                    deployment_id=deployment_id,
                    route_id=route.route_id,
                    integration_id=envelope.integration_id,
                    external_event_id=envelope.external_event_id,
                    received_at=envelope.received_at,
                    scope=scope,
                    payload=_normalized_inbound_payload(
                        route=route,
                        binding=binding,
                        envelope=envelope,
                        resources=resource_payloads,
                    ),
                    resource_keys=resource_keys,
                )
            )
        except StorageIntegrityError as exc:
            raise SemanticEventStoreError(
                code="integration.semantic_event_conflict",
                message="Inbound event identity or external event identity conflicts.",
            ) from exc
        return PersistedInboundEvent(
            cursor=record.delivery_cursor,
            event_id=record.event_id,
        )


class CanonicalSemanticEventStore:
    """Project frozen semantic v2 events onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: SemanticEventRepository,
        owner_scope: StorageScope,
        max_history_events: int = _DEFAULT_MAX_HISTORY_EVENTS,
    ) -> None:
        """Bind semantic persistence to one provider-authoritative Host owner.

        The exact active v2 kind, authored sequence, structured payload, extensions,
        and session scope are normalized before repository access. Bounded history
        reads translate durable integer delivery cursors independently of opaque
        provider pagination cursors.

        Examples:
            Bind an opened bundle repository:
                ```python
                store = CanonicalSemanticEventStore(
                    repository=bundle.semantic_events,
                    owner_scope=owner_scope,
                )
                ```

            Configure a smaller history ceiling:
                ```python
                store = CanonicalSemanticEventStore(
                    repository=fake_repository,
                    owner_scope=StorageScope(project_id="project-1"),
                    max_history_events=1_000,
                )
                ```

        Args:
            repository: Canonical semantic repository from one coherent bundle.
            owner_scope: Exact trusted Host ownership scope.
            max_history_events: Positive ceiling for calls without an explicit limit.

        Returns:
            None: The provider-backed service projection is ready without I/O.

        Notes:
            Legacy semantic v1 events and catch-all kind fallback are not supported.
        """
        validate_storage_owner_scope(owner_scope)
        if (
            isinstance(max_history_events, bool)
            or not isinstance(max_history_events, int)
            or not 1 <= max_history_events <= 100_000
        ):
            raise ValueError("max_history_events must be between 1 and 100000")
        self._repository = repository
        self._owner_scope = owner_scope
        self._max_history_events = max_history_events

    async def append(self, event: SemanticEvent) -> PersistedSemanticEvent:
        """Persist one exact semantic v2 event at its authored sequence.

        The active Host kind maps one-to-one to the canonical storage vocabulary.
        Structured payload and namespaced extensions are stored as normalized JSON,
        while provider assignment supplies the durable delivery cursor.

        Examples:
            Persist message completion:
                ```python
                persisted = await store.append(message_completed)
                ```

            Persist a terminal outcome:
                ```python
                persisted = await store.append(turn_outcome)
                ```

        Args:
            event: Frozen active semantic v2 event.

        Returns:
            PersistedSemanticEvent: Exact event paired with its integer delivery cursor.

        Notes:
            Identity or authored-sequence reuse fails without renumbering or fallback.
        """
        scope = merge_storage_scope(self._owner_scope, session_id=event.session_id)
        try:
            record = await self._repository.append(
                SemanticEventDraft(
                    event_id=event.event_id,
                    deployment_id=event.deployment_id,
                    turn_id=event.turn_id,
                    sequence=event.sequence,
                    producer=event.producer,
                    occurred_at=event.timestamp,
                    kind=StorageSemanticEventKind(event.kind.value),
                    scope=scope,
                    payload={
                        "protocol": event.schema_version,
                        "payload": event.payload.model_dump(mode="json"),
                        "extensions": event.extensions,
                    },
                )
            )
        except StorageIntegrityError as exc:
            raise SemanticEventStoreError(
                code="integration.semantic_event_conflict",
                message="Semantic event identity or turn sequence already exists.",
            ) from exc
        return PersistedSemanticEvent(cursor=record.delivery_cursor, event=event)

    async def list_session(
        self,
        *,
        deployment_id: str,
        session_id: str,
        after_cursor: int | None = None,
        limit: int | None = None,
    ) -> tuple[PersistedSemanticEvent, ...]:
        """Read bounded delivery-cursor-ordered semantic v2 history.

        The durable integer cursor filters inside the provider before opaque page
        traversal. Calls without an explicit limit may return at most the configured
        history ceiling and fail explicitly if more matching events exist.

        Examples:
            Read initial bounded history:
                ```python
                history = await store.list_session(
                    deployment_id="deployment-1",
                    session_id="session-1",
                    limit=100,
                )
                ```

            Resume after delivery:
                ```python
                delta = await store.list_session(
                    deployment_id="deployment-1",
                    session_id="session-1",
                    after_cursor=last_cursor,
                    limit=100,
                )
                ```

        Args:
            deployment_id: Exact Host deployment identity.
            session_id: Exact AG session identity.
            after_cursor: Optional exclusive non-negative delivery cursor.
            limit: Optional positive result limit; the configured ceiling applies when absent.

        Returns:
            tuple[PersistedSemanticEvent, ...]: Validated events in delivery order.

        Notes:
            Opaque provider cursors never cross this frozen Host service boundary.
        """
        requested = self._max_history_events if limit is None else limit
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError("limit must be a positive integer when supplied")
        if requested > self._max_history_events:
            raise ValueError("limit exceeds max_history_events")
        if after_cursor is not None and (
            isinstance(after_cursor, bool) or not isinstance(after_cursor, int) or after_cursor < 0
        ):
            raise ValueError("after_cursor must be a non-negative integer when supplied")

        scope = merge_storage_scope(self._owner_scope, session_id=session_id)
        out: list[PersistedSemanticEvent] = []
        opaque_cursor: str | None = None
        previous_delivery_cursor = after_cursor or 0
        while len(out) < requested:
            page = await self._repository.query(
                SemanticEventQuery(
                    deployment_id=deployment_id,
                    scope=scope,
                    page=PageRequest(
                        limit=min(_MAX_PAGE_SIZE, requested - len(out)),
                        cursor=opaque_cursor,
                    ),
                    after_delivery_cursor=after_cursor,
                )
            )
            for record in page.items:
                if record.delivery_cursor <= previous_delivery_cursor:
                    raise SemanticEventStoreError(
                        code="integration.semantic_event_corrupt",
                        message="Canonical semantic delivery cursors are not ascending.",
                    )
                out.append(_persisted_semantic(record))
                previous_delivery_cursor = record.delivery_cursor
            opaque_cursor = page.next_cursor
            if opaque_cursor is None:
                return tuple(out)
        if opaque_cursor is not None and limit is None:
            raise SemanticEventStoreError(
                code="integration.semantic_event_history_limit",
                message="Semantic event history exceeds the configured bounded limit.",
            )
        return tuple(out)


def _canonical_resource(resource: InputResource) -> dict[str, Any]:
    if resource.status != "materialized" or not resource.artifact_id:
        raise ValueError("canonical inbound resources must be materialized Artifacts")
    payload: dict[str, Any] = {
        "artifact_id": resource.artifact_id,
        "source": resource.source,
        "status": resource.status,
    }
    for key, value in (
        ("name", resource.name),
        ("media_type", resource.mime),
        ("size_bytes", resource.size),
    ):
        if value is not None:
            payload[key] = value
    if resource.labels:
        payload["labels"] = dict(resource.labels)
    return payload


def _normalized_inbound_payload(
    *,
    route: IntegrationRoute,
    binding: ExternalSessionBinding,
    envelope: IngressEnvelope,
    resources: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "route": {
            "entry_agent_id": route.entry_agent_id,
            "integration_kind": route.integration_kind.value,
            "endpoint_id": route.endpoint_id,
        },
        "binding": {
            "binding_id": binding.binding_id,
            "build_id": binding.build_id,
        },
        "external_identity": envelope.external_identity.model_dump(mode="json"),
        "command": {
            "text": envelope.text,
            "choice": envelope.choice.model_dump(mode="json") if envelope.choice else None,
            "structured_input": envelope.structured_input,
            "transport_metadata": envelope.transport_metadata,
            "origin_address": envelope.origin_address.model_dump(mode="json"),
            "attachments": tuple(
                attachment.model_dump(mode="json") for attachment in envelope.attachments
            ),
        },
        "resources": resources,
    }


def _persisted_semantic(record: SemanticEventRecord) -> PersistedSemanticEvent:
    payload = record.payload.get("payload")
    extensions = record.payload.get("extensions", {})
    protocol = record.payload.get("protocol")
    if not isinstance(payload, Mapping) or not isinstance(extensions, Mapping):
        raise SemanticEventStoreError(
            code="integration.semantic_event_corrupt",
            message="Canonical semantic payload is malformed.",
        )
    try:
        event = SemanticEvent.model_validate(
            {
                "schema_version": protocol,
                "event_id": record.event_id,
                "deployment_id": record.deployment_id,
                "session_id": record.scope.session_id,
                "turn_id": record.turn_id,
                "sequence": record.sequence,
                "producer": record.producer,
                "timestamp": record.occurred_at,
                "kind": SemanticEventKind(record.kind.value),
                "payload": _thaw_json(payload),
                "extensions": _thaw_json(extensions),
            }
        )
    except (TypeError, ValueError) as exc:
        raise SemanticEventStoreError(
            code="integration.semantic_event_corrupt",
            message="Canonical semantic event cannot be projected to active v2.",
        ) from exc
    return PersistedSemanticEvent(cursor=record.delivery_cursor, event=event)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value

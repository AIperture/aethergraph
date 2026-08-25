"""Provider-neutral Host ingress and semantic-event service contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from aethergraph.contracts.integration import (
    ExternalSessionBinding,
    IngressEnvelope,
    IntegrationRoute,
    SemanticEvent,
)
from aethergraph.services.channel.resources import InputResource


class SemanticEventStoreError(RuntimeError):
    """Structured failure raised when semantic-event persistence is invalid."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.semantic_event_conflict",
            "integration.semantic_event_cursor_required",
            "integration.semantic_event_corrupt",
            "integration.semantic_event_history_limit",
        ],
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class PersistedSemanticEvent:
    """One current semantic Event paired with its provider delivery cursor."""

    cursor: int
    event: SemanticEvent


@dataclass(frozen=True, slots=True)
class PersistedInboundEvent:
    """One canonical inbound event paired with its provider delivery cursor."""

    cursor: int
    event_id: str


class InboundEventStore(Protocol):
    """Provider-neutral persistence for validated materialized Host ingress."""

    async def append(
        self,
        *,
        deployment_id: str,
        route: IntegrationRoute,
        binding: ExternalSessionBinding,
        envelope: IngressEnvelope,
        resources: tuple[InputResource, ...],
    ) -> PersistedInboundEvent:
        """Persist one validated ingress event before runtime dispatch.

        Examples:
            Persist text ingress:
                ```python
                stored = await inbound.append(
                    deployment_id="deployment-1",
                    route=route,
                    binding=binding,
                    envelope=envelope,
                    resources=(),
                )
                ```

            Persist materialized resources:
                ```python
                stored = await inbound.append(
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
            Implementations must not persist raw transport payloads or secret bytes.
        """
        ...


class SemanticEventStore(Protocol):
    """Provider-neutral ordered semantic-event persistence contract."""

    async def append(self, event: SemanticEvent) -> PersistedSemanticEvent:
        """Append one current semantic Event at its authored sequence.

        Examples:
            Persist a completed message:
                ```python
                persisted = await store.append(event)
                ```

            Retain its delivery cursor:
                ```python
                receipt_cursor = persisted.cursor
                ```

        Args:
            event: Closed current semantic Event.

        Returns:
            PersistedSemanticEvent: Event paired with its provider delivery cursor.

        Notes:
            Event identity and turn sequence are single-assignment.
        """
        ...

    async def list_session(
        self,
        *,
        deployment_id: str,
        session_id: str,
        after_cursor: int | None = None,
        limit: int | None = None,
    ) -> tuple[PersistedSemanticEvent, ...]:
        """Read ordered current semantic Events from one deployment session.

        Examples:
            Read bounded history:
                ```python
                events = await store.list_session(
                    deployment_id="deployment-1",
                    session_id="session-1",
                    limit=100,
                )
                ```

            Resume after delivery:
                ```python
                events = await store.list_session(
                    deployment_id="deployment-1",
                    session_id="session-1",
                    after_cursor=cursor,
                    limit=100,
                )
                ```

        Args:
            deployment_id: Exact Host deployment identity.
            session_id: Exact AG session identity.
            after_cursor: Optional exclusive provider delivery cursor.
            limit: Optional positive result limit.

        Returns:
            tuple[PersistedSemanticEvent, ...]: Events ordered by cursor ascending.

        Notes:
            History and live reconnect use the shared provider cursor domain.
        """
        ...

"""Canonical semantic-event persistence over the shared AG event log."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Literal, Protocol

from aethergraph.contracts.integration import SemanticEvent
from aethergraph.contracts.storage.event_log import EventLog


class SemanticEventStoreError(RuntimeError):
    """Structured failure raised when semantic event persistence is invalid."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.semantic_event_conflict",
            "integration.semantic_event_cursor_required",
            "integration.semantic_event_corrupt",
        ],
        message: str,
    ) -> None:
        """Create one stable semantic-event persistence failure.

        Examples:
            Report duplicate event identity:
            ```python
            SemanticEventStoreError(
                code="integration.semantic_event_conflict",
                message="Event identity or sequence already exists.",
            )
            ```

            Report invalid persisted data:
            ```python
            SemanticEventStoreError(
                code="integration.semantic_event_corrupt",
                message="Stored semantic event is invalid.",
            )
            ```

        Args:
            code: Stable machine-readable event-store failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Failures never skip, renumber, or rewrite a semantic event.
        """
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PersistedSemanticEvent:
    """One validated semantic event paired with its durable event-log cursor."""

    cursor: int
    event: SemanticEvent


class SemanticEventStore(Protocol):
    """Provider-neutral ordered semantic-event persistence contract."""

    async def append(self, event: SemanticEvent) -> PersistedSemanticEvent:
        """Append one event at its authored turn sequence.

        Examples:
            Persist a completed message:
            ```python
            persisted = await store.append(event)
            ```

            Retain the delivery cursor:
            ```python
            receipt_cursor = persisted.cursor
            ```

        Args:
            event: Closed semantic event contract.

        Returns:
            PersistedSemanticEvent: Event paired with its durable cursor.

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
        """Read ordered semantic events from one deployment session.

        Examples:
            Read complete history:
            ```python
            events = await store.list_session(
                deployment_id="deployment-1",
                session_id="session-1",
            )
            ```

            Resume after a delivered cursor:
            ```python
            events = await store.list_session(
                deployment_id="deployment-1",
                session_id="session-1",
                after_cursor=cursor,
            )
            ```

        Args:
            deployment_id: Exact host deployment identity.
            session_id: Exact AG session identity.
            after_cursor: Optional exclusive event-log cursor.
            limit: Optional maximum result count.

        Returns:
            tuple[PersistedSemanticEvent, ...]: Events ordered by cursor ascending.

        Notes:
            History and live reconnect use this same cursor contract.
        """
        ...


class EventLogSemanticEventStore:
    """Persist semantic events in the shared canonical AG EventLog."""

    def __init__(self, event_log: EventLog) -> None:
        """Bind semantic persistence to the host's existing event log.

        Examples:
            Bind a SQLite event log:
            ```python
            store = EventLogSemanticEventStore(event_log)
            ```

            Share the same log used by other AG runtime services:
            ```python
            store = EventLogSemanticEventStore(container.eventlog)
            ```

        Args:
            event_log: Canonical AG event log with durable append cursors.

        Returns:
            None.

        Notes:
            This service does not create a parallel integration event database.
        """
        self.event_log = event_log

    async def append(self, event: SemanticEvent) -> PersistedSemanticEvent:
        """Append one closed semantic event and return its durable cursor.

        Examples:
            Persist an event:
            ```python
            persisted = await store.append(event)
            ```

            Use the cursor in an ingress receipt:
            ```python
            cursor = (await store.append(event)).cursor
            ```

        Args:
            event: Closed semantic event contract.

        Returns:
            PersistedSemanticEvent: Exact event and assigned EventLog cursor.

        Notes:
            SQLite uniqueness rejects duplicate event IDs and duplicate sequence
            values within one deployment/session/turn.
        """
        row = {
            "scope_id": _scope_id(event.deployment_id, event.session_id),
            "_partition_scope_id": _scope_id(event.deployment_id, event.session_id),
            "kind": event.kind.value,
            "tags": ["semantic-event", f"semantic-kind:{event.kind.value}"],
            "ts": event.timestamp,
            "deployment_id": event.deployment_id,
            "session_id": event.session_id,
            "semantic_event_id": event.event_id,
            "semantic_turn_id": event.turn_id,
            "semantic_sequence": event.sequence,
            "semantic_event": event.model_dump(mode="json"),
        }
        try:
            cursor = await self.event_log.append(row)
        except sqlite3.IntegrityError as exc:
            raise SemanticEventStoreError(
                code="integration.semantic_event_conflict",
                message="Semantic event identity or turn sequence already exists.",
            ) from exc
        if not isinstance(cursor, int) or cursor < 1:
            raise SemanticEventStoreError(
                code="integration.semantic_event_cursor_required",
                message="Canonical EventLog append did not return a durable cursor.",
            )
        return PersistedSemanticEvent(cursor=cursor, event=event)

    async def list_session(
        self,
        *,
        deployment_id: str,
        session_id: str,
        after_cursor: int | None = None,
        limit: int | None = None,
    ) -> tuple[PersistedSemanticEvent, ...]:
        """Read and validate cursor-ordered events for one deployment session.

        Examples:
            Read complete history:
            ```python
            history = await store.list_session(
                deployment_id="deployment-1",
                session_id="session-1",
            )
            ```

            Read reconnect events:
            ```python
            delta = await store.list_session(
                deployment_id="deployment-1",
                session_id="session-1",
                after_cursor=last_cursor,
            )
            ```

        Args:
            deployment_id: Exact host deployment identity.
            session_id: Exact AG session identity.
            after_cursor: Optional exclusive durable cursor.
            limit: Optional maximum result count.

        Returns:
            tuple[PersistedSemanticEvent, ...]: Validated events in cursor order.

        Notes:
            Invalid stored rows fail the read instead of being skipped.
        """
        rows = await self.event_log.query(
            scope_id=_scope_id(deployment_id, session_id),
            after_id=after_cursor,
            limit=limit,
            order_dir="asc",
        )
        out: list[PersistedSemanticEvent] = []
        for row in rows:
            cursor = row.get("_row_id")
            payload = row.get("semantic_event")
            if not isinstance(cursor, int) or not isinstance(payload, dict):
                raise SemanticEventStoreError(
                    code="integration.semantic_event_corrupt",
                    message="Canonical EventLog contains an invalid semantic event row.",
                )
            try:
                event = SemanticEvent.model_validate(payload)
            except Exception as exc:
                raise SemanticEventStoreError(
                    code="integration.semantic_event_corrupt",
                    message="Canonical EventLog contains an invalid semantic event payload.",
                ) from exc
            out.append(PersistedSemanticEvent(cursor=cursor, event=event))
        return tuple(out)


def _scope_id(deployment_id: str, session_id: str) -> str:
    return f"semantic:{deployment_id}:{session_id}"

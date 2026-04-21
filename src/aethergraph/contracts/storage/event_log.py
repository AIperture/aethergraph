from datetime import datetime
from typing import Protocol

"""
Event log interface for appending and querying events.

Typical implementations include:
- InMemoryEventLog: Transient, in-memory event log for testing or ephemeral use cases
- FSPersistenceEventLog: File system-based event log for durable storage
- DatabaseEventLog: (future) Database-backed event log for scalable storage and querying

It is used in various parts of the system for logging events with metadata.
- memory persistent implementation for saving events durably
- graph state store for appending state change events
"""


class EventLog(Protocol):
    async def append(self, evt: dict) -> None: ...

    async def query(
        self,
        *,
        scope_id: str | None = None,  # filter by scope ID, e.g., run ID, memory ID
        since: datetime | None = None,  # filter events after this time
        until: datetime | None = None,  # filter events before this time
        kinds: list[str] | None = None,  # filter by event kinds
        limit: int | None = None,  # max number of events to return
        tags: list[str] | None = None,  # filter by tags
        offset: int = 0,  # pagination offset
        user_id: str | None = None,
        org_id: str | None = None,
        client_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        after_id: int | None = None,  # keyset cursor: return events with id > after_id
        before_id: int | None = None,  # keyset cursor: return events with id < before_id (backward)
    ) -> list[dict]: ...

    async def get_many(
        self,
        scope_id: str,
        event_ids: list[str],
    ) -> list[dict]: ...

    """Fetch events for a given scope_id (timeline) by event_id."""

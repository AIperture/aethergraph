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
    ) -> list[dict]: ...

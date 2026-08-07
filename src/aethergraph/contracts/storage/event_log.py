from datetime import datetime
from typing import Literal, Protocol

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


class StateSnapshotConflictError(RuntimeError):
    """Report a failed storage-level state snapshot revision comparison."""

    def __init__(self, *, key: str, expected_revision: int, actual_revision: int) -> None:
        self.key = str(key)
        self.expected_revision = int(expected_revision)
        self.actual_revision = int(actual_revision)
        super().__init__(
            f"State snapshot {self.key!r} revision changed: expected "
            f"{self.expected_revision}, actual {self.actual_revision}"
        )


class EventLog(Protocol):
    async def append(self, evt: dict) -> int:
        """Append one event and return its durable cursor.

        Examples:
            Append a runtime event:
            ```python
            cursor = await event_log.append(event)
            ```

            Persist a scoped event:
            ```python
            cursor = await event_log.append({"scope_id": "session-1", "kind": "message"})
            ```

        Args:
            evt: Event mapping accepted by the configured persistence backend.

        Returns:
            int: Durable monotonically increasing event-log cursor.

        Notes:
            Cursor ordering is the history and reconnect ordering contract.
        """
        ...

    async def append_state_snapshot_if_revision(
        self,
        evt: dict,
        *,
        state_key: str,
        expected_revision: int,
    ) -> int:
        """Append one state snapshot only at the expected durable revision.

        Intro:
            The revision read and append occur in one backend transaction or
            cross-process critical section for the snapshot scope, kind, and key.

        Examples:
            Create the first state revision:
            ```python
            cursor = await event_log.append_state_snapshot_if_revision(
                event,
                state_key="agent:writer",
                expected_revision=0,
            )
            ```

            Reject a stale writer:
            ```python
            with pytest.raises(StateSnapshotConflictError):
                await event_log.append_state_snapshot_if_revision(
                    stale_event,
                    state_key="agent:writer",
                    expected_revision=0,
                )
            ```

        Args:
            evt: Complete state snapshot Event mapping to append.
            state_key: Exact logical state key carried by the snapshot.
            expected_revision: Exact current durable enclosing revision.

        Returns:
            int: Durable monotonically increasing event-log cursor.

        Notes:
            The appended payload must carry revision `expected_revision + 1`.
        """
        ...

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
        order_dir: Literal["asc", "desc"] = "desc",
    ) -> list[dict]: ...

    async def get_many(
        self,
        scope_id: str,
        event_ids: list[str],
    ) -> list[dict]: ...

    """Fetch events for a given scope_id (timeline) by event_id."""

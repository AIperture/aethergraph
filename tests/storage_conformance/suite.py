from __future__ import annotations

from datetime import UTC, datetime

import pytest

from aethergraph.storage.contracts import (
    EventDraft,
    EventQuery,
    EventStore,
    PageRequest,
    StateHistoryQuery,
    StateStore,
    StorageConflictError,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)


def event(event_id: str, scope: StorageScope, *, topic: str = "memory") -> EventDraft:
    return EventDraft(
        event_id=event_id,
        occurred_at=NOW,
        scope=scope,
        kind="memory.event",
        topic=topic,
        tags=("memory",),
        payload={"id": event_id},
    )


async def check_event_store_conformance(store: EventStore) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1", session_id="s-1")
    other_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-2",
        session_id="s-1",
    )

    first = await store.append(event("event-1", scope))
    batch = await store.append_many((event("event-2", scope), event("event-3", scope)))

    assert first.event_id == "event-1"
    assert [row.event_id for row in batch] == ["event-2", "event-3"]
    assert len({first.cursor, *(row.cursor for row in batch)}) == 3
    assert await store.get(scope, "event-1") == first
    assert await store.get(other_scope, "event-1") is None
    assert await store.append_many(()) == ()

    page_one = await store.query(EventQuery(scope=scope, page=PageRequest(limit=2)))
    assert [row.event_id for row in page_one.items] == ["event-3", "event-2"]
    assert page_one.next_cursor is not None
    page_two = await store.query(
        EventQuery(
            scope=scope,
            page=PageRequest(limit=2, cursor=page_one.next_cursor),
        )
    )
    assert [row.event_id for row in page_two.items] == ["event-1"]
    assert page_two.next_cursor is None


async def check_state_store_conformance(store: StateStore) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1", run_id="run-1")
    other_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-2",
        run_id="run-1",
    )

    assert await store.get(scope, "agent", "writer") is None
    created = await store.compare_and_set(
        scope,
        "agent",
        "writer",
        0,
        {"count": 1},
        {"reason": "create"},
    )
    assert created.revision == 1
    assert await store.get(scope, "agent", "writer") == created
    assert await store.get(other_scope, "agent", "writer") is None

    with pytest.raises(StorageConflictError):
        await store.compare_and_set(scope, "agent", "writer", 0, {"count": 2}, {})
    updated = await store.compare_and_set(
        scope,
        "agent",
        "writer",
        1,
        {"count": 2},
        {"reason": "advance"},
    )
    assert updated.revision == 2

    hydrated = await store.get_many(scope, "agent", ("writer", "missing"))
    assert hydrated == (updated, None)
    history = await store.history(
        StateHistoryQuery(
            scope=scope,
            namespace="agent",
            key="writer",
            page=PageRequest(limit=1),
        )
    )
    assert history.items == (updated,)
    assert history.next_cursor is not None

    with pytest.raises(StorageConflictError):
        await store.delete(scope, "agent", "writer", 1)
    assert await store.delete(scope, "agent", "writer", 2) is True
    assert await store.delete(scope, "agent", "writer", 0) is False

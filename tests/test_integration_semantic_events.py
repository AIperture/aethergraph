from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from aethergraph.contracts.integration import (
    MessageCompletedPayload,
    SemanticEvent,
    SemanticEventKind,
)
from aethergraph.services.integration import (
    EventLogSemanticEventStore,
    SemanticEventStoreError,
)
from aethergraph.storage.eventlog.fs_event import FSEventLog
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog

_NOW = datetime(2026, 8, 3, tzinfo=UTC)


def _event(*, event_id: str, sequence: int, text: str) -> SemanticEvent:
    return SemanticEvent(
        event_id=event_id,
        deployment_id="deployment-1",
        session_id="session-1",
        turn_id="turn-1",
        sequence=sequence,
        producer="agent.support",
        timestamp=_NOW + timedelta(seconds=sequence),
        kind=SemanticEventKind.MESSAGE_COMPLETED,
        payload=MessageCompletedPayload(
            message_id=f"message-{sequence}",
            text=text,
        ),
    )


@pytest.mark.asyncio
async def test_semantic_events_share_sqlite_log_and_resume_from_cursor(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    first = _event(event_id="event-1", sequence=0, text="First")
    second = _event(event_id="event-2", sequence=1, text="Second")

    first_record = await store.append(first)
    second_record = await store.append(second)

    assert first_record.cursor < second_record.cursor
    assert await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    ) == (first_record, second_record)
    assert await store.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
        after_cursor=first_record.cursor,
    ) == (second_record,)

    await event_log.close()
    restored = EventLogSemanticEventStore(SqliteEventLog(str(tmp_path / "events.db")))
    restored_rows = await restored.list_session(
        deployment_id="deployment-1",
        session_id="session-1",
    )
    assert restored_rows == (first_record, second_record)
    await restored.event_log.close()


@pytest.mark.asyncio
async def test_sqlite_semantic_events_reject_duplicate_identity_and_sequence(tmp_path) -> None:
    event_log = SqliteEventLog(str(tmp_path / "events.db"))
    store = EventLogSemanticEventStore(event_log)
    await store.append(_event(event_id="event-1", sequence=0, text="First"))

    with pytest.raises(SemanticEventStoreError) as exc_info:
        await store.append(_event(event_id="event-1", sequence=1, text="Duplicate ID"))
    assert exc_info.value.code == "integration.semantic_event_conflict"

    with pytest.raises(SemanticEventStoreError) as exc_info:
        await store.append(_event(event_id="event-2", sequence=0, text="Duplicate sequence"))
    assert exc_info.value.code == "integration.semantic_event_conflict"
    await event_log.close()


@pytest.mark.asyncio
async def test_filesystem_event_log_returns_and_filters_durable_cursors(tmp_path) -> None:
    event_log = FSEventLog(str(tmp_path / "events"))
    first_cursor = await event_log.append(
        {"scope_id": "scope-1", "kind": "test", "ts": _NOW, "value": "first"}
    )
    second_cursor = await event_log.append(
        {"scope_id": "scope-1", "kind": "test", "ts": _NOW, "value": "second"}
    )

    rows = await event_log.query(
        scope_id="scope-1",
        after_id=first_cursor,
        order_dir="asc",
    )

    assert second_cursor == first_cursor + 1
    assert [(row["_row_id"], row["value"]) for row in rows] == [(second_cursor, "second")]

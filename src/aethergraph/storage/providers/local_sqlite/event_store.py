"""Ordered canonical event streams for the local SQLite provider."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import re
import sqlite3
from typing import Any

from ...contracts import (
    EventDraft,
    EventQuery,
    EventRecord,
    Page,
    SortDirection,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalSQLiteDatabase

_EVENT_COMPONENT_VERSION = 2
_MAX_APPEND_BATCH = 1_000
_STREAM_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SCOPE_FIELDS = tuple(item.name for item in fields(StorageScope))
_CREATE_EVENTS = f"""
CREATE TABLE local_events (
    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
    stream TEXT NOT NULL,
    event_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    {", ".join(f"{name} TEXT" for name in _SCOPE_FIELDS)},
    kind TEXT NOT NULL,
    stage TEXT,
    topic TEXT,
    text TEXT,
    tags_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    severity INTEGER,
    signal REAL,
    schema_version INTEGER NOT NULL,
    UNIQUE(stream, event_id)
)
"""
_CREATE_EVENT_TAGS = """
CREATE TABLE local_event_tags (
    event_cursor INTEGER NOT NULL REFERENCES local_events(cursor) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (event_cursor, tag)
)
"""
_EVENT_INDEXES = (
    "CREATE INDEX ix_local_events_stream_cursor ON local_events(stream, cursor)",
    "CREATE INDEX ix_local_events_scope ON local_events("
    + "stream, "
    + ", ".join(_SCOPE_FIELDS)
    + ", cursor)",
    "CREATE INDEX ix_local_events_kind ON local_events(stream, kind, cursor)",
    "CREATE INDEX ix_local_events_topic ON local_events(stream, topic, cursor)",
    "CREATE INDEX ix_local_events_occurred ON local_events(stream, occurred_at, cursor)",
    "CREATE INDEX ix_local_events_session ON local_events("
    "stream, project_id, org_id, user_id, session_id, cursor)",
    "CREATE INDEX ix_local_events_run ON local_events("
    "stream, project_id, org_id, user_id, run_id, cursor)",
    "CREATE INDEX ix_local_events_agent ON local_events("
    "stream, project_id, org_id, user_id, session_id, agent_id, cursor)",
    "CREATE INDEX ix_local_event_tags_tag ON local_event_tags(tag, event_cursor)",
)


class LocalEventStore:
    """One logical ordered event stream backed by the shared events database."""

    def __init__(self, *, database: LocalSQLiteDatabase, stream: str) -> None:
        if _STREAM_NAME.fullmatch(stream) is None:
            raise StorageConfigurationError("event stream must be a lowercase identifier")
        self._database = database
        self._stream = stream
        self._mode = database.mode
        database.install_component(
            name="events",
            version=_EVENT_COMPONENT_VERSION,
            statements=(_CREATE_EVENTS, _CREATE_EVENT_TAGS, *_EVENT_INDEXES),
        )

    async def append(self, event: EventDraft) -> EventRecord:
        """Commit one canonical event with a monotonic provider cursor.

        Exact retries return the original committed record. Reusing an event ID with
        different immutable content fails as an integrity conflict.

        Examples:
            Append one runtime event:
                ```python
                record = await store.append(event)
                ```

            Retry the same event safely:
                ```python
                assert await store.append(event) == record
                ```

        Args:
            event: Immutable canonical event draft.

        Returns:
            EventRecord: Authoritative event with an opaque monotonic cursor.

        Notes:
            The logical stream is fixed at construction and cannot fall back to
            another event family.
        """
        records = await self.append_many((event,))
        return records[0]

    async def append_many(self, events: tuple[EventDraft, ...]) -> tuple[EventRecord, ...]:
        """Commit one bounded event batch atomically in caller order.

        All inserts and exact idempotent retries share one immediate transaction.
        Any conflicting identity rolls back the complete batch.

        Examples:
            Append an ordered batch:
                ```python
                records = await store.append_many((first, second))
                ```

            Append an empty batch:
                ```python
                assert await store.append_many(()) == ()
                ```

        Args:
            events: Immutable event tuple containing at most 1000 drafts.

        Returns:
            tuple[EventRecord, ...]: Committed records in input order.

        Notes:
            Oversized batches fail directly and are never split into weaker commits.
        """
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError(f"Local event stream {self._stream!r} is read-only")
        if not isinstance(events, tuple):
            raise TypeError("events must be an immutable tuple")
        if len(events) > _MAX_APPEND_BATCH:
            raise StorageConfigurationError("event append batch exceeds 1000 records")
        if not events:
            return ()

        def commit(connection: sqlite3.Connection) -> tuple[EventRecord, ...]:
            return tuple(self._append_sync(connection, event) for event in events)

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, event_id: str) -> EventRecord | None:
        """Read one event ID constrained by populated canonical scope dimensions.

        Every populated scope dimension participates in the lookup. Omitted execution
        dimensions remain wildcards under the caller's required owner boundary.

        Examples:
            Read a committed event:
                ```python
                record = await store.get(scope, "event-1")
                ```

            Detect a missing identity:
                ```python
                assert await store.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner and optional execution constraints.
            event_id: Stable event identifier, never a provider cursor alias.

        Returns:
            EventRecord | None: Matching scoped event or `None`.

        Notes:
            Event identity is stream-unique; populated owner constraints are still
            checked before hydration.
        """
        where, parameters = _scope_predicate(scope)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_events WHERE stream = ? AND event_id = ? AND {where}",
            (self._stream, event_id, *parameters),
        )
        return _record(rows[0]) if rows else None

    async def query(self, query: EventQuery) -> Page[EventRecord]:
        """Read one bounded stable page after applying exact event filters.

        The opaque continuation cursor binds the stream, direction, and complete
        filter set to prevent accidental reuse against another query shape.

        Examples:
            Read recent scoped events:
                ```python
                page = await store.query(EventQuery(scope=scope))
                ```

            Continue the same query:
                ```python
                page = await store.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact scope, filters, ordering, and bounded page request.

        Returns:
            Page[EventRecord]: Stable records and optional opaque continuation cursor.

        Notes:
            Tags use indexed-stream filtering before pagination; no offset or
            unbounded list operation exists.
        """
        where, parameters = _scope_predicate(query.scope)
        clauses = ["stream = ?", where]
        values: list[Any] = [self._stream, *parameters]
        if query.kinds:
            clauses.append(f"kind IN ({', '.join('?' for _ in query.kinds)})")
            values.extend(query.kinds)
        for column in ("stage", "topic"):
            if (value := getattr(query, column)) is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        for tag in query.tags:
            clauses.append(
                "EXISTS (SELECT 1 FROM local_event_tags "
                "WHERE event_cursor = local_events.cursor AND tag = ?)"
            )
            values.append(tag)
        if query.occurred_at_min is not None:
            clauses.append("occurred_at >= ?")
            values.append(query.occurred_at_min.isoformat())
        if query.occurred_at_max is not None:
            clauses.append("occurred_at <= ?")
            values.append(query.occurred_at_max.isoformat())

        fingerprint = _query_fingerprint(query)
        direction = "ASC" if query.order is SortDirection.ASCENDING else "DESC"
        if query.page.cursor is not None:
            anchor = _decode_page_cursor(
                query.page.cursor,
                stream=self._stream,
                direction=direction,
                fingerprint=fingerprint,
            )
            clauses.append(f"cursor {'>' if direction == 'ASC' else '<'} ?")
            values.append(anchor)
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_events WHERE {' AND '.join(clauses)} "
            f"ORDER BY cursor {direction} LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        next_cursor = None
        if len(rows) > query.page.limit:
            next_cursor = _encode_page_cursor(
                stream=self._stream,
                direction=direction,
                fingerprint=fingerprint,
                anchor=int(selected[-1]["cursor"]),
            )
        return Page(items=tuple(_record(row) for row in selected), next_cursor=next_cursor)

    def _append_sync(
        self,
        connection: sqlite3.Connection,
        event: EventDraft,
    ) -> EventRecord:
        existing = connection.execute(
            "SELECT * FROM local_events WHERE stream = ? AND event_id = ?",
            (self._stream, event.event_id),
        ).fetchone()
        if existing is not None:
            record = _record(existing)
            if _draft(record) != event:
                raise StorageIntegrityError(
                    f"Event identity {event.event_id!r} has conflicting content"
                )
            return record
        scope_values = tuple(getattr(event.scope, name) for name in _SCOPE_FIELDS)
        columns = (
            "stream",
            "event_id",
            "occurred_at",
            *_SCOPE_FIELDS,
            "kind",
            "stage",
            "topic",
            "text",
            "tags_json",
            "payload_json",
            "metrics_json",
            "severity",
            "signal",
            "schema_version",
        )
        values = (
            self._stream,
            event.event_id,
            event.occurred_at.isoformat(),
            *scope_values,
            event.kind,
            event.stage,
            event.topic,
            event.text,
            _json(event.tags),
            _json(event.payload),
            _json(event.metrics),
            event.severity,
            event.signal,
            event.schema_version,
        )
        cursor = connection.execute(
            f"INSERT INTO local_events({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})",
            values,
        ).lastrowid
        row = connection.execute(
            "SELECT * FROM local_events WHERE cursor = ?",
            (cursor,),
        ).fetchone()
        connection.executemany(
            "INSERT INTO local_event_tags(event_cursor, tag) VALUES (?, ?)",
            ((cursor, tag) for tag in event.tags),
        )
        return _record(row)


def _scope_predicate(scope: StorageScope) -> tuple[str, tuple[str, ...]]:
    populated = scope.as_filter()
    if not populated:
        return "1 = 1", ()
    return (
        " AND ".join(f"{name} = ?" for name in populated),
        tuple(populated.values()),
    )


def _record(row: sqlite3.Row) -> EventRecord:
    try:
        return EventRecord(
            event_id=str(row["event_id"]),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            scope=StorageScope(**{name: row[name] for name in _SCOPE_FIELDS}),
            kind=str(row["kind"]),
            stage=row["stage"],
            topic=row["topic"],
            text=row["text"],
            tags=tuple(json.loads(row["tags_json"])),
            payload=json.loads(row["payload_json"]),
            metrics=json.loads(row["metrics_json"]),
            severity=row["severity"],
            signal=row["signal"],
            schema_version=int(row["schema_version"]),
            cursor=f"event:{int(row['cursor'])}",
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local event row is malformed") from exc


def _draft(record: EventRecord) -> EventDraft:
    return EventDraft(**{name: getattr(record, name) for name in EventDraft.__dataclass_fields__})


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _query_fingerprint(query: EventQuery) -> str:
    payload = {
        "scope": query.scope.as_filter(),
        "kinds": query.kinds,
        "stage": query.stage,
        "topic": query.topic,
        "tags": query.tags,
        "occurred_at_min": (query.occurred_at_min.isoformat() if query.occurred_at_min else None),
        "occurred_at_max": (query.occurred_at_max.isoformat() if query.occurred_at_max else None),
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()[:24]


def _encode_page_cursor(
    *,
    stream: str,
    direction: str,
    fingerprint: str,
    anchor: int,
) -> str:
    payload = _json(
        {"stream": stream, "direction": direction, "fingerprint": fingerprint, "anchor": anchor}
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_page_cursor(
    cursor: str,
    *,
    stream: str,
    direction: str,
    fingerprint: str,
) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if payload != {
            "stream": stream,
            "direction": direction,
            "fingerprint": fingerprint,
            "anchor": payload.get("anchor"),
        }:
            raise ValueError("cursor context")
        anchor = payload["anchor"]
        if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 1:
            raise ValueError("cursor anchor")
        return anchor
    except (
        AttributeError,
        binascii.Error,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StorageConfigurationError("Invalid or mismatched event page cursor") from exc

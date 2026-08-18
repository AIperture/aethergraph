"""Local inbound, semantic, and runtime-output stream persistence."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import sqlite3
import threading
from typing import Any

from ...contracts import (
    InboundEventDraft,
    InboundEventRecord,
    Page,
    RuntimeOutputFrame,
    RuntimeOutputQuery,
    RuntimeOutputRecord,
    RuntimeOutputStream,
    SemanticEventDraft,
    SemanticEventKind,
    SemanticEventQuery,
    SemanticEventRecord,
    StorageCapacityError,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalDatabaseRole, LocalSQLiteDatabase

_COMPONENT_VERSION = 2
_SCOPE_COLUMNS = tuple(item.name for item in fields(StorageScope))
_CREATE_DELIVERY_CURSOR_ALLOCATOR = """
CREATE TABLE local_delivery_cursor_allocator (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    current_cursor INTEGER NOT NULL CHECK (current_cursor >= 0)
)
"""
_INITIALIZE_DELIVERY_CURSOR_ALLOCATOR = """
INSERT INTO local_delivery_cursor_allocator(singleton, current_cursor) VALUES (1, 0)
"""
_CREATE_INBOUND = f"""
CREATE TABLE local_inbound_events (
    cursor INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    deployment_id TEXT NOT NULL,
    route_id TEXT NOT NULL,
    integration_id TEXT NOT NULL,
    external_event_id TEXT NOT NULL,
    received_at TEXT NOT NULL,
    {", ".join(f"{name} TEXT" for name in _SCOPE_COLUMNS)},
    payload_json TEXT NOT NULL,
    resource_keys_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    content_digest TEXT NOT NULL,
    UNIQUE(deployment_id, integration_id, external_event_id)
)
"""
_CREATE_INBOUND_SESSION_INDEX = """
CREATE INDEX ix_local_inbound_session_received
ON local_inbound_events(session_id, received_at, cursor)
"""
_CREATE_SEMANTIC = f"""
CREATE TABLE local_semantic_events (
    cursor INTEGER PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    deployment_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    authored_sequence INTEGER NOT NULL CHECK (authored_sequence >= 0),
    producer TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    {", ".join(f"{name} TEXT" for name in _SCOPE_COLUMNS)},
    payload_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    UNIQUE(deployment_id, session_id, turn_id, authored_sequence)
)
"""
_CREATE_SEMANTIC_SESSION_INDEX = """
CREATE INDEX ix_local_semantic_session_cursor
ON local_semantic_events(deployment_id, session_id, cursor)
"""
_CREATE_SEMANTIC_KIND_INDEX = """
CREATE INDEX ix_local_semantic_kind_cursor
ON local_semantic_events(deployment_id, session_id, kind, cursor)
"""
_CREATE_SEMANTIC_TURN_INDEX = """
CREATE INDEX ix_local_semantic_turn_cursor
ON local_semantic_events(deployment_id, session_id, turn_id, cursor)
"""
_CREATE_OUTPUT = f"""
CREATE TABLE local_runtime_output (
    cursor INTEGER PRIMARY KEY,
    output_id TEXT NOT NULL UNIQUE,
    execution_id TEXT NOT NULL,
    {", ".join(f"{name} TEXT" for name in _SCOPE_COLUMNS)},
    stream TEXT NOT NULL,
    execution_sequence INTEGER NOT NULL CHECK (execution_sequence > 0),
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    tool_name TEXT,
    partial INTEGER NOT NULL CHECK (partial IN (0, 1)),
    truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
    eof INTEGER NOT NULL CHECK (eof IN (0, 1)),
    tags_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    content_digest TEXT NOT NULL,
    UNIQUE(execution_id, execution_sequence)
)
"""
_CREATE_OUTPUT_EXECUTION_INDEX = """
CREATE INDEX ix_local_runtime_output_execution
ON local_runtime_output(execution_id, execution_sequence)
"""
_CREATE_OUTPUT_RUN_INDEX = """
CREATE INDEX ix_local_runtime_output_run
ON local_runtime_output(run_id, cursor)
"""


class LocalInboundEventRepository:
    """Canonical validated inbound events in the local events database."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def append(self, event: InboundEventDraft) -> InboundEventRecord:
        """Atomically append one normalized inbound event.

        The provider assigns an integer delivery cursor and derives a separate opaque
        record cursor. Exact retries are idempotent; conflicting reuse fails closed.

        Examples:
            Persist ingress evidence:
                ```python
                stored = await repository.append(event)
                ```

            Retain the durable cursor:
                ```python
                cursor = (await repository.append(event)).delivery_cursor
                ```

        Args:
            event: Validated canonical payload and materialized resource keys.

        Returns:
            InboundEventRecord: Authoritative event with delivery and opaque cursors.

        Notes:
            Raw provider payload and Host schema objects are outside this boundary.
        """
        self._require_writable()
        digest = _inbound_digest(event)

        def commit(connection: sqlite3.Connection) -> InboundEventRecord:
            row = connection.execute(
                "SELECT * FROM local_inbound_events WHERE event_id = ?", (event.event_id,)
            ).fetchone()
            if row is not None:
                if str(row["content_digest"]) != digest:
                    raise StorageIntegrityError("Inbound event identity conflicts")
                return _inbound(row)
            external = connection.execute(
                """SELECT 1 FROM local_inbound_events
                WHERE deployment_id = ? AND integration_id = ? AND external_event_id = ?""",
                (event.deployment_id, event.integration_id, event.external_event_id),
            ).fetchone()
            if external is not None:
                raise StorageIntegrityError("Inbound external event identity conflicts")
            cursor = connection.execute(
                f"""INSERT INTO local_inbound_events(
                    event_id, deployment_id, route_id, integration_id, external_event_id,
                    received_at, {", ".join(_SCOPE_COLUMNS)}, payload_json,
                    resource_keys_json, schema_version, content_digest
                ) VALUES ({", ".join("?" for _ in range(6 + len(_SCOPE_COLUMNS) + 4))})""",
                (
                    event.event_id,
                    event.deployment_id,
                    event.route_id,
                    event.integration_id,
                    event.external_event_id,
                    event.received_at.isoformat(),
                    *_scope_values(event.scope),
                    _json(event.payload),
                    _json(event.resource_keys),
                    event.schema_version,
                    digest,
                ),
            ).lastrowid
            return _inbound_from_draft(event, int(cursor))

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, event_id: str) -> InboundEventRecord | None:
        """Read one exact inbound event under canonical scope constraints.

        Event identity and every populated caller dimension are applied in one SQL
        lookup without consulting ingress claims or external-event aliases.

        Examples:
            Read ingress evidence:
                ```python
                event = await repository.get(scope, "ingress-1")
                ```

            Detect absence:
                ```python
                assert await repository.get(scope, "missing") is None
                ```

        Args:
            scope: Populated canonical owner/session scope.
            event_id: Exact stable inbound event identity.

        Returns:
            InboundEventRecord | None: Authorized event or `None`.

        Notes:
            Deprecated App identity is not a lookup dimension.
        """
        _nonempty("event_id", event_id)
        if not scope.as_filter():
            return None
        clauses, values = _scope_filters(scope)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_inbound_events WHERE event_id = ? AND " + " AND ".join(clauses),
            (event_id, *values),
        )
        return _inbound(rows[0]) if rows else None

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local inbound event repository is read-only")


class LocalSemanticEventRepository:
    """Canonical authored semantic events with stable reconnect cursors."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def append(self, event: SemanticEventDraft) -> SemanticEventRecord:
        """Append one semantic event at its single-assignment authored sequence.

        Event identity, deployment/session/turn sequence, integer delivery cursor,
        and opaque record cursor commit together. Reuse fails without renumbering.

        Examples:
            Persist completion:
                ```python
                stored = await repository.append(event)
                ```

            Publish its cursor:
                ```python
                await delivery.publish(
                    cursor=(await repository.append(event)).delivery_cursor,
                )
                ```

        Args:
            event: Closed provider-neutral semantic event.

        Returns:
            SemanticEventRecord: Committed event with delivery and opaque cursors.

        Notes:
            Duplicate identity or sequence raises `StorageIntegrityError` even when
            the attempted content is otherwise identical.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> SemanticEventRecord:
            if connection.execute(
                "SELECT 1 FROM local_semantic_events WHERE event_id = ?", (event.event_id,)
            ).fetchone():
                raise StorageIntegrityError("Semantic event identity conflicts")
            if connection.execute(
                """SELECT 1 FROM local_semantic_events
                WHERE deployment_id = ? AND session_id = ? AND turn_id = ?
                  AND authored_sequence = ?""",
                (
                    event.deployment_id,
                    event.scope.session_id,
                    event.turn_id,
                    event.sequence,
                ),
            ).fetchone():
                raise StorageIntegrityError("Semantic event authored sequence conflicts")
            cursor = _allocate_delivery_cursor(connection)
            connection.execute(
                f"""INSERT INTO local_semantic_events(
                    cursor, event_id, deployment_id, turn_id, authored_sequence, producer,
                    occurred_at, kind, {", ".join(_SCOPE_COLUMNS)}, payload_json,
                    schema_version
                ) VALUES ({", ".join("?" for _ in range(8 + len(_SCOPE_COLUMNS) + 2))})""",
                (
                    cursor,
                    event.event_id,
                    event.deployment_id,
                    event.turn_id,
                    event.sequence,
                    event.producer,
                    event.occurred_at.isoformat(),
                    event.kind.value,
                    *_scope_values(event.scope),
                    _json(event.payload),
                    event.schema_version,
                ),
            )
            return _semantic_from_draft(event, cursor)

        return await self._database.transaction(commit)

    async def query(self, query: SemanticEventQuery) -> Page[SemanticEventRecord]:
        """Read one ascending bounded semantic-event reconnect page.

        Deployment, session scope, delivery cursor, kind, and turn filters execute
        before provider-cursor pagination.

        Examples:
            Read session history:
                ```python
                page = await repository.query(query)
                ```

            Continue reconnect delivery:
                ```python
                page = await repository.query(next_query)
                ```

        Args:
            query: Exact deployment/session filters and opaque page request.

        Returns:
            Page[SemanticEventRecord]: Ascending records and optional next cursor.

        Notes:
            The cursor is bound to all filters and page size.
        """
        clauses, values = _scope_filters(query.scope)
        clauses.append("deployment_id = ?")
        values.append(query.deployment_id)
        if query.after_delivery_cursor is not None:
            clauses.append("cursor > ?")
            values.append(query.after_delivery_cursor)
        if query.kinds:
            clauses.append(f"kind IN ({','.join('?' for _ in query.kinds)})")
            values.extend(kind.value for kind in query.kinds)
        if query.turn_id is not None:
            clauses.append("turn_id = ?")
            values.append(query.turn_id)
        fingerprint = _semantic_fingerprint(query)
        if query.page.cursor:
            clauses.append("cursor > ?")
            values.append(_decode_cursor(query.page.cursor, fingerprint))
        rows = await self._database.fetch_all(
            "SELECT * FROM local_semantic_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY cursor ASC LIMIT ?",
            (*values, query.page.limit + 1),
        )
        visible = rows[: query.page.limit]
        next_cursor = None
        if len(rows) > query.page.limit:
            next_cursor = _encode_cursor(fingerprint, int(visible[-1]["cursor"]))
        return Page(items=tuple(_semantic(row) for row in visible), next_cursor=next_cursor)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local semantic event repository is read-only")


class LocalRuntimeOutputSink:
    """Bounded synchronous admission with transactional async durability barriers."""

    def __init__(
        self,
        *,
        database: LocalSQLiteDatabase,
        max_pending_frames: int = 10_000,
    ) -> None:
        _install(database)
        if isinstance(max_pending_frames, bool) or not 1 <= max_pending_frames <= 1_000_000:
            raise StorageConfigurationError("max_pending_frames must be between 1 and 1000000")
        self._database = database
        self._mode = database.mode
        self._capacity = max_pending_frames
        self._pending: deque[RuntimeOutputFrame] = deque()
        self._by_id: dict[str, RuntimeOutputFrame] = {}
        self._by_sequence: dict[tuple[str, int], RuntimeOutputFrame] = {}
        self._pending_lock = threading.Lock()
        self._flush_lock = asyncio.Lock()

    def emit(self, frame: RuntimeOutputFrame) -> None:
        """Synchronously accept one frame into the bounded provider queue.

        Admission validates pending identity/sequence uniqueness without blocking on
        SQLite. Durability is established by the execution, run, or bundle barrier.

        Examples:
            Emit stdout:
                ```python
                sink.emit(frame)
                ```

            Emit an EOF frame:
                ```python
                sink.emit(replace(frame, eof=True))
                ```

        Args:
            frame: Canonical bounded runtime-output frame.

        Returns:
            None: The frame is queued or was an exact pending retry.

        Notes:
            Full capacity raises `StorageCapacityError`; frames never redirect to a
            file, alternate provider, or unredacted fallback.
        """
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local runtime output sink is read-only")
        with self._pending_lock:
            existing = self._by_id.get(frame.output_id)
            if existing is not None:
                if existing != frame:
                    raise StorageIntegrityError("Runtime output identity conflicts")
                return
            sequence_key = (frame.execution_id, frame.sequence)
            if sequence_key in self._by_sequence:
                raise StorageIntegrityError("Runtime output execution sequence conflicts")
            if len(self._pending) >= self._capacity:
                raise StorageCapacityError("Local runtime output queue is full")
            self._pending.append(frame)
            self._by_id[frame.output_id] = frame
            self._by_sequence[sequence_key] = frame

    async def flush_execution(self, execution_id: str) -> None:
        """Persist frames accepted for one execution before this barrier.

        Matching frames commit atomically in accepted order while other executions
        remain queued and may continue accepting output.

        Examples:
            Flush one tool execution:
                ```python
                await sink.flush_execution("execution-1")
                ```

            Flush captured output:
                ```python
                await sink.flush_execution(execution_id)
                ```

        Args:
            execution_id: Exact stable execution identity.

        Returns:
            None: Previously accepted matching frames are durable.

        Notes:
            Persistence and identity failures propagate; failed frames remain queued.
        """
        _nonempty("execution_id", execution_id)
        await self._flush(lambda frame: frame.execution_id == execution_id)

    async def flush_run(self, run_id: str) -> None:
        """Persist frames accepted for one run before this barrier.

        All matching executions commit atomically in provider admission order while
        unrelated run frames remain queued.

        Examples:
            Flush before publishing a result:
                ```python
                await sink.flush_run("run-1")
                ```

            Flush cancellation output:
                ```python
                await sink.flush_run(canceled_run_id)
                ```

        Args:
            run_id: Exact stable run identity.

        Returns:
            None: Previously accepted run frames are durable.

        Notes:
            Bundle shutdown owns the final all-frame barrier.
        """
        _nonempty("run_id", run_id)
        await self._flush(lambda frame: frame.scope.run_id == run_id)

    async def query(self, query: RuntimeOutputQuery) -> Page[RuntimeOutputRecord]:
        """Read one ascending bounded page of committed runtime output.

        Intro:
            Applies run scope and optional execution/stream filters before pagination.
            Pending frames remain invisible until a durability barrier commits.

        Examples:
            Read durable run output:
                ```python
                page = await sink.query(query)
                ```

            Resume merged delivery:
                ```python
                page = await sink.query(
                    replace(query, after_delivery_cursor=last_cursor)
                )
                ```

        Args:
            query: Exact run scope, delivery boundary, filters, and page request.

        Returns:
            Page[RuntimeOutputRecord]: Committed records and continuation cursor.

        Notes:
            The cursor belongs to the same provider-wide domain as semantic events;
            this method never reads pending memory or a legacy EventLog.
        """
        clauses, values = _scope_filters(query.scope)
        if query.after_delivery_cursor is not None:
            clauses.append("cursor > ?")
            values.append(query.after_delivery_cursor)
        if query.streams:
            clauses.append(f"stream IN ({','.join('?' for _ in query.streams)})")
            values.extend(stream.value for stream in query.streams)
        if query.execution_id is not None:
            clauses.append("execution_id = ?")
            values.append(query.execution_id)
        fingerprint = _runtime_output_fingerprint(query)
        if query.page.cursor:
            clauses.append("cursor > ?")
            values.append(_decode_cursor(query.page.cursor, fingerprint))
        rows = await self._database.fetch_all(
            "SELECT * FROM local_runtime_output WHERE "
            + " AND ".join(clauses)
            + " ORDER BY cursor ASC LIMIT ?",
            (*values, query.page.limit + 1),
        )
        visible = rows[: query.page.limit]
        next_cursor = None
        if len(rows) > query.page.limit:
            next_cursor = _encode_cursor(fingerprint, int(visible[-1]["cursor"]))
        return Page(items=tuple(_runtime_output(row) for row in visible), next_cursor=next_cursor)

    async def _flush_all(self) -> None:
        await self._flush(lambda _frame: True)

    async def _flush(self, selected: Callable[[RuntimeOutputFrame], bool]) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            return
        async with self._flush_lock:
            with self._pending_lock:
                frames = tuple(frame for frame in self._pending if selected(frame))
            if not frames:
                return

            def commit(connection: sqlite3.Connection) -> None:
                for frame in frames:
                    _append_output(connection, frame)

            await self._database.transaction(commit)
            removed = {id(frame) for frame in frames}
            with self._pending_lock:
                self._pending = deque(frame for frame in self._pending if id(frame) not in removed)
                self._by_id = {frame.output_id: frame for frame in self._pending}
                self._by_sequence = {
                    (frame.execution_id, frame.sequence): frame for frame in self._pending
                }


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.EVENTS:
        raise StorageConfigurationError("Local stream repositories require events database")
    database.install_component(
        name="integration_streams",
        version=_COMPONENT_VERSION,
        statements=(
            _CREATE_DELIVERY_CURSOR_ALLOCATOR,
            _INITIALIZE_DELIVERY_CURSOR_ALLOCATOR,
            _CREATE_INBOUND,
            _CREATE_INBOUND_SESSION_INDEX,
            _CREATE_SEMANTIC,
            _CREATE_SEMANTIC_SESSION_INDEX,
            _CREATE_SEMANTIC_KIND_INDEX,
            _CREATE_SEMANTIC_TURN_INDEX,
            _CREATE_OUTPUT,
            _CREATE_OUTPUT_EXECUTION_INDEX,
            _CREATE_OUTPUT_RUN_INDEX,
        ),
    )


def _inbound_from_draft(event: InboundEventDraft, cursor: int) -> InboundEventRecord:
    return InboundEventRecord(
        event_id=event.event_id,
        deployment_id=event.deployment_id,
        route_id=event.route_id,
        integration_id=event.integration_id,
        external_event_id=event.external_event_id,
        received_at=event.received_at,
        scope=event.scope,
        delivery_cursor=cursor,
        cursor=_record_cursor("inbound", cursor),
        payload=event.payload,
        resource_keys=event.resource_keys,
        schema_version=event.schema_version,
    )


def _inbound(row: sqlite3.Row) -> InboundEventRecord:
    try:
        resources = _json_array(row["resource_keys_json"])
        if any(not isinstance(value, str) for value in resources):
            raise TypeError("resource keys must be strings")
        return InboundEventRecord(
            event_id=str(row["event_id"]),
            deployment_id=str(row["deployment_id"]),
            route_id=str(row["route_id"]),
            integration_id=str(row["integration_id"]),
            external_event_id=str(row["external_event_id"]),
            received_at=datetime.fromisoformat(str(row["received_at"])),
            scope=_scope(row),
            delivery_cursor=int(row["cursor"]),
            cursor=_record_cursor("inbound", int(row["cursor"])),
            payload=_json_object(row["payload_json"]),
            resource_keys=tuple(resources),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted inbound event is malformed") from exc


def _semantic_from_draft(event: SemanticEventDraft, cursor: int) -> SemanticEventRecord:
    return SemanticEventRecord(
        event_id=event.event_id,
        deployment_id=event.deployment_id,
        turn_id=event.turn_id,
        sequence=event.sequence,
        producer=event.producer,
        occurred_at=event.occurred_at,
        kind=event.kind,
        scope=event.scope,
        delivery_cursor=cursor,
        cursor=_record_cursor("semantic", cursor),
        payload=event.payload,
        schema_version=event.schema_version,
    )


def _semantic(row: sqlite3.Row) -> SemanticEventRecord:
    try:
        return SemanticEventRecord(
            event_id=str(row["event_id"]),
            deployment_id=str(row["deployment_id"]),
            turn_id=str(row["turn_id"]),
            sequence=int(row["authored_sequence"]),
            producer=str(row["producer"]),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            kind=SemanticEventKind(str(row["kind"])),
            scope=_scope(row),
            delivery_cursor=int(row["cursor"]),
            cursor=_record_cursor("semantic", int(row["cursor"])),
            payload=_json_object(row["payload_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted semantic event is malformed") from exc


def _append_output(connection: sqlite3.Connection, frame: RuntimeOutputFrame) -> None:
    digest = _output_digest(frame)
    existing = connection.execute(
        "SELECT * FROM local_runtime_output WHERE output_id = ?", (frame.output_id,)
    ).fetchone()
    if existing is not None:
        if str(existing["content_digest"]) != digest:
            raise StorageIntegrityError("Runtime output identity conflicts")
        return
    sequence = connection.execute(
        """SELECT output_id FROM local_runtime_output
        WHERE execution_id = ? AND execution_sequence = ?""",
        (frame.execution_id, frame.sequence),
    ).fetchone()
    if sequence is not None:
        raise StorageIntegrityError("Runtime output execution sequence conflicts")
    cursor = _allocate_delivery_cursor(connection)
    connection.execute(
        f"""INSERT INTO local_runtime_output(
            cursor, output_id, execution_id, {", ".join(_SCOPE_COLUMNS)}, stream,
            execution_sequence, text, source, tool_name, partial, truncated, eof,
            tags_json, schema_version, content_digest
        ) VALUES ({", ".join("?" for _ in range(3 + len(_SCOPE_COLUMNS) + 11))})""",
        (
            cursor,
            frame.output_id,
            frame.execution_id,
            *_scope_values(frame.scope),
            frame.stream.value,
            frame.sequence,
            frame.text,
            frame.source,
            frame.tool_name,
            int(frame.partial),
            int(frame.truncated),
            int(frame.eof),
            _json(frame.tags),
            frame.schema_version,
            digest,
        ),
    )


def _allocate_delivery_cursor(connection: sqlite3.Connection) -> int:
    row = connection.execute(
        "SELECT current_cursor FROM local_delivery_cursor_allocator WHERE singleton = 1"
    ).fetchone()
    if row is None:
        raise StorageIntegrityError("Shared delivery cursor allocator is missing")
    cursor = int(row[0]) + 1
    connection.execute(
        "UPDATE local_delivery_cursor_allocator SET current_cursor = ? WHERE singleton = 1",
        (cursor,),
    )
    return cursor


def _runtime_output(row: sqlite3.Row) -> RuntimeOutputRecord:
    try:
        tags = _json_array(row["tags_json"])
        if any(not isinstance(value, str) for value in tags):
            raise TypeError("runtime-output tags must be strings")
        return RuntimeOutputRecord(
            output_id=str(row["output_id"]),
            execution_id=str(row["execution_id"]),
            scope=_scope(row),
            stream=RuntimeOutputStream(str(row["stream"])),
            sequence=int(row["execution_sequence"]),
            text=str(row["text"]),
            source=str(row["source"]),
            delivery_cursor=int(row["cursor"]),
            cursor=_record_cursor("runtime-output", int(row["cursor"])),
            tool_name=row["tool_name"],
            partial=bool(row["partial"]),
            truncated=bool(row["truncated"]),
            eof=bool(row["eof"]),
            tags=tuple(tags),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted runtime output is malformed") from exc


def _scope_values(scope: StorageScope) -> tuple[str | None, ...]:
    return tuple(getattr(scope, name) for name in _SCOPE_COLUMNS)


def _scope(row: sqlite3.Row) -> StorageScope:
    try:
        return StorageScope(**{name: row[name] for name in _SCOPE_COLUMNS})
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted stream scope is malformed") from exc


def _scope_filters(scope: StorageScope) -> tuple[list[str], list[object]]:
    clauses = [f"{name} = ?" for name in scope.as_filter()]
    if not clauses:
        raise StorageConfigurationError("Stream operations require populated scope")
    return clauses, list(scope.as_filter().values())


def _inbound_digest(event: InboundEventDraft) -> str:
    return _digest(
        {
            "event_id": event.event_id,
            "deployment_id": event.deployment_id,
            "route_id": event.route_id,
            "integration_id": event.integration_id,
            "external_event_id": event.external_event_id,
            "received_at": event.received_at.isoformat(),
            "scope": event.scope.as_filter(),
            "payload": event.payload,
            "resource_keys": event.resource_keys,
            "schema_version": event.schema_version,
        }
    )


def _output_digest(frame: RuntimeOutputFrame) -> str:
    return _digest(
        {
            "output_id": frame.output_id,
            "execution_id": frame.execution_id,
            "scope": frame.scope.as_filter(),
            "stream": frame.stream.value,
            "sequence": frame.sequence,
            "text": frame.text,
            "source": frame.source,
            "tool_name": frame.tool_name,
            "partial": frame.partial,
            "truncated": frame.truncated,
            "eof": frame.eof,
            "tags": frame.tags,
            "schema_version": frame.schema_version,
        }
    )


def _semantic_fingerprint(query: SemanticEventQuery) -> str:
    return _digest(
        {
            "kind": "semantic",
            "deployment_id": query.deployment_id,
            "scope": query.scope.as_filter(),
            "after_delivery_cursor": query.after_delivery_cursor,
            "kinds": tuple(kind.value for kind in query.kinds),
            "turn_id": query.turn_id,
            "limit": query.page.limit,
        }
    )[:24]


def _runtime_output_fingerprint(query: RuntimeOutputQuery) -> str:
    return _digest(
        {
            "kind": "runtime-output",
            "scope": query.scope.as_filter(),
            "after_delivery_cursor": query.after_delivery_cursor,
            "streams": tuple(stream.value for stream in query.streams),
            "execution_id": query.execution_id,
            "limit": query.page.limit,
        }
    )[:24]


def _record_cursor(kind: str, cursor: int) -> str:
    return base64.urlsafe_b64encode(f"{kind}:{cursor}".encode()).decode().rstrip("=")


def _encode_cursor(fingerprint: str, cursor: int) -> str:
    return (
        base64.urlsafe_b64encode(_json({"fingerprint": fingerprint, "cursor": cursor}).encode())
        .decode()
        .rstrip("=")
    )


def _decode_cursor(cursor: str, fingerprint: str) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if (
            not isinstance(payload, dict)
            or set(payload) != {"fingerprint", "cursor"}
            or payload["fingerprint"] != fingerprint
            or isinstance(payload["cursor"], bool)
            or not isinstance(payload["cursor"], int)
            or payload["cursor"] < 1
        ):
            raise ValueError("cursor payload")
        return payload["cursor"]
    except (binascii.Error, ValueError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageConfigurationError("Invalid or mismatched semantic cursor") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_object(value: object) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("persisted JSON value must be an object")
    return parsed


def _json_array(value: object) -> list[Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise TypeError("persisted JSON value must be an array")
    return parsed


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StorageConfigurationError(f"{name} must be a non-empty string")

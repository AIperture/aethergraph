"""Indexed current state and revision history for the local SQLite provider."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
import sqlite3
from typing import Any

from ...contracts import (
    FrozenJson,
    Page,
    SortDirection,
    StateHistoryQuery,
    StateRecord,
    StorageClock,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalSQLiteDatabase

_STATE_COMPONENT_VERSION = 1
_MAX_GET_MANY = 1_000
_CREATE_CURRENT_STATE = """
CREATE TABLE local_state_current (
    scope_identity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    state_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    PRIMARY KEY (scope_identity, namespace, state_key)
)
"""
_CREATE_STATE_HISTORY = """
CREATE TABLE local_state_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_identity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    state_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL
)
"""
_CREATE_STATE_HISTORY_INDEX = """
CREATE INDEX ix_local_state_history_identity
ON local_state_history(scope_identity, namespace, state_key, history_id)
"""
_CREATE_STATE_OUTBOX = """
CREATE TABLE local_state_outbox (
    outbox_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_identity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    state_key TEXT NOT NULL,
    revision INTEGER NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('updated', 'deleted')),
    recorded_at TEXT NOT NULL,
    payload_json TEXT
)
"""
_CREATE_STATE_OUTBOX_INDEX = """
CREATE INDEX ix_local_state_outbox_order ON local_state_outbox(outbox_id)
"""


class LocalStateStore:
    """Transactional current-state repository with retained revision history."""

    def __init__(self, *, database: LocalSQLiteDatabase, clock: StorageClock) -> None:
        self._database = database
        self._clock = clock
        self._mode = database.mode
        database.install_component(
            name="state",
            version=_STATE_COMPONENT_VERSION,
            statements=(
                _CREATE_CURRENT_STATE,
                _CREATE_STATE_HISTORY,
                _CREATE_STATE_HISTORY_INDEX,
                _CREATE_STATE_OUTBOX,
                _CREATE_STATE_OUTBOX_INDEX,
            ),
        )

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
    ) -> StateRecord | None:
        """Read one indexed current state record by exact identity.

        The lookup addresses the canonical scope, namespace, and key directly and
        never reconstructs current state by scanning events or history.

        Examples:
            Read current Agent state:
                ```python
                state = await store.get(scope, "agent", "writer")
                ```

            Detect an absent key:
                ```python
                assert await store.get(scope, "graph", "missing") is None
                ```

        Args:
            scope: Exact canonical owner/execution scope.
            namespace: Exact service-owned state namespace.
            key: Exact state key within the namespace.

        Returns:
            StateRecord | None: Current record or `None` when absent.

        Notes:
            `app_id` and legacy memory tags are not state identity dimensions.
        """
        _validate_identity(namespace, key)
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_state_current
            WHERE scope_identity = ? AND namespace = ? AND state_key = ?
            """,
            (_scope_identity(scope), namespace, key),
        )
        return _record(rows[0], scope) if rows else None

    async def get_many(
        self,
        scope: StorageScope,
        namespace: str,
        keys: tuple[str, ...],
    ) -> tuple[StateRecord | None, ...]:
        """Hydrate a bounded key tuple with one indexed database query.

        Results preserve caller order and duplicate positions while missing keys
        produce explicit `None` placeholders.

        Examples:
            Hydrate graph node state:
                ```python
                rows = await store.get_many(scope, "graph", ("a", "b"))
                ```

            Hydrate no keys:
                ```python
                assert await store.get_many(scope, "graph", ()) == ()
                ```

        Args:
            scope: Exact canonical owner/execution scope.
            namespace: Exact shared namespace for all requested keys.
            keys: Immutable tuple containing at most 1000 exact keys.

        Returns:
            tuple[StateRecord | None, ...]: Records or missing slots in key order.

        Notes:
            The implementation opens no additional connections and performs no
            per-key query loop.
        """
        if not isinstance(keys, tuple):
            raise TypeError("keys must be an immutable tuple")
        if len(keys) > _MAX_GET_MANY:
            raise StorageConfigurationError("state get_many exceeds 1000 keys")
        _validate_identity(namespace, *keys)
        if not keys:
            return ()
        unique_keys = tuple(dict.fromkeys(keys))
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_state_current
            WHERE scope_identity = ? AND namespace = ?
            AND state_key IN ("""
            + ", ".join("?" for _ in unique_keys)
            + ")",
            (_scope_identity(scope), namespace, *unique_keys),
        )
        records = {str(row["state_key"]): _record(row, scope) for row in rows}
        return tuple(records.get(key) for key in keys)

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
        value: FrozenJson,
        metadata: Mapping[str, FrozenJson],
    ) -> StateRecord:
        """Commit the next current revision and history row atomically.

        `expected_revision=0` creates only an absent identity. Every other value must
        match the indexed current row before the next revision is constructed.

        Examples:
            Create initial state:
                ```python
                row = await store.compare_and_set(scope, "agent", "writer", 0, {}, {})
                ```

            Advance an existing revision:
                ```python
                row = await store.compare_and_set(scope, "agent", "writer", 1, value, meta)
                ```

        Args:
            scope: Exact canonical owner/execution scope.
            namespace: Exact service-owned state namespace.
            key: Exact state key.
            expected_revision: Required current revision, or zero for create.
            value: Complete JSON-compatible next state value.
            metadata: JSON-compatible immutable audit metadata.

        Returns:
            StateRecord: Newly committed state at `expected_revision + 1`.

        Notes:
            Stale expectations raise `StorageConflictError`; no read-then-write
            fallback occurs outside the transaction.
        """
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local state store is read-only")
        _validate_identity(namespace, key)
        if isinstance(expected_revision, bool) or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        identity = _scope_identity(scope)
        updated_at = _now(self._clock)
        candidate = StateRecord(
            namespace=namespace,
            key=key,
            value=value,
            revision=expected_revision + 1,
            scope=scope,
            updated_at=updated_at,
            metadata=metadata,
        )
        encoded_value = _json(candidate.value)
        encoded_metadata = _json(candidate.metadata)

        def commit(connection: sqlite3.Connection) -> StateRecord:
            current = connection.execute(
                """
                SELECT revision FROM local_state_current
                WHERE scope_identity = ? AND namespace = ? AND state_key = ?
                """,
                (identity, namespace, key),
            ).fetchone()
            actual = int(current[0]) if current is not None else 0
            if actual != expected_revision:
                raise StorageConflictError(
                    f"State expected revision {expected_revision}, found {actual}"
                )
            values = (
                encoded_value,
                candidate.revision,
                candidate.updated_at.isoformat(),
                encoded_metadata,
                candidate.schema_version,
                identity,
                namespace,
                key,
            )
            if current is None:
                connection.execute(
                    """
                    INSERT INTO local_state_current(
                        value_json, revision, updated_at, metadata_json, schema_version,
                        scope_identity, namespace, state_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
            else:
                connection.execute(
                    """
                    UPDATE local_state_current
                    SET value_json = ?, revision = ?, updated_at = ?, metadata_json = ?,
                        schema_version = ?
                    WHERE scope_identity = ? AND namespace = ? AND state_key = ?
                    """,
                    values,
                )
            connection.execute(
                """
                INSERT INTO local_state_history(
                    value_json, revision, updated_at, metadata_json, schema_version,
                    scope_identity, namespace, state_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            connection.execute(
                """
                INSERT INTO local_state_outbox(
                    scope_identity, namespace, state_key, revision, operation,
                    recorded_at, payload_json
                ) VALUES (?, ?, ?, ?, 'updated', ?, ?)
                """,
                (
                    identity,
                    namespace,
                    key,
                    candidate.revision,
                    candidate.updated_at.isoformat(),
                    _json({"value": candidate.value, "metadata": candidate.metadata}),
                ),
            )
            return candidate

        return await self._database.transaction(commit)

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
    ) -> bool:
        """Delete current state only at the exact optimistic revision.

        Retained history remains immutable. An absent identity succeeds only for the
        explicit zero expectation.

        Examples:
            Delete current state:
                ```python
                deleted = await store.delete(scope, "agent", "writer", 3)
                ```

            Delete an absent identity idempotently:
                ```python
                assert await store.delete(scope, "agent", "missing", 0) is False
                ```

        Args:
            scope: Exact canonical owner/execution scope.
            namespace: Exact service-owned namespace.
            key: Exact state key.
            expected_revision: Revision that must still be current.

        Returns:
            bool: `True` when current state was deleted; `False` for absent revision zero.

        Notes:
            A stale or nonzero absent expectation raises `StorageConflictError`.
        """
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local state store is read-only")
        _validate_identity(namespace, key)
        if isinstance(expected_revision, bool) or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        identity = _scope_identity(scope)
        deleted_at = _now(self._clock)

        def commit(connection: sqlite3.Connection) -> bool:
            current = connection.execute(
                """
                SELECT revision FROM local_state_current
                WHERE scope_identity = ? AND namespace = ? AND state_key = ?
                """,
                (identity, namespace, key),
            ).fetchone()
            if current is None:
                if expected_revision == 0:
                    return False
                raise StorageConflictError("State is absent")
            if int(current[0]) != expected_revision:
                raise StorageConflictError("State revision changed")
            connection.execute(
                """
                DELETE FROM local_state_current
                WHERE scope_identity = ? AND namespace = ? AND state_key = ?
                """,
                (identity, namespace, key),
            )
            connection.execute(
                """
                INSERT INTO local_state_outbox(
                    scope_identity, namespace, state_key, revision, operation,
                    recorded_at, payload_json
                ) VALUES (?, ?, ?, ?, 'deleted', ?, NULL)
                """,
                (identity, namespace, key, expected_revision, deleted_at.isoformat()),
            )
            return True

        return await self._database.transaction(commit)

    async def history(self, query: StateHistoryQuery) -> Page[StateRecord]:
        """Read one bounded stable page of retained state revisions.

        The opaque cursor binds exact state identity and order. History ordering uses
        the provider sequence so delete-and-recreate generations remain unambiguous.

        Examples:
            Read recent revisions:
                ```python
                page = await store.history(StateHistoryQuery(scope=scope, namespace="agent", key="writer"))
                ```

            Continue the same history query:
                ```python
                page = await store.history(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact identity, ordering, and bounded page request.

        Returns:
            Page[StateRecord]: Retained revisions and optional opaque continuation cursor.

        Notes:
            Deleting current state does not erase audit history.
        """
        _validate_identity(query.namespace, query.key)
        identity = _scope_identity(query.scope)
        direction = "ASC" if query.order is SortDirection.ASCENDING else "DESC"
        fingerprint = _history_fingerprint(identity, query.namespace, query.key)
        clauses = ["scope_identity = ?", "namespace = ?", "state_key = ?"]
        values: list[Any] = [identity, query.namespace, query.key]
        if query.page.cursor is not None:
            anchor = _decode_cursor(query.page.cursor, direction, fingerprint)
            clauses.append(f"history_id {'>' if direction == 'ASC' else '<'} ?")
            values.append(anchor)
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_state_history WHERE {' AND '.join(clauses)} "
            f"ORDER BY history_id {direction} LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        next_cursor = None
        if len(rows) > query.page.limit:
            next_cursor = _encode_cursor(
                direction,
                fingerprint,
                int(selected[-1]["history_id"]),
            )
        return Page(
            items=tuple(_record(row, query.scope) for row in selected),
            next_cursor=next_cursor,
        )


def _validate_identity(namespace: str, *keys: str) -> None:
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    if any(not isinstance(key, str) or not key.strip() for key in keys):
        raise ValueError("state keys must be non-empty strings")


def _now(clock: StorageClock) -> datetime:
    value = clock.now()
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise StorageIntegrityError("Local state clock must return a UTC datetime")
    return value


def _scope_identity(scope: StorageScope) -> str:
    return json.dumps(scope.as_filter(), sort_keys=True, separators=(",", ":"))


def _record(row: sqlite3.Row, scope: StorageScope) -> StateRecord:
    try:
        return StateRecord(
            namespace=str(row["namespace"]),
            key=str(row["state_key"]),
            value=json.loads(row["value_json"]),
            revision=int(row["revision"]),
            scope=scope,
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            metadata=json.loads(row["metadata_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local state row is malformed") from exc


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _history_fingerprint(identity: str, namespace: str, key: str) -> str:
    raw = "\x00".join((identity, namespace, key)).encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _encode_cursor(direction: str, fingerprint: str, anchor: int) -> str:
    payload = json.dumps(
        {"direction": direction, "fingerprint": fingerprint, "anchor": anchor},
        sort_keys=True,
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, direction: str, fingerprint: str) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, dict):
            raise ValueError("cursor payload")
        if payload.get("direction") != direction or payload.get("fingerprint") != fingerprint:
            raise ValueError("cursor context")
        if set(payload) != {"direction", "fingerprint", "anchor"}:
            raise ValueError("cursor fields")
        anchor = payload["anchor"]
        if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 1:
            raise ValueError("cursor anchor")
        return anchor
    except (
        binascii.Error,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StorageConfigurationError("Invalid or mismatched state history cursor") from exc

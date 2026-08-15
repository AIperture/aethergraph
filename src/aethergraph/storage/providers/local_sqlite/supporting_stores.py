"""Revisioned local key-value and document supporting stores."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import sqlite3

from ...contracts import (
    DocumentQuery,
    DocumentRecord,
    FrozenJson,
    KeyValueQuery,
    KeyValueRecord,
    Page,
    StorageClock,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalDatabaseRole, LocalSQLiteDatabase

_SUPPORTING_COMPONENT_VERSION = 1
_MAX_METADATA_FILTERS = 100
_CREATE_KV = """
CREATE TABLE local_key_values (
    scope_identity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    expires_at TEXT,
    PRIMARY KEY(scope_identity, namespace, key)
)
"""
_CREATE_KV_EXPIRY_INDEX = """
CREATE INDEX ix_local_key_values_expiry
ON local_key_values(namespace, expires_at, scope_identity, key)
"""
_CREATE_DOCUMENTS = """
CREATE TABLE local_documents (
    scope_identity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_json TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    PRIMARY KEY(scope_identity, namespace, document_id)
)
"""
_CREATE_DOCUMENT_METADATA = """
CREATE TABLE local_document_metadata (
    scope_identity TEXT NOT NULL,
    namespace TEXT NOT NULL,
    document_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY(scope_identity, namespace, document_id, key),
    FOREIGN KEY(scope_identity, namespace, document_id)
        REFERENCES local_documents(scope_identity, namespace, document_id)
        ON DELETE CASCADE
)
"""
_CREATE_DOCUMENT_METADATA_INDEX = """
CREATE INDEX ix_local_document_metadata_value
ON local_document_metadata(scope_identity, namespace, key, value_json, document_id)
"""


class LocalKeyValueStore:
    """Scoped revision-CAS JSON values with exact TTL visibility."""

    def __init__(self, *, database: LocalSQLiteDatabase, clock: StorageClock) -> None:
        _install(database)
        self._database = database
        self._clock = clock
        self._mode = database.mode

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
    ) -> KeyValueRecord | None:
        """Read one exact current unexpired key-value record.

        Expiration is evaluated against the provider clock without mutating storage.

        Examples:
            Read one grant:
                ```python
                record = await store.get(scope, "auth.grants", grant_id)
                ```

            Detect absence:
                ```python
                assert await store.get(scope, "runtime", "missing") is None
                ```

        Args:
            scope: Exact canonical owner scope.
            namespace: Exact logical namespace.
            key: Exact key within the namespace.

        Returns:
            KeyValueRecord | None: Current unexpired value or `None`.

        Notes:
            Expired rows remain hidden until explicit mutation or maintenance.
        """
        _identity(namespace, key)
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_key_values
            WHERE scope_identity = ? AND namespace = ? AND key = ?
              AND (expires_at IS NULL OR expires_at > ?)
            """,
            (_scope_identity(scope), namespace, key, _now(self._clock).isoformat()),
        )
        return _key_value(rows[0]) if rows else None

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
        value: FrozenJson,
        expires_at: datetime | None = None,
    ) -> KeyValueRecord:
        """Atomically create or advance one exact key-value revision.

        Expired rows behave as absent and may be recreated only with expected
        revision zero.

        Examples:
            Create an invite:
                ```python
                row = await store.compare_and_set(scope, "auth.invites", key, 0, value)
                ```

            Renew a grant:
                ```python
                row = await store.compare_and_set(scope, "auth.grants", key, 1, value, expiry)
                ```

        Args:
            scope: Exact canonical owner scope.
            namespace: Exact logical namespace.
            key: Exact key within the namespace.
            expected_revision: Required current revision, or zero for absence.
            value: Complete JSON-compatible next value.
            expires_at: Optional exact UTC expiration.

        Returns:
            KeyValueRecord: Newly committed next revision.

        Notes:
            Stale writes raise `StorageConflictError`; no unconditional set exists.
        """
        self._require_writable()
        _identity(namespace, key)
        _expected_revision(expected_revision)
        now = _now(self._clock)
        record = KeyValueRecord(
            namespace=namespace,
            key=key,
            value=value,
            revision=expected_revision + 1,
            scope=scope,
            updated_at=now,
            expires_at=expires_at,
        )

        def commit(connection: sqlite3.Connection) -> KeyValueRecord:
            existing = connection.execute(
                """
                SELECT revision, expires_at FROM local_key_values
                WHERE scope_identity = ? AND namespace = ? AND key = ?
                """,
                (_scope_identity(scope), namespace, key),
            ).fetchone()
            current_revision = _visible_revision(existing, now)
            if current_revision != expected_revision:
                raise StorageConflictError(
                    f"KV revision conflict: expected {expected_revision}, found {current_revision}"
                )
            connection.execute(
                """
                INSERT INTO local_key_values(
                    scope_identity, namespace, key, value_json, revision,
                    updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_identity, namespace, key) DO UPDATE SET
                    value_json = excluded.value_json,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    _scope_identity(scope),
                    namespace,
                    key,
                    _json(record.value),
                    record.revision,
                    record.updated_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                ),
            )
            return record

        return await self._database.transaction(commit)

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
    ) -> bool:
        """Delete one exact visible value at its current revision.

        Absence, including expiration, is idempotent only for expected revision zero.

        Examples:
            Delete a consumed invite:
                ```python
                deleted = await store.delete(scope, "auth.invites", key, revision)
                ```

            Confirm absence:
                ```python
                assert await store.delete(scope, "runtime", "missing", 0) is False
                ```

        Args:
            scope: Exact canonical owner scope.
            namespace: Exact logical namespace.
            key: Exact key within the namespace.
            expected_revision: Required visible revision.

        Returns:
            bool: `True` when deleted; `False` for absence at revision zero.

        Notes:
            A mismatched visible revision raises `StorageConflictError`.
        """
        self._require_writable()
        _identity(namespace, key)
        _expected_revision(expected_revision)
        now = _now(self._clock)

        def commit(connection: sqlite3.Connection) -> bool:
            existing = connection.execute(
                """
                SELECT revision, expires_at FROM local_key_values
                WHERE scope_identity = ? AND namespace = ? AND key = ?
                """,
                (_scope_identity(scope), namespace, key),
            ).fetchone()
            current_revision = _visible_revision(existing, now)
            if current_revision == 0 and expected_revision == 0:
                return False
            if current_revision != expected_revision:
                raise StorageConflictError(
                    f"KV revision conflict: expected {expected_revision}, found {current_revision}"
                )
            connection.execute(
                """
                DELETE FROM local_key_values
                WHERE scope_identity = ? AND namespace = ? AND key = ?
                """,
                (_scope_identity(scope), namespace, key),
            )
            return True

        return await self._database.transaction(commit)

    async def scan(self, query: KeyValueQuery) -> Page[KeyValueRecord]:
        """Scan one bounded stable key-ordered namespace page.

        Exact scope, namespace, prefix, expiration, and cursor apply before the bound.

        Examples:
            Scan grants:
                ```python
                page = await store.scan(KeyValueQuery(scope=scope, namespace="auth.grants"))
                ```

            Continue a prefix scan:
                ```python
                page = await store.scan(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact scope, namespace, optional prefix, and page request.

        Returns:
            Page[KeyValueRecord]: Current records and opaque continuation cursor.

        Notes:
            Key ordering stays stable when values are revised concurrently.
        """
        fingerprint = _fingerprint(
            "kv", _scope_identity(query.scope), query.namespace, query.key_prefix or ""
        )
        clauses = [
            "scope_identity = ?",
            "namespace = ?",
            "(expires_at IS NULL OR expires_at > ?)",
        ]
        values: list[object] = [
            _scope_identity(query.scope),
            query.namespace,
            _now(self._clock).isoformat(),
        ]
        _prefix_filter(clauses, values, "key", query.key_prefix)
        if query.page.cursor is not None:
            clauses.append("key > ?")
            values.append(_decode_cursor(query.page.cursor, fingerprint))
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_key_values WHERE {' AND '.join(clauses)} "
            "ORDER BY key ASC LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        return Page(
            items=tuple(_key_value(row) for row in selected),
            next_cursor=(
                _encode_cursor(fingerprint, str(selected[-1]["key"]))
                if len(rows) > query.page.limit
                else None
            ),
        )

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local key-value store is read-only")


class LocalDocumentStore:
    """Scoped revision-CAS JSON documents with normalized exact metadata."""

    def __init__(self, *, database: LocalSQLiteDatabase, clock: StorageClock) -> None:
        _install(database)
        self._database = database
        self._clock = clock
        self._mode = database.mode

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        document_id: str,
    ) -> DocumentRecord | None:
        """Read one exact current canonical document.

        The compound primary key provides scope authorization and identity lookup.

        Examples:
            Read a registry manifest:
                ```python
                record = await store.get(scope, "registry", manifest_id)
                ```

            Detect absence:
                ```python
                assert await store.get(scope, "registry", "missing") is None
                ```

        Args:
            scope: Exact canonical owner scope.
            namespace: Exact logical namespace.
            document_id: Exact stable document identity.

        Returns:
            DocumentRecord | None: Current document or `None`.

        Notes:
            Provider rows and physical paths never cross this boundary.
        """
        _identity(namespace, document_id)
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_documents
            WHERE scope_identity = ? AND namespace = ? AND document_id = ?
            """,
            (_scope_identity(scope), namespace, document_id),
        )
        return _document(rows[0]) if rows else None

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        document_id: str,
        expected_revision: int,
        document: Mapping[str, FrozenJson],
        schema_version: int,
    ) -> DocumentRecord:
        """Atomically create or advance one document revision.

        The document and its normalized top-level exact metadata projection commit in
        one transaction.

        Examples:
            Create a manifest:
                ```python
                row = await store.compare_and_set(scope, "registry", key, 0, document, 1)
                ```

            Advance a manifest:
                ```python
                row = await store.compare_and_set(scope, "registry", key, 2, document, 1)
                ```

        Args:
            scope: Exact canonical owner scope.
            namespace: Exact logical namespace.
            document_id: Exact stable document identity.
            expected_revision: Required current revision, or zero for creation.
            document: Complete JSON-compatible document mapping.
            schema_version: Positive owning-record schema version.

        Returns:
            DocumentRecord: Newly committed next revision.

        Notes:
            Stale writes raise `StorageConflictError`; no unconditional upsert exists.
        """
        self._require_writable()
        _identity(namespace, document_id)
        _expected_revision(expected_revision)
        record = DocumentRecord(
            namespace=namespace,
            document_id=document_id,
            document=document,
            revision=expected_revision + 1,
            scope=scope,
            updated_at=_now(self._clock),
            schema_version=schema_version,
        )

        def commit(connection: sqlite3.Connection) -> DocumentRecord:
            existing = connection.execute(
                """
                SELECT revision FROM local_documents
                WHERE scope_identity = ? AND namespace = ? AND document_id = ?
                """,
                (_scope_identity(scope), namespace, document_id),
            ).fetchone()
            current_revision = int(existing[0]) if existing is not None else 0
            if current_revision != expected_revision:
                raise StorageConflictError(
                    "Document revision conflict: "
                    f"expected {expected_revision}, found {current_revision}"
                )
            connection.execute(
                """
                INSERT INTO local_documents(
                    scope_identity, namespace, document_id, document_json,
                    revision, updated_at, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_identity, namespace, document_id) DO UPDATE SET
                    document_json = excluded.document_json,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at,
                    schema_version = excluded.schema_version
                """,
                (
                    _scope_identity(scope),
                    namespace,
                    document_id,
                    _json(record.document),
                    record.revision,
                    record.updated_at.isoformat(),
                    record.schema_version,
                ),
            )
            connection.execute(
                """
                DELETE FROM local_document_metadata
                WHERE scope_identity = ? AND namespace = ? AND document_id = ?
                """,
                (_scope_identity(scope), namespace, document_id),
            )
            connection.executemany(
                """
                INSERT INTO local_document_metadata(
                    scope_identity, namespace, document_id, key, value_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        _scope_identity(scope),
                        namespace,
                        document_id,
                        key,
                        _json(value),
                    )
                    for key, value in sorted(record.document.items())
                ),
            )
            return record

        return await self._database.transaction(commit)

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        document_id: str,
        expected_revision: int,
    ) -> bool:
        """Delete one exact document at its current revision.

        Foreign-key cascade removes the normalized metadata projection atomically.

        Examples:
            Delete a manifest:
                ```python
                deleted = await store.delete(scope, "registry", key, revision)
                ```

            Confirm absence:
                ```python
                assert await store.delete(scope, "registry", "missing", 0) is False
                ```

        Args:
            scope: Exact canonical owner scope.
            namespace: Exact logical namespace.
            document_id: Exact stable document identity.
            expected_revision: Required current revision.

        Returns:
            bool: `True` when deleted; `False` for absence at revision zero.

        Notes:
            Revision mismatch raises `StorageConflictError`.
        """
        self._require_writable()
        _identity(namespace, document_id)
        _expected_revision(expected_revision)

        def commit(connection: sqlite3.Connection) -> bool:
            existing = connection.execute(
                """
                SELECT revision FROM local_documents
                WHERE scope_identity = ? AND namespace = ? AND document_id = ?
                """,
                (_scope_identity(scope), namespace, document_id),
            ).fetchone()
            current_revision = int(existing[0]) if existing is not None else 0
            if current_revision == 0 and expected_revision == 0:
                return False
            if current_revision != expected_revision:
                raise StorageConflictError(
                    "Document revision conflict: "
                    f"expected {expected_revision}, found {current_revision}"
                )
            connection.execute(
                """
                DELETE FROM local_documents
                WHERE scope_identity = ? AND namespace = ? AND document_id = ?
                """,
                (_scope_identity(scope), namespace, document_id),
            )
            return True

        return await self._database.transaction(commit)

    async def query(self, query: DocumentQuery) -> Page[DocumentRecord]:
        """Query one bounded stable document-identity page.

        Exact scope, namespace, identifier prefix, and top-level metadata filter
        before opaque cursor pagination.

        Examples:
            List registry manifests:
                ```python
                page = await store.query(DocumentQuery(scope=scope, namespace="registry"))
                ```

            Continue a filtered query:
                ```python
                page = await store.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact canonical filters and page request.

        Returns:
            Page[DocumentRecord]: Matching documents and continuation cursor.

        Notes:
            Metadata is normalized and filtered in SQL, never after an unbounded read.
        """
        if len(query.metadata) > _MAX_METADATA_FILTERS:
            raise StorageConfigurationError(
                f"Document query exceeds {_MAX_METADATA_FILTERS} metadata filters"
            )
        fingerprint = _fingerprint(
            "document",
            _scope_identity(query.scope),
            query.namespace,
            query.id_prefix or "",
            _json(query.metadata),
        )
        clauses = ["d.scope_identity = ?", "d.namespace = ?"]
        values: list[object] = [_scope_identity(query.scope), query.namespace]
        _prefix_filter(clauses, values, "d.document_id", query.id_prefix)
        for index, (key, value) in enumerate(sorted(query.metadata.items())):
            alias = f"m{index}"
            clauses.append(
                "EXISTS (SELECT 1 FROM local_document_metadata AS "
                f"{alias} WHERE {alias}.scope_identity = d.scope_identity "
                f"AND {alias}.namespace = d.namespace "
                f"AND {alias}.document_id = d.document_id "
                f"AND {alias}.key = ? AND {alias}.value_json = ?)"
            )
            values.extend((key, _json(value)))
        if query.page.cursor is not None:
            clauses.append("d.document_id > ?")
            values.append(_decode_cursor(query.page.cursor, fingerprint))
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            f"SELECT d.* FROM local_documents AS d WHERE {' AND '.join(clauses)} "
            "ORDER BY d.document_id ASC LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        return Page(
            items=tuple(_document(row) for row in selected),
            next_cursor=(
                _encode_cursor(fingerprint, str(selected[-1]["document_id"]))
                if len(rows) > query.page.limit
                else None
            ),
        )

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local document store is read-only")


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.CONTROL:
        raise StorageConfigurationError("Local supporting stores require the control database")
    database.install_component(
        name="supporting",
        version=_SUPPORTING_COMPONENT_VERSION,
        statements=(
            _CREATE_KV,
            _CREATE_KV_EXPIRY_INDEX,
            _CREATE_DOCUMENTS,
            _CREATE_DOCUMENT_METADATA,
            _CREATE_DOCUMENT_METADATA_INDEX,
        ),
    )


def _visible_revision(row: sqlite3.Row | None, now: datetime) -> int:
    if row is None:
        return 0
    try:
        expires_at = row["expires_at"]
        if expires_at is not None and datetime.fromisoformat(str(expires_at)) <= now:
            return 0
        revision = int(row["revision"])
        if revision < 1:
            raise ValueError("revision")
        return revision
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted KV revision or expiration is malformed") from exc


def _key_value(row: sqlite3.Row) -> KeyValueRecord:
    try:
        expires_at = row["expires_at"]
        return KeyValueRecord(
            namespace=str(row["namespace"]),
            key=str(row["key"]),
            value=json.loads(row["value_json"]),
            revision=int(row["revision"]),
            scope=_scope(str(row["scope_identity"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            expires_at=datetime.fromisoformat(str(expires_at)) if expires_at else None,
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local key-value row is malformed") from exc


def _document(row: sqlite3.Row) -> DocumentRecord:
    try:
        return DocumentRecord(
            namespace=str(row["namespace"]),
            document_id=str(row["document_id"]),
            document=json.loads(row["document_json"]),
            revision=int(row["revision"]),
            scope=_scope(str(row["scope_identity"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local document row is malformed") from exc


def _now(clock: StorageClock) -> datetime:
    value = clock.now()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise StorageIntegrityError("Provider clock returned a naive or invalid timestamp")
    if value.utcoffset().total_seconds() != 0:
        raise StorageIntegrityError("Provider clock must return UTC timestamps")
    return value


def _expected_revision(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected_revision must be a non-negative integer")


def _identity(namespace: str, identity: str) -> None:
    for name, value in (("namespace", namespace), ("identity", identity)):
        if not isinstance(value, str) or not value.strip():
            raise StorageConfigurationError(f"{name} must be a non-empty string")


def _prefix_filter(
    clauses: list[str],
    values: list[object],
    column: str,
    prefix: str | None,
) -> None:
    if prefix is None:
        return
    clauses.extend((f"{column} >= ?", f"{column} < ?"))
    values.extend((prefix, prefix + chr(0x10FFFF)))


def _scope_identity(scope: StorageScope) -> str:
    return json.dumps(scope.as_filter(), sort_keys=True, separators=(",", ":"))


def _scope(identity: str) -> StorageScope:
    try:
        payload = json.loads(identity)
        if not isinstance(payload, dict):
            raise TypeError("scope")
        return StorageScope(**payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted supporting-store scope is malformed") from exc


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:24]


def _encode_cursor(fingerprint: str, identity: str) -> str:
    payload = json.dumps(
        {"fingerprint": fingerprint, "identity": identity},
        sort_keys=True,
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> str:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, dict) or set(payload) != {"fingerprint", "identity"}:
            raise ValueError("cursor payload")
        if payload["fingerprint"] != fingerprint:
            raise ValueError("cursor context")
        identity = payload["identity"]
        if not isinstance(identity, str) or not identity:
            raise ValueError("cursor identity")
        return identity
    except (
        binascii.Error,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StorageConfigurationError("Invalid or mismatched supporting cursor") from exc

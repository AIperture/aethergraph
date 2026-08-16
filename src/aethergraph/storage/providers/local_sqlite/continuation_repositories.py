"""Transactional local continuation and timer-lease repositories."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
import hashlib
import hmac
import json
import secrets
import sqlite3

from ...contracts import (
    ContinuationCorrelator,
    ContinuationDraft,
    ContinuationLeaseQuery,
    ContinuationLeaseRecord,
    ContinuationLeaseRequest,
    ContinuationLeaseStatus,
    ContinuationQuery,
    ContinuationRecord,
    ContinuationStatus,
    CreatedContinuation,
    Page,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalDatabaseRole, LocalSQLiteDatabase

_COMPONENT_VERSION = 1
_SCOPE_COLUMNS = (
    "tenant_id",
    "project_id",
    "org_id",
    "user_id",
    "session_id",
    "run_id",
    "graph_id",
    "node_id",
    "agent_id",
    "scope_key",
)
_CREATE_CONTINUATIONS = """
CREATE TABLE local_continuations (
    continuation_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    run_id TEXT NOT NULL,
    graph_id TEXT,
    node_id TEXT NOT NULL,
    agent_id TEXT,
    scope_key TEXT,
    created_at TEXT NOT NULL,
    token_digest TEXT NOT NULL UNIQUE,
    revision INTEGER NOT NULL CHECK (revision > 0),
    status TEXT NOT NULL,
    prompt TEXT,
    resume_schema_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    poll_payload_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    deadline TEXT,
    next_wakeup_at TEXT,
    channel TEXT,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    closed_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_CONTINUATION_SCOPE_INDEX = """
CREATE INDEX ix_local_continuations_scope_created
ON local_continuations(run_id, node_id, created_at DESC, continuation_id DESC)
"""
_CREATE_CONTINUATION_DUE_INDEX = """
CREATE INDEX ix_local_continuations_scope_due
ON local_continuations(run_id, node_id, status, next_wakeup_at, continuation_id)
"""
_CREATE_CONTINUATION_CHANNEL_INDEX = """
CREATE INDEX ix_local_continuations_channel_created
ON local_continuations(channel, created_at DESC, continuation_id DESC)
"""
_CREATE_CONTINUATION_SESSION_OPEN_INDEX = """
CREATE INDEX ix_local_continuations_session_open
ON local_continuations(session_id, status, kind, deadline, created_at DESC, continuation_id DESC)
"""
_CREATE_CORRELATORS = """
CREATE TABLE local_continuation_correlators (
    continuation_id TEXT NOT NULL
        REFERENCES local_continuations(continuation_id) ON DELETE CASCADE,
    scheme TEXT NOT NULL,
    channel TEXT NOT NULL,
    thread TEXT NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY(continuation_id, scheme, channel, thread, message)
)
"""
_CREATE_CORRELATOR_INDEX = """
CREATE INDEX ix_local_continuation_correlators_lookup
ON local_continuation_correlators(scheme, channel, thread, message, continuation_id)
"""
_CREATE_LEASES = """
CREATE TABLE local_continuation_leases (
    fire_id TEXT PRIMARY KEY,
    continuation_id TEXT NOT NULL
        REFERENCES local_continuations(continuation_id) ON DELETE RESTRICT,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    run_id TEXT NOT NULL,
    graph_id TEXT,
    node_id TEXT NOT NULL,
    agent_id TEXT,
    scope_key TEXT,
    scheduled_for TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts > 0),
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    worker_id TEXT,
    lease_until TEXT,
    next_attempt_at TEXT,
    last_error TEXT,
    finished_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_LEASE_SCOPE_INDEX = """
CREATE INDEX ix_local_continuation_leases_scope_updated
ON local_continuation_leases(run_id, node_id, updated_at DESC, fire_id DESC)
"""
_CREATE_LEASE_STATUS_INDEX = """
CREATE INDEX ix_local_continuation_leases_status_updated
ON local_continuation_leases(status, updated_at DESC, fire_id DESC)
"""
_CREATE_LEASE_CONTINUATION_INDEX = """
CREATE INDEX ix_local_continuation_leases_continuation_updated
ON local_continuation_leases(continuation_id, updated_at DESC, fire_id DESC)
"""


class LocalContinuationRepository:
    """Canonical local continuations with atomic secret and correlator indexes."""

    def __init__(self, *, database: LocalSQLiteDatabase, token_secret: str | bytes) -> None:
        _install(database)
        if isinstance(token_secret, str):
            token_secret = token_secret.encode()
        if not isinstance(token_secret, bytes) or len(token_secret) < 32:
            raise StorageConfigurationError("Continuation token secret must be at least 32 bytes")
        self._database = database
        self._mode = database.mode
        self._token_secret = token_secret

    async def create(self, draft: ContinuationDraft) -> CreatedContinuation:
        """Atomically create a continuation and all initial lookup indexes.

        Examples:
            Create a wait:
                ```python
                created = await repository.create(draft)
                ```

        Args:
            draft: Immutable continuation content and initial correlators.

        Returns:
            CreatedContinuation: Revision-one record and its one-time raw token.

        Notes:
            Raw token material is never persisted and identity retries are rejected.
        """
        self._require_writable()
        token = secrets.token_urlsafe(32)
        record = ContinuationRecord(
            continuation_id=draft.continuation_id,
            kind=draft.kind,
            scope=draft.scope,
            created_at=draft.created_at,
            token_digest=self._digest(token),
            revision=1,
            prompt=draft.prompt,
            resume_schema=draft.resume_schema,
            payload=draft.payload,
            poll_payload=draft.poll_payload,
            metadata=draft.metadata,
            deadline=draft.deadline,
            next_wakeup_at=draft.next_wakeup_at,
            channel=draft.channel,
            correlators=draft.correlators,
            schema_version=draft.schema_version,
        )

        def commit(connection: sqlite3.Connection) -> CreatedContinuation:
            if connection.execute(
                "SELECT 1 FROM local_continuations WHERE continuation_id = ?",
                (record.continuation_id,),
            ).fetchone():
                raise StorageIntegrityError(
                    f"Continuation identity {record.continuation_id!r} conflicts"
                )
            _insert_continuation(connection, record)
            _replace_correlators(connection, record.continuation_id, record.correlators)
            return CreatedContinuation(record=record, token=token)

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, continuation_id: str) -> ContinuationRecord | None:
        """Read one exact authorized continuation.

        Examples:
            Read a wait:
                ```python
                wait = await repository.get(scope, "cont-1")
                ```

        Args:
            scope: Populated canonical scope constraining access.
            continuation_id: Exact continuation identity.

        Returns:
            ContinuationRecord | None: Authorized record or `None`.

        Notes:
            A miss never broadens to another scope.
        """
        _nonempty("continuation_id", continuation_id)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_continuations WHERE continuation_id = ?",
            (continuation_id,),
        )
        if not rows:
            return None
        record = await self._hydrate(rows[0])
        return record if _scope_authorizes(record.scope, scope) else None

    async def resolve_token(self, token: str) -> ContinuationRecord | None:
        """Resolve a raw bearer token through the protected exact index.

        Examples:
            Resolve an inbound response:
                ```python
                wait = await repository.resolve_token(token)
                ```

        Args:
            token: Exact non-empty raw bearer token.

        Returns:
            ContinuationRecord | None: Matching current record or `None`.

        Notes:
            Digest verification uses a constant-time comparison.
        """
        _nonempty("token", token)
        digest = self._digest(token)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_continuations WHERE token_digest = ?", (digest,)
        )
        if not rows or not hmac.compare_digest(str(rows[0]["token_digest"]), digest):
            return None
        return await self._hydrate(rows[0])

    async def compare_and_set(
        self, record: ContinuationRecord, expected_revision: int
    ) -> ContinuationRecord:
        """Atomically replace mutable continuation state at the next revision.

        Examples:
            Close a resumed wait:
                ```python
                stored = await repository.compare_and_set(resumed, current.revision)
                ```

        Args:
            record: Complete canonical next revision.
            expected_revision: Exact current revision required for the update.

        Returns:
            ContinuationRecord: Newly committed authoritative record.

        Notes:
            Identity, token digest, creation fields, and terminal records are immutable.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)

        def commit(connection: sqlite3.Connection) -> ContinuationRecord:
            row = connection.execute(
                "SELECT * FROM local_continuations WHERE continuation_id = ?",
                (record.continuation_id,),
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(record.continuation_id)
            current = _continuation(row, _correlators(connection, record.continuation_id))
            if current.revision != expected_revision:
                raise StorageConflictError("Continuation revision is stale")
            if _continuation_identity(current) != _continuation_identity(record):
                raise StorageIntegrityError("Continuation immutable identity changed")
            if current.status is not ContinuationStatus.WAITING:
                raise StorageConflictError("Terminal continuation is immutable")
            _update_continuation(connection, record)
            _replace_correlators(connection, record.continuation_id, record.correlators)
            return record

        return await self._database.transaction(commit)

    async def bind_correlator(
        self,
        scope: StorageScope,
        continuation_id: str,
        correlator: ContinuationCorrelator,
        expected_revision: int,
    ) -> ContinuationRecord:
        """Atomically add one idempotent correlator and reverse index.

        Examples:
            Bind a sent message:
                ```python
                updated = await repository.bind_correlator(scope, wait_id, corr, revision)
                ```

        Args:
            scope: Populated canonical scope constraining the update.
            continuation_id: Exact continuation identity.
            correlator: Transport-neutral identity to bind.
            expected_revision: Current revision required for a new binding.

        Returns:
            ContinuationRecord: Current record containing the correlator.

        Notes:
            An already-present binding is idempotent even after its original revision.
        """
        self._require_writable()
        _nonempty("continuation_id", continuation_id)

        def commit(connection: sqlite3.Connection) -> ContinuationRecord:
            row = connection.execute(
                "SELECT * FROM local_continuations WHERE continuation_id = ?",
                (continuation_id,),
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(continuation_id)
            current = _continuation(row, _correlators(connection, continuation_id))
            if not _scope_authorizes(current.scope, scope):
                raise StorageNotFoundError(continuation_id)
            if correlator in current.correlators:
                return current
            if current.status is not ContinuationStatus.WAITING:
                raise StorageConflictError("Terminal continuation is immutable")
            if current.revision != expected_revision:
                raise StorageConflictError("Continuation revision is stale")
            updated = replace(
                current,
                revision=current.revision + 1,
                correlators=(*current.correlators, correlator),
            )
            _update_continuation(connection, updated)
            _insert_correlator(connection, continuation_id, correlator)
            return updated

        return await self._database.transaction(commit)

    async def query(self, query: ContinuationQuery) -> Page[ContinuationRecord]:
        """Query one bounded stable page through canonical continuation indexes.

        Examples:
            Read due waits:
                ```python
                page = await repository.query(ContinuationQuery(scope=scope))
                ```

        Args:
            query: Scope, indexed filters, and opaque page request.

        Returns:
            Page[ContinuationRecord]: Matching records and continuation cursor.

        Notes:
            Cursors are bound to the complete query and cannot be reused elsewhere.
        """
        clauses, values = _scope_filters(query.scope, alias="c")
        joins = ""
        if query.statuses:
            clauses.append(f"c.status IN ({','.join('?' for _ in query.statuses)})")
            values.extend(value.value for value in query.statuses)
        if query.kinds:
            clauses.append(f"c.kind IN ({','.join('?' for _ in query.kinds)})")
            values.extend(query.kinds)
        if query.channel is not None:
            clauses.append("c.channel = ?")
            values.append(query.channel)
        if query.correlator is not None:
            joins = (
                " JOIN local_continuation_correlators r ON r.continuation_id = c.continuation_id"
            )
            clauses.extend(("r.scheme = ?", "r.channel = ?", "r.thread = ?", "r.message = ?"))
            values.extend(_correlator_values(query.correlator))
        due = query.due_at_or_before is not None
        if due:
            clauses.extend(("c.next_wakeup_at IS NOT NULL", "c.next_wakeup_at <= ?"))
            values.append(query.due_at_or_before.isoformat())
        if query.open_at is not None:
            clauses.append("(c.deadline IS NULL OR c.deadline >= ?)")
            values.append(query.open_at.isoformat())
        fingerprint = _query_fingerprint("continuations", query, without_cursor=True)
        if query.page.cursor:
            timestamp, identity = _decode_cursor(query.page.cursor, fingerprint)
            operator = ">" if due else "<"
            column = "c.next_wakeup_at" if due else "c.created_at"
            clauses.append(f"({column}, c.continuation_id) {operator} (?, ?)")
            values.extend((timestamp, identity))
        order_column = "c.next_wakeup_at ASC" if due else "c.created_at DESC"
        order_id = "c.continuation_id ASC" if due else "c.continuation_id DESC"
        rows = await self._database.fetch_all(
            f"SELECT c.* FROM local_continuations c{joins} "
            f"WHERE {' AND '.join(clauses)} ORDER BY {order_column}, {order_id} LIMIT ?",
            (*values, query.page.limit + 1),
        )
        visible = rows[: query.page.limit]
        records = tuple([await self._hydrate(row) for row in visible])
        next_cursor = None
        if len(rows) > query.page.limit:
            anchor = visible[-1]
            timestamp = str(anchor["next_wakeup_at"] if due else anchor["created_at"])
            next_cursor = _encode_cursor(fingerprint, timestamp, str(anchor["continuation_id"]))
        return Page(items=records, next_cursor=next_cursor)

    async def _hydrate(self, row: sqlite3.Row) -> ContinuationRecord:
        correlator_rows = await self._database.fetch_all(
            """
            SELECT scheme, channel, thread, message
            FROM local_continuation_correlators
            WHERE continuation_id = ?
            ORDER BY rowid
            """,
            (str(row["continuation_id"]),),
        )
        return _continuation(row, tuple(_correlator(item) for item in correlator_rows))

    def _digest(self, token: str) -> str:
        return (
            "hmac-sha256:"
            + hmac.new(self._token_secret, token.encode(), hashlib.sha256).hexdigest()
        )

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local continuation repository is read-only")


class LocalContinuationLeaseRepository:
    """Canonical local claims and durable receipts for continuation timers."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def claim(self, request: ContinuationLeaseRequest) -> ContinuationLeaseRecord | None:
        """Atomically create, retry, or reclaim one eligible timer fire.

        Examples:
            Claim an occurrence:
                ```python
                lease = await repository.claim(request)
                ```

        Args:
            request: Exact occurrence, worker, clock, and lease interval.

        Returns:
            ContinuationLeaseRecord | None: Worker-owned lease or `None`.

        Notes:
            Terminal receipts, active leases, and delayed retries are not claimable.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> ContinuationLeaseRecord | None:
            continuation_row = connection.execute(
                "SELECT * FROM local_continuations WHERE continuation_id = ?",
                (request.continuation_id,),
            ).fetchone()
            if continuation_row is None:
                raise StorageNotFoundError(request.continuation_id)
            continuation = _continuation(
                continuation_row, _correlators(connection, request.continuation_id)
            )
            if not _scope_authorizes(continuation.scope, request.scope):
                raise StorageNotFoundError(request.continuation_id)
            if (
                continuation.status is not ContinuationStatus.WAITING
                or continuation.next_wakeup_at != request.scheduled_for
            ):
                return None
            row = connection.execute(
                "SELECT * FROM local_continuation_leases WHERE fire_id = ?",
                (request.fire_id,),
            ).fetchone()
            if row is None:
                claimed = ContinuationLeaseRecord(
                    fire_id=request.fire_id,
                    continuation_id=request.continuation_id,
                    scope=request.scope,
                    scheduled_for=request.scheduled_for,
                    status=ContinuationLeaseStatus.LEASED,
                    attempts=1,
                    revision=1,
                    updated_at=request.now,
                    worker_id=request.worker_id,
                    lease_until=request.lease_until,
                )
                _insert_lease(connection, claimed)
                return claimed
            current = _lease(row)
            if (
                current.continuation_id != request.continuation_id
                or current.scope != request.scope
                or current.scheduled_for != request.scheduled_for
            ):
                raise StorageIntegrityError("Continuation lease occurrence identity conflicts")
            if request.now < current.updated_at:
                raise StorageIntegrityError("Continuation lease claim clock moved backward")
            if current.status in {
                ContinuationLeaseStatus.DELIVERED,
                ContinuationLeaseStatus.DEAD_LETTER,
            }:
                return None
            if (
                current.status is ContinuationLeaseStatus.LEASED
                and current.lease_until is not None
                and current.lease_until > request.now
            ):
                return None
            if (
                current.status is ContinuationLeaseStatus.RETRY
                and current.next_attempt_at is not None
                and current.next_attempt_at > request.now
            ):
                return None
            claimed = replace(
                current,
                status=ContinuationLeaseStatus.LEASED,
                attempts=current.attempts + 1,
                revision=current.revision + 1,
                updated_at=request.now,
                worker_id=request.worker_id,
                lease_until=request.lease_until,
                next_attempt_at=None,
                last_error=None,
                finished_at=None,
            )
            _update_lease(connection, claimed)
            return claimed

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, fire_id: str) -> ContinuationLeaseRecord | None:
        """Read one exact authorized timer claim or receipt.

        Examples:
            Inspect a receipt:
                ```python
                receipt = await repository.get(scope, "fire-1")
                ```

        Args:
            scope: Populated canonical scope constraining access.
            fire_id: Exact scheduled occurrence identity.

        Returns:
            ContinuationLeaseRecord | None: Authorized record or `None`.

        Notes:
            Reads never renew or reclaim leases.
        """
        _nonempty("fire_id", fire_id)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_continuation_leases WHERE fire_id = ?", (fire_id,)
        )
        if not rows:
            return None
        record = _lease(rows[0])
        return record if _scope_authorizes(record.scope, scope) else None

    async def compare_and_set(
        self, record: ContinuationLeaseRecord, expected_revision: int
    ) -> ContinuationLeaseRecord:
        """Atomically renew, release, or terminalize a claimed lease.

        Examples:
            Record delivery:
                ```python
                receipt = await repository.compare_and_set(delivered, lease.revision)
                ```

        Args:
            record: Complete canonical next lease or receipt revision.
            expected_revision: Exact current revision required for the transition.

        Returns:
            ContinuationLeaseRecord: Newly committed authoritative record.

        Notes:
            Only active leases transition; terminal receipts remain immutable.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)

        def commit(connection: sqlite3.Connection) -> ContinuationLeaseRecord:
            row = connection.execute(
                "SELECT * FROM local_continuation_leases WHERE fire_id = ?",
                (record.fire_id,),
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(record.fire_id)
            current = _lease(row)
            if current.revision != expected_revision:
                raise StorageConflictError("Continuation lease revision is stale")
            if _lease_identity(current) != _lease_identity(record):
                raise StorageIntegrityError("Continuation lease immutable identity changed")
            if current.status is not ContinuationLeaseStatus.LEASED:
                raise StorageConflictError("Only an active continuation lease may transition")
            if record.status is ContinuationLeaseStatus.LEASED:
                if record.worker_id != current.worker_id:
                    raise StorageConflictError("Continuation lease worker ownership changed")
                if record.lease_until is None or current.lease_until is None:
                    raise StorageIntegrityError("Continuation lease renewal is malformed")
                if record.lease_until <= current.lease_until:
                    raise StorageIntegrityError("Continuation lease renewal must extend ownership")
            if record.updated_at < current.updated_at:
                raise StorageIntegrityError("Continuation lease timestamp moved backward")
            _update_lease(connection, record)
            return record

        return await self._database.transaction(commit)

    async def query(self, query: ContinuationLeaseQuery) -> Page[ContinuationLeaseRecord]:
        """Query one bounded stable page of timer claims and receipts.

        Examples:
            Inspect dead letters:
                ```python
                page = await repository.query(ContinuationLeaseQuery(scope=scope))
                ```

        Args:
            query: Scope, indexed filters, and opaque page request.

        Returns:
            Page[ContinuationLeaseRecord]: Matching records and continuation cursor.

        Notes:
            Workers claim exact occurrences rather than scanning this diagnostic API.
        """
        clauses, values = _scope_filters(query.scope, alias="l")
        if query.statuses:
            clauses.append(f"l.status IN ({','.join('?' for _ in query.statuses)})")
            values.extend(value.value for value in query.statuses)
        if query.continuation_id is not None:
            clauses.append("l.continuation_id = ?")
            values.append(query.continuation_id)
        fingerprint = _query_fingerprint("continuation-leases", query, without_cursor=True)
        if query.page.cursor:
            timestamp, identity = _decode_cursor(query.page.cursor, fingerprint)
            clauses.append("(l.updated_at, l.fire_id) < (?, ?)")
            values.extend((timestamp, identity))
        rows = await self._database.fetch_all(
            "SELECT l.* FROM local_continuation_leases l "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY l.updated_at DESC, l.fire_id DESC LIMIT ?",
            (*values, query.page.limit + 1),
        )
        visible = rows[: query.page.limit]
        next_cursor = None
        if len(rows) > query.page.limit:
            anchor = visible[-1]
            next_cursor = _encode_cursor(
                fingerprint, str(anchor["updated_at"]), str(anchor["fire_id"])
            )
        return Page(items=tuple(_lease(row) for row in visible), next_cursor=next_cursor)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local continuation lease repository is read-only")


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.CONTROL:
        raise StorageConfigurationError("Local continuation repositories require control database")
    database.install_component(
        name="continuations",
        version=_COMPONENT_VERSION,
        statements=(
            _CREATE_CONTINUATIONS,
            _CREATE_CONTINUATION_SCOPE_INDEX,
            _CREATE_CONTINUATION_DUE_INDEX,
            _CREATE_CONTINUATION_CHANNEL_INDEX,
            _CREATE_CONTINUATION_SESSION_OPEN_INDEX,
            _CREATE_CORRELATORS,
            _CREATE_CORRELATOR_INDEX,
            _CREATE_LEASES,
            _CREATE_LEASE_SCOPE_INDEX,
            _CREATE_LEASE_STATUS_INDEX,
            _CREATE_LEASE_CONTINUATION_INDEX,
        ),
    )


def _insert_continuation(connection: sqlite3.Connection, record: ContinuationRecord) -> None:
    connection.execute(
        """
        INSERT INTO local_continuations(
            continuation_id, kind, tenant_id, project_id, org_id, user_id,
            session_id, run_id, graph_id, node_id, agent_id, scope_key, created_at,
            token_digest, revision, status, prompt, resume_schema_json, payload_json,
            poll_payload_json, metadata_json, deadline, next_wakeup_at, channel,
            attempts, closed_at, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _continuation_values(record),
    )


def _update_continuation(connection: sqlite3.Connection, record: ContinuationRecord) -> None:
    values = _continuation_values(record)
    connection.execute(
        """
        UPDATE local_continuations SET
            kind = ?, tenant_id = ?, project_id = ?, org_id = ?, user_id = ?,
            session_id = ?, run_id = ?, graph_id = ?, node_id = ?, agent_id = ?,
            scope_key = ?, created_at = ?, token_digest = ?, revision = ?, status = ?,
            prompt = ?, resume_schema_json = ?, payload_json = ?, poll_payload_json = ?,
            metadata_json = ?, deadline = ?, next_wakeup_at = ?, channel = ?, attempts = ?,
            closed_at = ?, schema_version = ?
        WHERE continuation_id = ?
        """,
        (*values[1:], values[0]),
    )


def _continuation_values(record: ContinuationRecord) -> tuple[object, ...]:
    return (
        record.continuation_id,
        record.kind,
        *(_scope_values(record.scope)),
        record.created_at.isoformat(),
        record.token_digest,
        record.revision,
        record.status.value,
        record.prompt,
        _json(record.resume_schema),
        _json(record.payload),
        _json(record.poll_payload),
        _json(record.metadata),
        _optional_iso(record.deadline),
        _optional_iso(record.next_wakeup_at),
        record.channel,
        record.attempts,
        _optional_iso(record.closed_at),
        record.schema_version,
    )


def _continuation(
    row: sqlite3.Row, correlators: tuple[ContinuationCorrelator, ...]
) -> ContinuationRecord:
    try:
        return ContinuationRecord(
            continuation_id=str(row["continuation_id"]),
            kind=str(row["kind"]),
            scope=_scope(row),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            token_digest=str(row["token_digest"]),
            revision=int(row["revision"]),
            status=ContinuationStatus(str(row["status"])),
            prompt=row["prompt"],
            resume_schema=_json_object(row["resume_schema_json"]),
            payload=_json_object(row["payload_json"]),
            poll_payload=_json_object(row["poll_payload_json"]),
            metadata=_json_object(row["metadata_json"]),
            deadline=_optional_time(row["deadline"]),
            next_wakeup_at=_optional_time(row["next_wakeup_at"]),
            channel=row["channel"],
            correlators=correlators,
            attempts=int(row["attempts"]),
            closed_at=_optional_time(row["closed_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local continuation row is malformed") from exc


def _continuation_identity(record: ContinuationRecord) -> tuple[object, ...]:
    return (
        record.continuation_id,
        record.kind,
        record.scope,
        record.created_at,
        record.token_digest,
        record.schema_version,
    )


def _replace_correlators(
    connection: sqlite3.Connection,
    continuation_id: str,
    correlators: tuple[ContinuationCorrelator, ...],
) -> None:
    connection.execute(
        "DELETE FROM local_continuation_correlators WHERE continuation_id = ?",
        (continuation_id,),
    )
    for correlator in correlators:
        _insert_correlator(connection, continuation_id, correlator)


def _insert_correlator(
    connection: sqlite3.Connection,
    continuation_id: str,
    correlator: ContinuationCorrelator,
) -> None:
    connection.execute(
        """
        INSERT INTO local_continuation_correlators(
            continuation_id, scheme, channel, thread, message
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (continuation_id, *_correlator_values(correlator)),
    )


def _correlators(
    connection: sqlite3.Connection, continuation_id: str
) -> tuple[ContinuationCorrelator, ...]:
    rows = connection.execute(
        """
        SELECT scheme, channel, thread, message
        FROM local_continuation_correlators
        WHERE continuation_id = ? ORDER BY rowid
        """,
        (continuation_id,),
    ).fetchall()
    return tuple(_correlator(row) for row in rows)


def _correlator(row: sqlite3.Row) -> ContinuationCorrelator:
    try:
        return ContinuationCorrelator(
            scheme=str(row["scheme"]),
            channel=str(row["channel"]),
            thread=str(row["thread"]),
            message=str(row["message"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted local continuation correlator is malformed") from exc


def _correlator_values(correlator: ContinuationCorrelator) -> tuple[str, str, str, str]:
    return correlator.scheme, correlator.channel, correlator.thread, correlator.message


def _insert_lease(connection: sqlite3.Connection, record: ContinuationLeaseRecord) -> None:
    connection.execute(
        """
        INSERT INTO local_continuation_leases(
            fire_id, continuation_id, tenant_id, project_id, org_id, user_id,
            session_id, run_id, graph_id, node_id, agent_id, scope_key, scheduled_for,
            status, attempts, revision, updated_at, worker_id, lease_until,
            next_attempt_at, last_error, finished_at, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _lease_values(record),
    )


def _update_lease(connection: sqlite3.Connection, record: ContinuationLeaseRecord) -> None:
    values = _lease_values(record)
    connection.execute(
        """
        UPDATE local_continuation_leases SET
            continuation_id = ?, tenant_id = ?, project_id = ?, org_id = ?,
            user_id = ?, session_id = ?, run_id = ?, graph_id = ?, node_id = ?,
            agent_id = ?, scope_key = ?, scheduled_for = ?, status = ?, attempts = ?,
            revision = ?, updated_at = ?, worker_id = ?, lease_until = ?,
            next_attempt_at = ?, last_error = ?, finished_at = ?, schema_version = ?
        WHERE fire_id = ?
        """,
        (*values[1:], values[0]),
    )


def _lease_values(record: ContinuationLeaseRecord) -> tuple[object, ...]:
    return (
        record.fire_id,
        record.continuation_id,
        *_scope_values(record.scope),
        record.scheduled_for.isoformat(),
        record.status.value,
        record.attempts,
        record.revision,
        record.updated_at.isoformat(),
        record.worker_id,
        _optional_iso(record.lease_until),
        _optional_iso(record.next_attempt_at),
        record.last_error,
        _optional_iso(record.finished_at),
        record.schema_version,
    )


def _lease(row: sqlite3.Row) -> ContinuationLeaseRecord:
    try:
        return ContinuationLeaseRecord(
            fire_id=str(row["fire_id"]),
            continuation_id=str(row["continuation_id"]),
            scope=_scope(row),
            scheduled_for=datetime.fromisoformat(str(row["scheduled_for"])),
            status=ContinuationLeaseStatus(str(row["status"])),
            attempts=int(row["attempts"]),
            revision=int(row["revision"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            worker_id=row["worker_id"],
            lease_until=_optional_time(row["lease_until"]),
            next_attempt_at=_optional_time(row["next_attempt_at"]),
            last_error=row["last_error"],
            finished_at=_optional_time(row["finished_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted local continuation lease row is malformed") from exc


def _lease_identity(record: ContinuationLeaseRecord) -> tuple[object, ...]:
    return (
        record.fire_id,
        record.continuation_id,
        record.scope,
        record.scheduled_for,
        record.attempts,
        record.schema_version,
    )


def _scope_values(scope: StorageScope) -> tuple[str | None, ...]:
    return tuple(getattr(scope, name) for name in _SCOPE_COLUMNS)


def _scope(row: sqlite3.Row) -> StorageScope:
    try:
        return StorageScope(**{name: row[name] for name in _SCOPE_COLUMNS})
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted local continuation scope is malformed") from exc


def _scope_filters(scope: StorageScope, *, alias: str) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    values: list[object] = []
    for name, value in scope.as_filter().items():
        clauses.append(f"{alias}.{name} = ?")
        values.append(value)
    if not clauses:
        raise StorageConfigurationError("Continuation queries require populated scope")
    return clauses, values


def _scope_authorizes(owner: StorageScope, operation: StorageScope) -> bool:
    filters = operation.as_filter()
    return bool(filters) and all(getattr(owner, name) == value for name, value in filters.items())


def _next_revision(revision: int, expected_revision: int) -> None:
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise ValueError("expected_revision must be an integer")
    if expected_revision < 0 or revision != expected_revision + 1:
        raise ValueError("record revision must equal expected_revision plus one")


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StorageConfigurationError(f"{name} must be a non-empty string")


def _optional_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _optional_time(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value is not None else None


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _json_object(value: object) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("persisted JSON value must be an object")
    return parsed


def _query_fingerprint(
    kind: str,
    query: ContinuationQuery | ContinuationLeaseQuery,
    *,
    without_cursor: bool,
) -> str:
    page = query.page
    payload = {
        "kind": kind,
        "scope": query.scope.as_filter(),
        "statuses": [value.value for value in query.statuses],
        "limit": page.limit,
    }
    if isinstance(query, ContinuationQuery):
        payload.update(
            kinds=list(query.kinds),
            channel=query.channel,
            correlator=(list(_correlator_values(query.correlator)) if query.correlator else None),
            due_at_or_before=_optional_iso(query.due_at_or_before),
        )
    else:
        payload["continuation_id"] = query.continuation_id
    if not without_cursor:
        payload["cursor"] = page.cursor
    return hashlib.sha256(_json(payload).encode()).hexdigest()[:24]


def _encode_cursor(fingerprint: str, timestamp: str, identity: str) -> str:
    payload = _json({"fingerprint": fingerprint, "timestamp": timestamp, "identity": identity})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> tuple[str, str]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, dict) or set(payload) != {
            "fingerprint",
            "timestamp",
            "identity",
        }:
            raise ValueError("cursor payload")
        if payload["fingerprint"] != fingerprint:
            raise ValueError("cursor context")
        timestamp = payload["timestamp"]
        identity = payload["identity"]
        if not isinstance(timestamp, str) or not isinstance(identity, str) or not identity:
            raise ValueError("cursor values")
        datetime.fromisoformat(timestamp)
        return timestamp, identity
    except (
        binascii.Error,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StorageConfigurationError("Invalid or mismatched continuation cursor") from exc

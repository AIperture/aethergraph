"""Transactional local run, result, and session repositories."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import sqlite3

from ...contracts import (
    Page,
    RunQuery,
    RunRecord,
    RunResultRecord,
    RunStatus,
    SessionKind,
    SessionQuery,
    SessionRecord,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
    storage_scope_covers,
    storage_scope_matches_filter,
)
from .database import LocalDatabaseRole, LocalSQLiteDatabase

_CONTROL_COMPONENT_VERSION = 1
_CREATE_RUNS = """
CREATE TABLE local_runs (
    run_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    node_id TEXT,
    agent_id TEXT,
    scope_key TEXT,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    started_at TEXT NOT NULL,
    finished_at TEXT,
    tags_json TEXT NOT NULL,
    error TEXT,
    metadata_json TEXT NOT NULL,
    artifact_count INTEGER NOT NULL CHECK (artifact_count >= 0),
    first_artifact_at TEXT,
    last_artifact_at TEXT,
    recent_artifact_ids_json TEXT NOT NULL,
    result_available INTEGER NOT NULL CHECK (result_available IN (0, 1)),
    result_updated_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_RUN_PROJECT_INDEX = """
CREATE INDEX ix_local_runs_project_started
ON local_runs(tenant_id, project_id, started_at DESC, run_id DESC)
"""
_CREATE_RUN_SESSION_INDEX = """
CREATE INDEX ix_local_runs_session_started
ON local_runs(session_id, started_at DESC, run_id DESC)
"""
_CREATE_RUN_GRAPH_INDEX = """
CREATE INDEX ix_local_runs_graph_started
ON local_runs(graph_id, started_at DESC, run_id DESC)
"""
_CREATE_RUN_STATUS_INDEX = """
CREATE INDEX ix_local_runs_status_started
ON local_runs(status, started_at DESC, run_id DESC)
"""
_CREATE_RUN_KIND_INDEX = """
CREATE INDEX ix_local_runs_kind_started
ON local_runs(kind, started_at DESC, run_id DESC)
"""
_CREATE_RUN_ARTIFACT_OCCURRENCES = """
CREATE TABLE local_run_artifact_occurrences (
    run_id TEXT NOT NULL REFERENCES local_runs(run_id) ON DELETE CASCADE,
    occurrence_id TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    PRIMARY KEY(run_id, occurrence_id)
)
"""
_CREATE_RUN_RESULTS = """
CREATE TABLE local_run_results (
    run_id TEXT PRIMARY KEY REFERENCES local_runs(run_id) ON DELETE CASCADE,
    graph_id TEXT NOT NULL,
    status TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source TEXT NOT NULL,
    snapshot_revision INTEGER,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_SESSIONS = """
CREATE TABLE local_sessions (
    session_id TEXT PRIMARY KEY,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    run_id TEXT,
    graph_id TEXT,
    node_id TEXT,
    agent_id TEXT,
    scope_key TEXT,
    kind TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    title TEXT,
    source TEXT NOT NULL,
    external_reference TEXT,
    metadata_json TEXT NOT NULL,
    artifact_count INTEGER NOT NULL CHECK (artifact_count >= 0),
    last_artifact_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_SESSION_PROJECT_INDEX = """
CREATE INDEX ix_local_sessions_project_updated
ON local_sessions(tenant_id, project_id, updated_at DESC, session_id DESC)
"""
_CREATE_SESSION_USER_INDEX = """
CREATE INDEX ix_local_sessions_user_updated
ON local_sessions(user_id, updated_at DESC, session_id DESC)
"""
_CREATE_SESSION_KIND_INDEX = """
CREATE INDEX ix_local_sessions_kind_updated
ON local_sessions(kind, updated_at DESC, session_id DESC)
"""
_CREATE_SESSION_ARTIFACT_OCCURRENCES = """
CREATE TABLE local_session_artifact_occurrences (
    session_id TEXT NOT NULL REFERENCES local_sessions(session_id) ON DELETE CASCADE,
    occurrence_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    PRIMARY KEY(session_id, occurrence_id)
)
"""


class LocalRunRepository:
    """Canonical local runs with revision CAS and idempotent artifact counters."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def create(self, record: RunRecord) -> RunRecord:
        """Idempotently create one initial canonical run.

        Initial provider-owned artifact/result state must be empty; exact retries
        return the stored record and conflicting run identity fails atomically.

        Examples:
            Create a run:
                ```python
                stored = await runs.create(record)
                ```

            Retry creation:
                ```python
                assert await runs.create(record) == stored
                ```

        Args:
            record: Complete initial canonical run at revision one.

        Returns:
            RunRecord: Authoritative stored run.

        Notes:
            Only canonical scope dimensions participate in identity and authorization.
        """
        self._require_writable()
        if record.revision != 1:
            raise StorageIntegrityError("Initial run revision must be one")
        if record.artifact_count or record.result_available:
            raise StorageIntegrityError("Initial run provider-owned counters must be empty")

        def commit(connection: sqlite3.Connection) -> RunRecord:
            existing = connection.execute(
                "SELECT * FROM local_runs WHERE run_id = ?", (record.run_id,)
            ).fetchone()
            if existing is not None:
                stored = _run(existing)
                if stored != record:
                    raise StorageIntegrityError(f"Run identity {record.run_id!r} conflicts")
                return stored
            _insert_run(connection, record)
            return record

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, run_id: str) -> RunRecord | None:
        """Read one exact authorized current run.

        Every populated caller scope dimension must match the indexed run identity.

        Examples:
            Read a run:
                ```python
                run = await runs.get(scope, "run-1")
                ```

            Detect a cross-scope identity:
                ```python
                assert await runs.get(other_scope, "run-1") is None
                ```

        Args:
            scope: Canonical scope constraints supplied by the caller.
            run_id: Exact stable run identity.

        Returns:
            RunRecord | None: Authorized current run or `None`.

        Notes:
            Scope is never broadened after a miss.
        """
        _nonempty("run_id", run_id)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_runs WHERE run_id = ?", (run_id,)
        )
        if not rows:
            return None
        record = _run(rows[0])
        return record if storage_scope_matches_filter(record.scope, scope) else None

    async def compare_and_set(
        self,
        record: RunRecord,
        expected_revision: int,
    ) -> RunRecord:
        """Atomically replace mutable run state at the exact next revision.

        Identity, creation fields, artifact counters, and result availability remain
        provider-owned and cannot be rewritten through general run CAS.

        Examples:
            Mark a run waiting:
                ```python
                stored = await runs.compare_and_set(waiting, running.revision)
                ```

            Complete a run:
                ```python
                stored = await runs.compare_and_set(completed, waiting.revision)
                ```

        Args:
            record: Complete canonical next run revision.
            expected_revision: Exact current revision required.

        Returns:
            RunRecord: Newly committed run revision.

        Notes:
            Stale expectations raise `StorageConflictError`.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)

        def commit(connection: sqlite3.Connection) -> RunRecord:
            row = connection.execute(
                "SELECT * FROM local_runs WHERE run_id = ?", (record.run_id,)
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(record.run_id)
            current = _run(row)
            if current.revision != expected_revision:
                raise StorageConflictError(
                    f"Run revision conflict: expected {expected_revision}, found {current.revision}"
                )
            if _run_immutable(current) != _run_immutable(record):
                raise StorageIntegrityError("Run CAS cannot change immutable/provider-owned fields")
            _update_run(connection, record)
            return record

        return await self._database.transaction(commit)

    async def query(self, query: RunQuery) -> Page[RunRecord]:
        """Query one bounded recent-run cursor page.

        Populated canonical scope dimensions plus optional status and kind filters
        execute before descending start-time/run-identity pagination.

        Examples:
            List recent project runs:
                ```python
                page = await runs.query(RunQuery(scope=project_scope))
                ```

            Continue failed runs:
                ```python
                page = await runs.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact scope, filters, and bounded page request.

        Returns:
            Page[RunRecord]: Matching runs and continuation cursor.

        Notes:
            Offset and unbounded list operations are absent.
        """
        clauses, values = _scope_filters(query.scope, alias="r")
        if query.statuses:
            placeholders = ",".join("?" for _ in query.statuses)
            clauses.append(f"r.status IN ({placeholders})")
            values.extend(status.value for status in query.statuses)
        if query.kinds:
            placeholders = ",".join("?" for _ in query.kinds)
            clauses.append(f"r.kind IN ({placeholders})")
            values.extend(query.kinds)
        fingerprint = _fingerprint(
            "run",
            _scope_json(query.scope),
            _json(tuple(status.value for status in query.statuses)),
            _json(query.kinds),
        )
        if query.page.cursor is not None:
            timestamp, identity = _decode_cursor(query.page.cursor, fingerprint)
            clauses.append("(r.started_at < ? OR (r.started_at = ? AND r.run_id < ?))")
            values.extend((timestamp, timestamp, identity))
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            "SELECT r.* FROM local_runs AS r WHERE "
            + " AND ".join(clauses)
            + " ORDER BY r.started_at DESC, r.run_id DESC LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        return Page(
            items=tuple(_run(row) for row in selected),
            next_cursor=(
                _encode_cursor(
                    fingerprint,
                    str(selected[-1]["started_at"]),
                    str(selected[-1]["run_id"]),
                )
                if len(rows) > query.page.limit
                else None
            ),
        )

    async def record_artifact(
        self,
        scope: StorageScope,
        run_id: str,
        artifact_id: str,
        occurrence_id: str,
        occurred_at: datetime,
    ) -> RunRecord:
        """Atomically count one idempotent authorized run artifact occurrence.

        The occurrence receipt, bounded content preview, count, and first/last
        timestamps commit in one transaction; exact retry returns the current run.

        Examples:
            Count produced content:
                ```python
                run = await runs.record_artifact(
                    scope, run_id, artifact_id, occurrence_id, now
                )
                ```

            Retry the receipt:
                ```python
                assert await runs.record_artifact(
                    scope, run_id, artifact_id, occurrence_id, now
                ) == run
                ```

        Args:
            scope: Canonical caller scope constraining the run.
            run_id: Exact stable run identity.
            artifact_id: Stable content identity retained in the bounded preview.
            occurrence_id: Stable artifact occurrence identity.
            occurred_at: Exact UTC occurrence timestamp.

        Returns:
            RunRecord: Current run after idempotent counting.

        Notes:
            Reusing an occurrence with different content or time is an integrity error.
        """
        self._require_writable()
        _nonempty("run_id", run_id)
        _nonempty("artifact_id", artifact_id)
        _nonempty("occurrence_id", occurrence_id)
        _utc(occurred_at)

        def commit(connection: sqlite3.Connection) -> RunRecord:
            return _record_run_artifact_in_transaction(
                connection,
                scope=scope,
                run_id=run_id,
                artifact_id=artifact_id,
                occurrence_id=occurrence_id,
                occurred_at=occurred_at,
            )

        return await self._database.transaction(commit)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local run repository is read-only")


class LocalRunResultRepository:
    """Canonical successful outputs coordinated with owning run metadata."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def compare_and_set(
        self,
        record: RunResultRecord,
        expected_revision: int,
    ) -> RunResultRecord:
        """Atomically create or advance a result and its owning run marker.

        Result persistence and the run's availability timestamp/revision are one
        control-database transaction.

        Examples:
            Create final output:
                ```python
                stored = await results.compare_and_set(result, 0)
                ```

            Refine final output:
                ```python
                stored = await results.compare_and_set(next_result, result.revision)
                ```

        Args:
            record: Complete canonical next successful result.
            expected_revision: Required current result revision, or zero for creation.

        Returns:
            RunResultRecord: Newly committed durable result.

        Notes:
            The owning run must already be successful and in the exact same scope.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)

        def commit(connection: sqlite3.Connection) -> RunResultRecord:
            run_row = connection.execute(
                "SELECT * FROM local_runs WHERE run_id = ?", (record.run_id,)
            ).fetchone()
            if run_row is None:
                raise StorageNotFoundError(record.run_id)
            run = _run(run_row)
            if run.scope != record.scope or run.graph_id != record.graph_id:
                raise StorageNotFoundError(record.run_id)
            if run.status is not RunStatus.SUCCEEDED:
                raise StorageIntegrityError("Run result requires an already successful run")
            existing = connection.execute(
                "SELECT * FROM local_run_results WHERE run_id = ?", (record.run_id,)
            ).fetchone()
            current_revision = int(existing["revision"]) if existing is not None else 0
            if current_revision != expected_revision:
                raise StorageConflictError(
                    "Run result revision conflict: "
                    f"expected {expected_revision}, found {current_revision}"
                )
            if existing is not None:
                current = _result(existing, run.scope)
                if _result_immutable(current) != _result_immutable(record):
                    raise StorageIntegrityError("Run result CAS cannot change immutable fields")
                if record.updated_at < current.updated_at:
                    raise StorageIntegrityError("Run result updated_at must be monotonic")
            elif record.created_at < run.started_at:
                raise StorageIntegrityError("Run result cannot precede its owning run")
            connection.execute(
                """
                INSERT INTO local_run_results(
                    run_id, graph_id, status, outputs_json, revision, created_at,
                    updated_at, source, snapshot_revision, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    outputs_json = excluded.outputs_json,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at,
                    source = excluded.source,
                    snapshot_revision = excluded.snapshot_revision,
                    schema_version = excluded.schema_version
                """,
                (
                    record.run_id,
                    record.graph_id,
                    record.status.value,
                    _json(record.outputs),
                    record.revision,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.source,
                    record.snapshot_revision,
                    record.schema_version,
                ),
            )
            run = _replace_run_result(run, record.updated_at)
            _update_run(connection, run)
            return record

        return await self._database.transaction(commit)

    async def get(
        self,
        scope: StorageScope,
        run_id: str,
    ) -> RunResultRecord | None:
        """Read one exact authorized current durable result.

        Authorization uses the owning run's canonical scope rather than duplicated
        result-scope columns.

        Examples:
            Read final output:
                ```python
                result = await results.get(scope, "run-1")
                ```

            Detect no authorized output:
                ```python
                assert await results.get(other_scope, "run-1") is None
                ```

        Args:
            scope: Canonical caller scope constraints.
            run_id: Exact stable run identity.

        Returns:
            RunResultRecord | None: Authorized current result or `None`.

        Notes:
            Outputs are never inferred from run metadata.
        """
        _nonempty("run_id", run_id)
        rows = await self._database.fetch_all(
            """
            SELECT rr.*, r.tenant_id, r.project_id, r.org_id, r.user_id,
                   r.session_id, r.node_id, r.agent_id, r.scope_key
            FROM local_run_results AS rr
            JOIN local_runs AS r ON r.run_id = rr.run_id
            WHERE rr.run_id = ?
            """,
            (run_id,),
        )
        if not rows:
            return None
        run_scope = _run_scope(rows[0])
        return (
            _result(rows[0], run_scope) if storage_scope_matches_filter(run_scope, scope) else None
        )

    async def delete(
        self,
        scope: StorageScope,
        run_id: str,
        expected_revision: int,
    ) -> bool:
        """Atomically delete an authorized result and clear its run marker.

        Result removal and the owning run revision/availability update share one
        control-database transaction.

        Examples:
            Delete a corrupt output:
                ```python
                deleted = await results.delete(scope, run_id, result.revision)
                ```

            Detect an absent output:
                ```python
                assert not await results.delete(scope, "missing", 1)
                ```

        Args:
            scope: Canonical caller scope constraints.
            run_id: Exact stable run identity.
            expected_revision: Exact current result revision required.

        Returns:
            bool: `True` when deleted, or `False` when absent or unauthorized.

        Notes:
            Authorization is checked before revision comparison. Stale authorized
            expectations raise `StorageConflictError`.
        """
        self._require_writable()
        _delete_revision(expected_revision)
        _nonempty("run_id", run_id)

        def commit(connection: sqlite3.Connection) -> bool:
            run_row = connection.execute(
                "SELECT * FROM local_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run_row is None:
                return False
            run = _run(run_row)
            if not storage_scope_matches_filter(run.scope, scope):
                return False
            result_row = connection.execute(
                "SELECT revision FROM local_run_results WHERE run_id = ?", (run_id,)
            ).fetchone()
            if result_row is None:
                return False
            current_revision = int(result_row["revision"])
            if current_revision != expected_revision:
                raise StorageConflictError(
                    "Run result revision conflict: "
                    f"expected {expected_revision}, found {current_revision}"
                )
            connection.execute("DELETE FROM local_run_results WHERE run_id = ?", (run_id,))
            _update_run(connection, _clear_run_result(run))
            return True

        return await self._database.transaction(commit)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local run result repository is read-only")


class LocalSessionRepository:
    """Canonical local sessions with revision CAS and artifact receipts."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def create(self, record: SessionRecord) -> SessionRecord:
        """Idempotently create one initial canonical session.

        Initial artifact state must be empty; exact retries return the stored session
        and conflicting session identity fails atomically.

        Examples:
            Create a chat session:
                ```python
                stored = await sessions.create(record)
                ```

            Retry creation:
                ```python
                assert await sessions.create(record) == stored
                ```

        Args:
            record: Complete initial session at revision one.

        Returns:
            SessionRecord: Authoritative stored session.

        Notes:
            Deprecated App identity is absent from storage identity.
        """
        self._require_writable()
        if record.revision != 1 or record.artifact_count:
            raise StorageIntegrityError("Initial session revision/counter state is invalid")

        def commit(connection: sqlite3.Connection) -> SessionRecord:
            existing = connection.execute(
                "SELECT * FROM local_sessions WHERE session_id = ?", (record.session_id,)
            ).fetchone()
            if existing is not None:
                stored = _session(existing)
                if stored != record:
                    raise StorageIntegrityError(f"Session identity {record.session_id!r} conflicts")
                return stored
            _insert_session(connection, record)
            return record

        return await self._database.transaction(commit)

    async def get(
        self,
        scope: StorageScope,
        session_id: str,
    ) -> SessionRecord | None:
        """Read one exact authorized current session.

        Every populated caller scope dimension must match the stored session scope.

        Examples:
            Read a session:
                ```python
                session = await sessions.get(scope, "session-1")
                ```

            Detect a cross-scope identity:
                ```python
                assert await sessions.get(other_scope, "session-1") is None
                ```

        Args:
            scope: Canonical caller scope constraints.
            session_id: Exact stable session identity.

        Returns:
            SessionRecord | None: Authorized current session or `None`.

        Notes:
            Scope is never broadened after a miss.
        """
        _nonempty("session_id", session_id)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_sessions WHERE session_id = ?", (session_id,)
        )
        if not rows:
            return None
        record = _session(rows[0])
        return record if storage_scope_matches_filter(record.scope, scope) else None

    async def compare_and_set(
        self,
        record: SessionRecord,
        expected_revision: int,
    ) -> SessionRecord:
        """Atomically replace mutable session state at the exact next revision.

        Creation identity and provider-owned artifact counters cannot be changed by
        the general session CAS operation.

        Examples:
            Rename a session:
                ```python
                stored = await sessions.compare_and_set(renamed, current.revision)
                ```

            Update metadata:
                ```python
                stored = await sessions.compare_and_set(updated, current.revision)
                ```

        Args:
            record: Complete canonical next session revision.
            expected_revision: Exact current revision required.

        Returns:
            SessionRecord: Newly committed session revision.

        Notes:
            Stale expectations raise `StorageConflictError`.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)

        def commit(connection: sqlite3.Connection) -> SessionRecord:
            row = connection.execute(
                "SELECT * FROM local_sessions WHERE session_id = ?", (record.session_id,)
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(record.session_id)
            current = _session(row)
            if current.revision != expected_revision:
                raise StorageConflictError(
                    "Session revision conflict: "
                    f"expected {expected_revision}, found {current.revision}"
                )
            if _session_immutable(current) != _session_immutable(record):
                raise StorageIntegrityError(
                    "Session CAS cannot change immutable/provider-owned fields"
                )
            if record.updated_at < current.updated_at:
                raise StorageIntegrityError("Session updated_at must be monotonic")
            _update_session(connection, record)
            return record

        return await self._database.transaction(commit)

    async def delete(
        self,
        scope: StorageScope,
        session_id: str,
        expected_revision: int,
    ) -> bool:
        """Delete one authorized current session using revision CAS.

        SQLite foreign-key cascading removes only the deleted session's provider-
        owned artifact occurrence receipts.

        Examples:
            Delete a session:
                ```python
                deleted = await sessions.delete(scope, session_id, session.revision)
                ```

            Detect an absent session:
                ```python
                assert not await sessions.delete(scope, "missing", 1)
                ```

        Args:
            scope: Canonical caller scope constraints.
            session_id: Exact stable session identity.
            expected_revision: Exact current session revision required.

        Returns:
            bool: `True` when deleted, or `False` when absent or unauthorized.

        Notes:
            Authorization is checked before revision comparison. Stale authorized
            expectations raise `StorageConflictError`.
        """
        self._require_writable()
        _delete_revision(expected_revision)
        _nonempty("session_id", session_id)

        def commit(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                "SELECT * FROM local_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return False
            current = _session(row)
            if not storage_scope_matches_filter(current.scope, scope):
                return False
            if current.revision != expected_revision:
                raise StorageConflictError(
                    "Session revision conflict: "
                    f"expected {expected_revision}, found {current.revision}"
                )
            connection.execute("DELETE FROM local_sessions WHERE session_id = ?", (session_id,))
            return True

        return await self._database.transaction(commit)

    async def query(self, query: SessionQuery) -> Page[SessionRecord]:
        """Query one bounded recent-session cursor page.

        Populated canonical scope dimensions and optional kind filters execute before
        descending update-time/session-identity pagination.

        Examples:
            List project sessions:
                ```python
                page = await sessions.query(SessionQuery(scope=project_scope))
                ```

            Continue chat sessions:
                ```python
                page = await sessions.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact scope, kind filters, and bounded page request.

        Returns:
            Page[SessionRecord]: Matching sessions and continuation cursor.

        Notes:
            Offset and unbounded list operations are absent.
        """
        clauses, values = _scope_filters(query.scope, alias="s")
        if query.kinds:
            placeholders = ",".join("?" for _ in query.kinds)
            clauses.append(f"s.kind IN ({placeholders})")
            values.extend(kind.value for kind in query.kinds)
        fingerprint = _fingerprint(
            "session",
            _scope_json(query.scope),
            _json(tuple(kind.value for kind in query.kinds)),
        )
        if query.page.cursor is not None:
            timestamp, identity = _decode_cursor(query.page.cursor, fingerprint)
            clauses.append("(s.updated_at < ? OR (s.updated_at = ? AND s.session_id < ?))")
            values.extend((timestamp, timestamp, identity))
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            "SELECT s.* FROM local_sessions AS s WHERE "
            + " AND ".join(clauses)
            + " ORDER BY s.updated_at DESC, s.session_id DESC LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        return Page(
            items=tuple(_session(row) for row in selected),
            next_cursor=(
                _encode_cursor(
                    fingerprint,
                    str(selected[-1]["updated_at"]),
                    str(selected[-1]["session_id"]),
                )
                if len(rows) > query.page.limit
                else None
            ),
        )

    async def record_artifact(
        self,
        scope: StorageScope,
        session_id: str,
        occurrence_id: str,
        occurred_at: datetime,
    ) -> SessionRecord:
        """Atomically count one idempotent authorized session artifact occurrence.

        The receipt, count, last-artifact time, updated time, and revision commit in
        one transaction; retry returns current state without incrementing.

        Examples:
            Count an attachment:
                ```python
                session = await sessions.record_artifact(scope, session_id, occurrence_id, now)
                ```

            Retry the receipt:
                ```python
                assert await sessions.record_artifact(scope, session_id, occurrence_id, now) == session
                ```

        Args:
            scope: Canonical caller scope constraining the session.
            session_id: Exact stable session identity.
            occurrence_id: Stable artifact occurrence identity.
            occurred_at: Exact UTC occurrence timestamp.

        Returns:
            SessionRecord: Current session after idempotent counting.

        Notes:
            Reusing an occurrence identity with another timestamp is an integrity error.
        """
        self._require_writable()
        _nonempty("session_id", session_id)
        _nonempty("occurrence_id", occurrence_id)
        _utc(occurred_at)

        def commit(connection: sqlite3.Connection) -> SessionRecord:
            return _record_session_artifact_in_transaction(
                connection,
                scope=scope,
                session_id=session_id,
                occurrence_id=occurrence_id,
                occurred_at=occurred_at,
            )

        return await self._database.transaction(commit)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local session repository is read-only")


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.CONTROL:
        raise StorageConfigurationError("Local control repositories require control database")
    database.install_component(
        name="control",
        version=_CONTROL_COMPONENT_VERSION,
        statements=(
            _CREATE_RUNS,
            _CREATE_RUN_PROJECT_INDEX,
            _CREATE_RUN_SESSION_INDEX,
            _CREATE_RUN_GRAPH_INDEX,
            _CREATE_RUN_STATUS_INDEX,
            _CREATE_RUN_KIND_INDEX,
            _CREATE_RUN_ARTIFACT_OCCURRENCES,
            _CREATE_RUN_RESULTS,
            _CREATE_SESSIONS,
            _CREATE_SESSION_PROJECT_INDEX,
            _CREATE_SESSION_USER_INDEX,
            _CREATE_SESSION_KIND_INDEX,
            _CREATE_SESSION_ARTIFACT_OCCURRENCES,
        ),
    )


def _record_run_artifact_in_transaction(
    connection: sqlite3.Connection,
    *,
    scope: StorageScope,
    run_id: str,
    artifact_id: str,
    occurrence_id: str,
    occurred_at: datetime,
) -> RunRecord:
    row = connection.execute("SELECT * FROM local_runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise StorageNotFoundError(run_id)
    current = _run(row)
    if not storage_scope_covers(current.scope, scope):
        raise StorageNotFoundError(run_id)
    receipt = connection.execute(
        """
        SELECT artifact_id, occurred_at FROM local_run_artifact_occurrences
        WHERE run_id = ? AND occurrence_id = ?
        """,
        (run_id, occurrence_id),
    ).fetchone()
    if receipt is not None:
        if (
            str(receipt["artifact_id"]) != artifact_id
            or str(receipt["occurred_at"]) != occurred_at.isoformat()
        ):
            raise StorageIntegrityError("Run artifact occurrence identity conflicts")
        return current
    connection.execute(
        """
        INSERT INTO local_run_artifact_occurrences(
            run_id, occurrence_id, artifact_id, occurred_at
        ) VALUES (?, ?, ?, ?)
        """,
        (run_id, occurrence_id, artifact_id, occurred_at.isoformat()),
    )
    updated = _run_with_artifact(current, artifact_id, occurred_at)
    _update_run(connection, updated)
    return updated


def _record_session_artifact_in_transaction(
    connection: sqlite3.Connection,
    *,
    scope: StorageScope,
    session_id: str,
    occurrence_id: str,
    occurred_at: datetime,
) -> SessionRecord:
    row = connection.execute(
        "SELECT * FROM local_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise StorageNotFoundError(session_id)
    current = _session(row)
    if not storage_scope_covers(current.scope, scope):
        raise StorageNotFoundError(session_id)
    receipt = connection.execute(
        """
        SELECT occurred_at FROM local_session_artifact_occurrences
        WHERE session_id = ? AND occurrence_id = ?
        """,
        (session_id, occurrence_id),
    ).fetchone()
    if receipt is not None:
        if str(receipt[0]) != occurred_at.isoformat():
            raise StorageIntegrityError("Session artifact occurrence identity conflicts")
        return current
    connection.execute(
        """
        INSERT INTO local_session_artifact_occurrences(
            session_id, occurrence_id, occurred_at
        ) VALUES (?, ?, ?)
        """,
        (session_id, occurrence_id, occurred_at.isoformat()),
    )
    updated = _session_with_artifact(current, occurred_at)
    _update_session(connection, updated)
    return updated


def _insert_run(connection: sqlite3.Connection, record: RunRecord) -> None:
    connection.execute(
        """
        INSERT INTO local_runs(
            run_id, graph_id, tenant_id, project_id, org_id, user_id, session_id,
            node_id, agent_id, scope_key, kind, status, revision, started_at,
            finished_at, tags_json, error, metadata_json, artifact_count,
            first_artifact_at, last_artifact_at, recent_artifact_ids_json,
            result_available, result_updated_at, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _run_values(record),
    )


def _update_run(connection: sqlite3.Connection, record: RunRecord) -> None:
    values = _run_values(record)
    connection.execute(
        """
        UPDATE local_runs SET
            graph_id = ?, tenant_id = ?, project_id = ?, org_id = ?, user_id = ?,
            session_id = ?, node_id = ?, agent_id = ?, scope_key = ?, kind = ?,
            status = ?, revision = ?, started_at = ?, finished_at = ?, tags_json = ?,
            error = ?, metadata_json = ?, artifact_count = ?, first_artifact_at = ?,
            last_artifact_at = ?, recent_artifact_ids_json = ?, result_available = ?,
            result_updated_at = ?, schema_version = ?
        WHERE run_id = ?
        """,
        (*values[1:], values[0]),
    )


def _run_values(record: RunRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.graph_id,
        record.scope.tenant_id,
        record.scope.project_id,
        record.scope.org_id,
        record.scope.user_id,
        record.scope.session_id,
        record.scope.node_id,
        record.scope.agent_id,
        record.scope.scope_key,
        record.kind,
        record.status.value,
        record.revision,
        record.started_at.isoformat(),
        record.finished_at.isoformat() if record.finished_at else None,
        _json(record.tags),
        record.error,
        _json(record.metadata),
        record.artifact_count,
        record.first_artifact_at.isoformat() if record.first_artifact_at else None,
        record.last_artifact_at.isoformat() if record.last_artifact_at else None,
        _json(record.recent_artifact_ids),
        int(record.result_available),
        record.result_updated_at.isoformat() if record.result_updated_at else None,
        record.schema_version,
    )


def _run(row: sqlite3.Row) -> RunRecord:
    try:
        return RunRecord(
            run_id=str(row["run_id"]),
            graph_id=str(row["graph_id"]),
            kind=str(row["kind"]),
            status=RunStatus(str(row["status"])),
            scope=_run_scope(row),
            revision=int(row["revision"]),
            started_at=datetime.fromisoformat(str(row["started_at"])),
            finished_at=_optional_time(row["finished_at"]),
            tags=tuple(_json_list(row["tags_json"], "run tags")),
            error=row["error"],
            metadata=_json_mapping(row["metadata_json"], "run metadata"),
            artifact_count=int(row["artifact_count"]),
            first_artifact_at=_optional_time(row["first_artifact_at"]),
            last_artifact_at=_optional_time(row["last_artifact_at"]),
            recent_artifact_ids=tuple(
                str(value)
                for value in _json_list(row["recent_artifact_ids_json"], "recent artifact ids")
            ),
            result_available=bool(row["result_available"]),
            result_updated_at=_optional_time(row["result_updated_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local run row is malformed") from exc


def _run_scope(row: sqlite3.Row) -> StorageScope:
    try:
        return StorageScope(
            tenant_id=row["tenant_id"],
            project_id=row["project_id"],
            org_id=row["org_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            run_id=str(row["run_id"]),
            graph_id=str(row["graph_id"]),
            node_id=row["node_id"],
            agent_id=row["agent_id"],
            scope_key=row["scope_key"],
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted local run scope is malformed") from exc


def _run_immutable(record: RunRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.graph_id,
        record.kind,
        record.scope,
        record.started_at,
        record.artifact_count,
        record.first_artifact_at,
        record.last_artifact_at,
        record.recent_artifact_ids,
        record.result_available,
        record.result_updated_at,
    )


def _run_with_artifact(
    record: RunRecord,
    artifact_id: str,
    occurred_at: datetime,
) -> RunRecord:
    from dataclasses import replace

    return replace(
        record,
        revision=record.revision + 1,
        artifact_count=record.artifact_count + 1,
        first_artifact_at=(
            min(record.first_artifact_at, occurred_at) if record.first_artifact_at else occurred_at
        ),
        last_artifact_at=(
            max(record.last_artifact_at, occurred_at) if record.last_artifact_at else occurred_at
        ),
        recent_artifact_ids=(*record.recent_artifact_ids, artifact_id)[-10:],
    )


def _replace_run_result(record: RunRecord, updated_at: datetime) -> RunRecord:
    from dataclasses import replace

    return replace(
        record,
        revision=record.revision + 1,
        result_available=True,
        result_updated_at=updated_at,
    )


def _clear_run_result(record: RunRecord) -> RunRecord:
    from dataclasses import replace

    return replace(
        record,
        revision=record.revision + 1,
        result_available=False,
        result_updated_at=None,
    )


def _result(row: sqlite3.Row, scope: StorageScope) -> RunResultRecord:
    try:
        return RunResultRecord(
            run_id=str(row["run_id"]),
            graph_id=str(row["graph_id"]),
            scope=scope,
            status=RunStatus(str(row["status"])),
            outputs=json.loads(row["outputs_json"]),
            revision=int(row["revision"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            source=str(row["source"]),
            snapshot_revision=row["snapshot_revision"],
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local run result row is malformed") from exc


def _result_immutable(record: RunResultRecord) -> tuple[object, ...]:
    return (
        record.run_id,
        record.graph_id,
        record.scope,
        record.status,
        record.created_at,
        record.source,
    )


def _insert_session(connection: sqlite3.Connection, record: SessionRecord) -> None:
    connection.execute(
        """
        INSERT INTO local_sessions(
            session_id, tenant_id, project_id, org_id, user_id, run_id, graph_id,
            node_id, agent_id, scope_key, kind, revision, created_at, updated_at,
            title, source, external_reference, metadata_json, artifact_count,
            last_artifact_at, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _session_values(record),
    )


def _update_session(connection: sqlite3.Connection, record: SessionRecord) -> None:
    values = _session_values(record)
    connection.execute(
        """
        UPDATE local_sessions SET
            tenant_id = ?, project_id = ?, org_id = ?, user_id = ?, run_id = ?,
            graph_id = ?, node_id = ?, agent_id = ?, scope_key = ?, kind = ?,
            revision = ?, created_at = ?, updated_at = ?, title = ?, source = ?,
            external_reference = ?, metadata_json = ?, artifact_count = ?,
            last_artifact_at = ?, schema_version = ?
        WHERE session_id = ?
        """,
        (*values[1:], values[0]),
    )


def _session_values(record: SessionRecord) -> tuple[object, ...]:
    return (
        record.session_id,
        record.scope.tenant_id,
        record.scope.project_id,
        record.scope.org_id,
        record.scope.user_id,
        record.scope.run_id,
        record.scope.graph_id,
        record.scope.node_id,
        record.scope.agent_id,
        record.scope.scope_key,
        record.kind.value,
        record.revision,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
        record.title,
        record.source,
        record.external_reference,
        _json(record.metadata),
        record.artifact_count,
        record.last_artifact_at.isoformat() if record.last_artifact_at else None,
        record.schema_version,
    )


def _session(row: sqlite3.Row) -> SessionRecord:
    try:
        return SessionRecord(
            session_id=str(row["session_id"]),
            kind=SessionKind(str(row["kind"])),
            scope=StorageScope(
                tenant_id=row["tenant_id"],
                project_id=row["project_id"],
                org_id=row["org_id"],
                user_id=row["user_id"],
                session_id=str(row["session_id"]),
                run_id=row["run_id"],
                graph_id=row["graph_id"],
                node_id=row["node_id"],
                agent_id=row["agent_id"],
                scope_key=row["scope_key"],
            ),
            revision=int(row["revision"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            title=row["title"],
            source=str(row["source"]),
            external_reference=row["external_reference"],
            metadata=_json_mapping(row["metadata_json"], "session metadata"),
            artifact_count=int(row["artifact_count"]),
            last_artifact_at=_optional_time(row["last_artifact_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local session row is malformed") from exc


def _session_immutable(record: SessionRecord) -> tuple[object, ...]:
    return (
        record.session_id,
        record.kind,
        record.scope,
        record.created_at,
        record.source,
        record.artifact_count,
        record.last_artifact_at,
    )


def _session_with_artifact(record: SessionRecord, occurred_at: datetime) -> SessionRecord:
    from dataclasses import replace

    return replace(
        record,
        revision=record.revision + 1,
        updated_at=max(record.updated_at, occurred_at),
        artifact_count=record.artifact_count + 1,
        last_artifact_at=(
            max(record.last_artifact_at, occurred_at) if record.last_artifact_at else occurred_at
        ),
    )


def _scope_filters(
    scope: StorageScope,
    *,
    alias: str,
) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    values: list[object] = []
    for name, value in scope.as_filter().items():
        clauses.append(f"{alias}.{name} = ?")
        values.append(value)
    if not clauses:
        raise StorageConfigurationError("Control repository queries require populated scope")
    return clauses, values


def _next_revision(revision: int, expected_revision: int) -> None:
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
        raise ValueError("expected_revision must be an integer")
    if expected_revision < 0 or revision != expected_revision + 1:
        raise ValueError("record revision must equal expected_revision plus one")


def _delete_revision(expected_revision: int) -> None:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 1
    ):
        raise ValueError("expected_revision must be a positive integer")


def _optional_time(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value is not None else None


def _utc(value: datetime) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset().total_seconds() != 0
    ):
        raise ValueError("occurred_at must be timezone-aware UTC")


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StorageConfigurationError(f"{name} must be a non-empty string")


def _scope_json(scope: StorageScope) -> str:
    return _json(scope.as_filter())


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _json_mapping(value: object, label: str) -> dict[str, object]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} must be an object")
    return parsed


def _json_list(value: object, label: str) -> list[object]:
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise TypeError(f"{label} must be an array")
    return parsed


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:24]


def _encode_cursor(fingerprint: str, timestamp: str, identity: str) -> str:
    payload = json.dumps(
        {"fingerprint": fingerprint, "timestamp": timestamp, "identity": identity},
        sort_keys=True,
        separators=(",", ":"),
    )
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
        raise StorageConfigurationError("Invalid or mismatched control cursor") from exc

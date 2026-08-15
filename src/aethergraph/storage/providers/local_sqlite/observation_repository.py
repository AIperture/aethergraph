"""Transactional local observations, LLM captures, and retention controls."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import sqlite3
from typing import Any

from ...contracts import (
    LLMCallAttempt,
    LLMCallDetail,
    LLMCallDraft,
    LLMCallQuery,
    LLMCallRecord,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationPurgeRequest,
    ObservationPurgeResult,
    ObservationQuery,
    ObservationRecord,
    ObservationResourceLink,
    ObservationResourceRelation,
    ObservationScopeManagementRecord,
    ObservationSeverity,
    ObservationStatus,
    ObservationStorageStats,
    Page,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
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
_CREATE_OBSERVATIONS = """
CREATE TABLE local_observations (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id TEXT NOT NULL UNIQUE,
    scope_identity TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    run_id TEXT,
    graph_id TEXT,
    node_id TEXT,
    agent_id TEXT,
    scope_key TEXT,
    category TEXT NOT NULL,
    name TEXT NOT NULL,
    summary TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    status TEXT NOT NULL,
    severity TEXT NOT NULL,
    trace_id TEXT,
    turn_id TEXT,
    parent_observation_id TEXT,
    caused_by_observation_id TEXT,
    source_event_id TEXT,
    attributes_json TEXT NOT NULL,
    payload_fragment_id TEXT,
    retention_class TEXT NOT NULL,
    expires_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    content_digest TEXT NOT NULL
)
"""
_CREATE_OBSERVATION_PROJECT_INDEX = """
CREATE INDEX ix_local_observations_project_time
ON local_observations(tenant_id, project_id, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_RUN_INDEX = """
CREATE INDEX ix_local_observations_run_time
ON local_observations(run_id, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_SESSION_INDEX = """
CREATE INDEX ix_local_observations_session_time
ON local_observations(session_id, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_GRAPH_INDEX = """
CREATE INDEX ix_local_observations_graph_time
ON local_observations(graph_id, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_TRACE_INDEX = """
CREATE INDEX ix_local_observations_trace_time
ON local_observations(trace_id, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_CATEGORY_INDEX = """
CREATE INDEX ix_local_observations_category_time
ON local_observations(category, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_STATUS_INDEX = """
CREATE INDEX ix_local_observations_status_time
ON local_observations(status, occurred_at DESC, sequence DESC)
"""
_CREATE_OBSERVATION_SEVERITY_INDEX = """
CREATE INDEX ix_local_observations_severity_time
ON local_observations(severity, occurred_at DESC, sequence DESC)
"""
_CREATE_RESOURCE_LINKS = """
CREATE TABLE local_observation_resource_links (
    observation_id TEXT NOT NULL
        REFERENCES local_observations(observation_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    resource_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    resource_revision TEXT,
    content_hash TEXT,
    slot_key TEXT,
    PRIMARY KEY(observation_id, resource_key, relation),
    UNIQUE(observation_id, ordinal)
)
"""
_CREATE_RESOURCE_LOOKUP_INDEX = """
CREATE INDEX ix_local_observation_resources_lookup
ON local_observation_resource_links(resource_key, relation, observation_id)
"""
_CREATE_FRAGMENTS = """
CREATE TABLE local_observation_fragments (
    fragment_id TEXT PRIMARY KEY,
    content_kind TEXT NOT NULL,
    canonical_hash TEXT NOT NULL,
    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
    body_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""
_CREATE_MANIFESTS = """
CREATE TABLE local_observation_manifests (
    manifest_id TEXT PRIMARY KEY,
    capture_mode TEXT NOT NULL,
    request_fragment_id TEXT
        REFERENCES local_observation_fragments(fragment_id) ON DELETE RESTRICT,
    trace_fragment_id TEXT
        REFERENCES local_observation_fragments(fragment_id) ON DELETE RESTRICT,
    created_at TEXT NOT NULL
)
"""
_CREATE_LLM_CALLS = """
CREATE TABLE local_llm_calls (
    llm_call_id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL UNIQUE
        REFERENCES local_observations(observation_id) ON DELETE CASCADE,
    call_type TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    capture_mode TEXT NOT NULL,
    profile_name TEXT,
    call_name TEXT,
    request_options_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    latency_ms INTEGER,
    error_type TEXT,
    error_message TEXT,
    prompt_manifest_id TEXT
        REFERENCES local_observation_manifests(manifest_id) ON DELETE RESTRICT,
    request_preview_json TEXT NOT NULL,
    response_preview_json TEXT NOT NULL,
    trace_payload_preview_json TEXT NOT NULL,
    response_fragment_id TEXT
        REFERENCES local_observation_fragments(fragment_id) ON DELETE RESTRICT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    content_digest TEXT NOT NULL
)
"""
_CREATE_LLM_PROVIDER_INDEX = """
CREATE INDEX ix_local_llm_calls_provider ON local_llm_calls(provider, observation_id)
"""
_CREATE_LLM_MODEL_INDEX = """
CREATE INDEX ix_local_llm_calls_model ON local_llm_calls(model, observation_id)
"""
_CREATE_LLM_TYPE_INDEX = """
CREATE INDEX ix_local_llm_calls_type ON local_llm_calls(call_type, observation_id)
"""
_CREATE_LLM_MANIFEST_INDEX = """
CREATE INDEX ix_local_llm_calls_manifest ON local_llm_calls(prompt_manifest_id)
"""
_CREATE_ATTEMPTS = """
CREATE TABLE local_llm_attempts (
    llm_call_id TEXT NOT NULL
        REFERENCES local_llm_calls(llm_call_id) ON DELETE CASCADE,
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    elapsed_ms INTEGER NOT NULL CHECK (elapsed_ms >= 0),
    outcome TEXT NOT NULL,
    retryable INTEGER NOT NULL CHECK (retryable IN (0, 1)),
    status_code INTEGER,
    error_code TEXT,
    request_id TEXT,
    provider_delay_ms INTEGER,
    scheduled_delay_ms INTEGER,
    rate_limits_json TEXT NOT NULL,
    PRIMARY KEY(llm_call_id, attempt_number)
)
"""
_CREATE_MANAGEMENT = """
CREATE TABLE local_observation_scope_management (
    scope_identity TEXT NOT NULL,
    management_key TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    run_id TEXT,
    graph_id TEXT,
    node_id TEXT,
    agent_id TEXT,
    canonical_scope_key TEXT,
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    trace_id TEXT,
    pinned INTEGER NOT NULL CHECK (pinned IN (0, 1)),
    hidden INTEGER NOT NULL CHECK (hidden IN (0, 1)),
    deleted INTEGER NOT NULL CHECK (deleted IN (0, 1)),
    label TEXT,
    tags_json TEXT NOT NULL,
    retention_class TEXT NOT NULL,
    expires_at TEXT,
    PRIMARY KEY(scope_identity, management_key)
)
"""
_CREATE_MANAGEMENT_TRACE_INDEX = """
CREATE INDEX ix_local_observation_management_trace
ON local_observation_scope_management(trace_id, pinned, scope_identity)
"""


class LocalObservationRepository:
    """Canonical local observations, LLM details, and retention policy."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def append_many(
        self, observations: tuple[ObservationDraft, ...]
    ) -> tuple[ObservationRecord, ...]:
        """Atomically append ordered observations and normalized resource links.

        Provider cursors are assigned to new rows in input order. Exact identity
        retries return existing rows while one conflict rolls back the entire batch.

        Examples:
            Append one observation:
                ```python
                stored, = await repository.append_many((draft,))
                ```

            Append a span batch:
                ```python
                stored = await repository.append_many((started, finished))
                ```

        Args:
            observations: Non-empty immutable batch in required append order.

        Returns:
            tuple[ObservationRecord, ...]: Authoritative records in input order.

        Notes:
            Resource links commit with their observations; partial append is forbidden.
        """
        self._require_writable()
        if not isinstance(observations, tuple) or not observations:
            raise StorageConfigurationError("Observation append batch must be non-empty tuple")

        def commit(connection: sqlite3.Connection) -> tuple[ObservationRecord, ...]:
            return tuple(_append_observation(connection, draft) for draft in observations)

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, observation_id: str) -> ObservationRecord | None:
        """Read one exact authorized observation without LLM prompt hydration.

        The record and normalized links are reconstructed only after canonical scope
        authorization succeeds.

        Examples:
            Read an observation:
                ```python
                observation = await repository.get(scope, "obs-1")
                ```

            Detect absence:
                ```python
                assert await repository.get(scope, "missing") is None
                ```

        Args:
            scope: Populated canonical scope constraining access.
            observation_id: Exact stable observation identity.

        Returns:
            ObservationRecord | None: Authorized record or `None`.

        Notes:
            Captured LLM request and response content is never hydrated here.
        """
        _nonempty("observation_id", observation_id)
        if not scope.as_filter():
            return None
        clauses, values = _scope_filters(scope, alias="o")

        def read(connection: sqlite3.Connection) -> ObservationRecord | None:
            row = connection.execute(
                "SELECT o.* FROM local_observations o WHERE o.observation_id = ? AND "
                + " AND ".join(clauses),
                (observation_id, *values),
            ).fetchone()
            return _load_observation(connection, row) if row is not None else None

        return await self._database.read_transaction(read)

    async def query(self, query: ObservationQuery) -> Page[ObservationRecord]:
        """Query one bounded stable page through promoted observation indexes.

        Scope, category, lifecycle, trace, resource, and time filters execute in SQL
        before descending occurrence/sequence pagination.

        Examples:
            List error observations:
                ```python
                page = await repository.query(ObservationQuery(scope=scope, statuses=(status,)))
                ```

            Follow a resource:
                ```python
                page = await repository.query(ObservationQuery(scope=scope, resource_key=key))
                ```

        Args:
            query: Exact canonical scope, filters, and opaque page request.

        Returns:
            Page[ObservationRecord]: Matching records and optional continuation cursor.

        Notes:
            The cursor is bound to every query filter and the requested page size.
        """
        clauses, values = _scope_filters(query.scope, alias="o")
        if query.categories:
            clauses.append(f"o.category IN ({','.join('?' for _ in query.categories)})")
            values.extend(query.categories)
        if query.statuses:
            clauses.append(f"o.status IN ({','.join('?' for _ in query.statuses)})")
            values.extend(status.value for status in query.statuses)
        if query.severities:
            clauses.append(f"o.severity IN ({','.join('?' for _ in query.severities)})")
            values.extend(severity.value for severity in query.severities)
        for column, value in (("trace_id", query.trace_id), ("turn_id", query.turn_id)):
            if value is not None:
                clauses.append(f"o.{column} = ?")
                values.append(value)
        if query.resource_key is not None:
            resource_clause = (
                "EXISTS (SELECT 1 FROM local_observation_resource_links r "
                "WHERE r.observation_id = o.observation_id AND r.resource_key = ?"
            )
            values.append(query.resource_key)
            if query.resource_relation is not None:
                resource_clause += " AND r.relation = ?"
                values.append(query.resource_relation.value)
            clauses.append(resource_clause + ")")
        if query.occurred_at_or_after is not None:
            clauses.append("o.occurred_at >= ?")
            values.append(query.occurred_at_or_after.isoformat())
        if query.occurred_at_or_before is not None:
            clauses.append("o.occurred_at <= ?")
            values.append(query.occurred_at_or_before.isoformat())
        fingerprint = _observation_query_fingerprint(query)
        if query.page.cursor:
            timestamp, sequence = _decode_cursor(query.page.cursor, fingerprint)
            clauses.append("(o.occurred_at, o.sequence) < (?, ?)")
            values.extend((timestamp, sequence))

        def read(connection: sqlite3.Connection) -> Page[ObservationRecord]:
            rows = connection.execute(
                "SELECT o.* FROM local_observations o "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY o.occurred_at DESC, o.sequence DESC LIMIT ?",
                (*values, query.page.limit + 1),
            ).fetchall()
            visible = rows[: query.page.limit]
            records = _load_observations(connection, visible)
            next_cursor = None
            if len(rows) > query.page.limit:
                anchor = visible[-1]
                next_cursor = _encode_cursor(
                    fingerprint,
                    str(anchor["occurred_at"]),
                    int(anchor["sequence"]),
                )
            return Page(items=records, next_cursor=next_cursor)

        return await self._database.read_transaction(read)

    async def append_llm_call(self, call: LLMCallDraft) -> LLMCallRecord:
        """Atomically append LLM metadata, attempts, observation, and captures.

        Prepared capture content is content-addressed and deduplicated. The method
        returns a metadata-only record suitable for list and inspection summaries.

        Examples:
            Store one call:
                ```python
                record = await repository.append_llm_call(call)
                ```

            Retry exact identity:
                ```python
                assert await repository.append_llm_call(call) == record
                ```

        Args:
            call: Prepared canonical LLM call and policy-approved captured content.

        Returns:
            LLMCallRecord: Authoritative metadata-only call record.

        Notes:
            Conflicting LLM or observation identity rolls back every dependent row.
        """
        self._require_writable()
        digest = _llm_digest(call)

        def commit(connection: sqlite3.Connection) -> LLMCallRecord:
            existing = connection.execute(
                "SELECT * FROM local_llm_calls WHERE llm_call_id = ?", (call.llm_call_id,)
            ).fetchone()
            if existing is not None:
                if str(existing["content_digest"]) != digest:
                    raise StorageIntegrityError(f"LLM call identity {call.llm_call_id!r} conflicts")
                return _load_llm_record(connection, existing)
            if connection.execute(
                "SELECT 1 FROM local_observations WHERE observation_id = ?",
                (call.observation.observation_id,),
            ).fetchone():
                raise StorageIntegrityError("LLM observation identity was not created atomically")
            observation = _append_observation(connection, call.observation)
            request_fragment = _store_fragment(
                connection,
                content_kind="llm_request",
                value=call.captured_request,
                created_at=call.observation.occurred_at,
            )
            trace_fragment = _store_fragment(
                connection,
                content_kind="llm_trace",
                value=call.trace_payload,
                created_at=call.observation.occurred_at,
            )
            response_fragment = _store_fragment(
                connection,
                content_kind="llm_response",
                value=call.captured_response,
                created_at=call.observation.occurred_at,
            )
            if call.prompt_manifest_id is not None:
                _put_manifest(
                    connection,
                    manifest_id=call.prompt_manifest_id,
                    capture_mode=call.capture_mode,
                    request_fragment_id=request_fragment,
                    trace_fragment_id=trace_fragment,
                    created_at=call.observation.occurred_at,
                )
            connection.execute(
                """
                INSERT INTO local_llm_calls(
                    llm_call_id, observation_id, call_type, provider, model,
                    capture_mode, profile_name, call_name, request_options_json,
                    usage_json, latency_ms, error_type, error_message,
                    prompt_manifest_id, request_preview_json, response_preview_json,
                    trace_payload_preview_json, response_fragment_id, schema_version,
                    content_digest
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call.llm_call_id,
                    observation.observation_id,
                    call.call_type,
                    call.provider,
                    call.model,
                    call.capture_mode.value,
                    call.profile_name,
                    call.call_name,
                    _json(call.request_options),
                    _json(call.usage),
                    call.latency_ms,
                    call.error_type,
                    call.error_message,
                    call.prompt_manifest_id,
                    _json(call.request_preview),
                    _json(call.response_preview),
                    _json(None),
                    response_fragment,
                    call.schema_version,
                    digest,
                ),
            )
            for attempt in call.attempts:
                _insert_attempt(connection, call.llm_call_id, attempt)
            row = connection.execute(
                "SELECT * FROM local_llm_calls WHERE llm_call_id = ?", (call.llm_call_id,)
            ).fetchone()
            if row is None:
                raise StorageIntegrityError("Committed LLM call could not be read")
            return _load_llm_record(connection, row)

        return await self._database.transaction(commit)

    async def get_llm_call(self, scope: StorageScope, llm_call_id: str) -> LLMCallDetail | None:
        """Read one exact LLM call and its policy-retained captured content.

        Canonical scope authorization occurs through the owning observation. Only
        this detail API hydrates request, response, and trace fragments.

        Examples:
            Inspect one call:
                ```python
                detail = await repository.get_llm_call(scope, "call-1")
                ```

            Detect absence:
                ```python
                assert await repository.get_llm_call(scope, "missing") is None
                ```

        Args:
            scope: Populated canonical scope constraining access.
            llm_call_id: Exact stable LLM call identity.

        Returns:
            LLMCallDetail | None: Authorized detail or `None`.

        Notes:
            Metadata and off capture modes never expose retained bodies.
        """
        _nonempty("llm_call_id", llm_call_id)
        if not scope.as_filter():
            return None

        def read(connection: sqlite3.Connection) -> LLMCallDetail | None:
            clauses, values = _scope_filters(scope, alias="o")
            row = connection.execute(
                "SELECT l.* FROM local_llm_calls l JOIN local_observations o "
                "ON o.observation_id = l.observation_id "
                "WHERE l.llm_call_id = ? AND " + " AND ".join(clauses),
                (llm_call_id, *values),
            ).fetchone()
            if row is None:
                return None
            record = _load_llm_record(connection, row)
            request = response = trace = None
            if record.capture_mode in {
                ObservationCaptureMode.MANIFEST,
                ObservationCaptureMode.FULL,
            }:
                if record.prompt_manifest_id is not None:
                    manifest = connection.execute(
                        "SELECT * FROM local_observation_manifests WHERE manifest_id = ?",
                        (record.prompt_manifest_id,),
                    ).fetchone()
                    if manifest is None:
                        raise StorageIntegrityError("LLM prompt manifest is missing")
                    request = _read_fragment(connection, manifest["request_fragment_id"])
                    trace = _read_fragment(connection, manifest["trace_fragment_id"])
                response = _read_fragment(connection, row["response_fragment_id"])
            return LLMCallDetail(
                record=record,
                captured_request=request,
                captured_response=response,
                trace_payload=trace,
            )

        return await self._database.read_transaction(read)

    async def query_llm_calls(self, query: LLMCallQuery) -> Page[LLMCallRecord]:
        """Query a bounded metadata-only page through promoted LLM indexes.

        Scope, trace, provider, model, call type, status, and occurrence filters run
        before stable pagination. Attempts and links are batch hydrated.

        Examples:
            List failed calls:
                ```python
                page = await repository.query_llm_calls(LLMCallQuery(scope=scope, statuses=(status,)))
                ```

            Continue a page:
                ```python
                page = await repository.query_llm_calls(next_query)
                ```

        Args:
            query: Exact canonical scope, LLM filters, and opaque page request.

        Returns:
            Page[LLMCallRecord]: Metadata-only records and optional cursor.

        Notes:
            Captured prompt, response, and trace bodies are excluded from every page.
        """
        clauses, values = _scope_filters(query.scope, alias="o")
        if query.trace_id is not None:
            clauses.append("o.trace_id = ?")
            values.append(query.trace_id)
        for column, selected in (
            ("provider", query.providers),
            ("model", query.models),
            ("call_type", query.call_types),
        ):
            if selected:
                clauses.append(f"l.{column} IN ({','.join('?' for _ in selected)})")
                values.extend(selected)
        if query.statuses:
            clauses.append(f"o.status IN ({','.join('?' for _ in query.statuses)})")
            values.extend(status.value for status in query.statuses)
        if query.occurred_at_or_after is not None:
            clauses.append("o.occurred_at >= ?")
            values.append(query.occurred_at_or_after.isoformat())
        if query.occurred_at_or_before is not None:
            clauses.append("o.occurred_at <= ?")
            values.append(query.occurred_at_or_before.isoformat())
        fingerprint = _llm_query_fingerprint(query)
        if query.page.cursor:
            timestamp, sequence = _decode_cursor(query.page.cursor, fingerprint)
            clauses.append("(o.occurred_at, o.sequence) < (?, ?)")
            values.extend((timestamp, sequence))

        def read(connection: sqlite3.Connection) -> Page[LLMCallRecord]:
            rows = connection.execute(
                "SELECT l.*, o.occurred_at AS page_occurred_at, "
                "o.sequence AS page_sequence "
                "FROM local_llm_calls l JOIN local_observations o "
                "ON o.observation_id = l.observation_id "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY o.occurred_at DESC, o.sequence DESC LIMIT ?",
                (*values, query.page.limit + 1),
            ).fetchall()
            visible = rows[: query.page.limit]
            records = _load_llm_records(connection, visible)
            next_cursor = None
            if len(rows) > query.page.limit:
                anchor = visible[-1]
                next_cursor = _encode_cursor(
                    fingerprint,
                    str(anchor["page_occurred_at"]),
                    int(anchor["page_sequence"]),
                )
            return Page(items=records, next_cursor=next_cursor)

        return await self._database.read_transaction(read)

    async def purge(self, request: ObservationPurgeRequest) -> ObservationPurgeResult:
        """Preview or execute one bounded pin-aware retention transaction.

        Candidate selection, shared-fragment accounting, row deletion, manifest
        cleanup, and orphan-fragment collection use one consistent provider snapshot.

        Examples:
            Preview a purge:
                ```python
                preview = await repository.purge(request)
                ```

            Execute an approved purge:
                ```python
                result = await repository.purge(execute_request)
                ```

        Args:
            request: Exact scope, retention filters, safety bound, and dry-run mode.

        Returns:
            ObservationPurgeResult: Matching, retained, reclaimed, and deletion counts.

        Notes:
            Dry runs work through read-only handles; execution rejects read-only mode.
        """
        if request.dry_run:
            return await self._database.read_transaction(
                lambda connection: _purge(connection, request, execute=False)
            )
        self._require_writable()
        return await self._database.transaction(
            lambda connection: _purge(connection, request, execute=True)
        )

    async def storage_stats(self, scope: StorageScope) -> ObservationStorageStats:
        """Return logical observation accounting for one canonical scope.

        Counts include only observations, LLM metadata, manifests, and deduplicated
        captured fragments reachable from the requested populated scope.

        Examples:
            Inspect project usage:
                ```python
                stats = await repository.storage_stats(project_scope)
                ```

            Inspect fragment bytes:
                ```python
                print((await repository.storage_stats(scope)).fragment_bytes)
                ```

        Args:
            scope: Populated canonical scope constraining logical accounting.

        Returns:
            ObservationStorageStats: Logical counts, bytes, and safe provider metrics.

        Notes:
            Provider metrics expose allocation counts, never filenames or SQLite handles.
        """
        if not scope.as_filter():
            raise StorageConfigurationError("Observation stats require populated scope")
        return await self._database.read_transaction(
            lambda connection: _storage_stats(connection, scope)
        )

    async def get_scope_management(
        self, scope: StorageScope, scope_key: str
    ) -> ObservationScopeManagementRecord | None:
        """Read management state for one exact logical scope key.

        The lookup uses the complete canonical scope identity and does not synthesize
        defaults or inherit another trace/run/session policy.

        Examples:
            Read trace policy:
                ```python
                policy = await repository.get_scope_management(scope, "trace:trace-1")
                ```

            Detect provider default:
                ```python
                assert await repository.get_scope_management(scope, "trace:new") is None
                ```

        Args:
            scope: Populated exact canonical scope constraining access.
            scope_key: Exact opaque management identity.

        Returns:
            ObservationScopeManagementRecord | None: Stored policy or `None`.

        Notes:
            Missing records leave default-policy choice to AG rather than the provider.
        """
        _nonempty("scope_key", scope_key)
        identity = _scope_identity(scope)
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_observation_scope_management
            WHERE scope_identity = ? AND management_key = ?
            """,
            (identity, scope_key),
        )
        return _management(rows[0]) if rows else None

    async def compare_and_set_scope_management(
        self,
        record: ObservationScopeManagementRecord,
        expected_revision: int,
    ) -> ObservationScopeManagementRecord:
        """Atomically create or advance one exact scope-management record.

        Revision zero creates the first policy. Pin, visibility, deletion marker,
        labels, tags, retention, and expiry then advance only through revision CAS.

        Examples:
            Create a pinned trace policy:
                ```python
                stored = await repository.compare_and_set_scope_management(record, 0)
                ```

            Mark the scope hidden:
                ```python
                stored = await repository.compare_and_set_scope_management(updated, current.revision)
                ```

        Args:
            record: Complete canonical next management revision.
            expected_revision: Current revision, or zero for exact creation.

        Returns:
            ObservationScopeManagementRecord: Newly committed authoritative policy.

        Notes:
            Scope, management key, and trace association are immutable after creation.
        """
        self._require_writable()
        if not record.scope.as_filter():
            raise StorageConfigurationError("Observation management requires populated scope")
        _next_revision(record.revision, expected_revision)
        identity = _scope_identity(record.scope)

        def commit(connection: sqlite3.Connection) -> ObservationScopeManagementRecord:
            row = connection.execute(
                """
                SELECT * FROM local_observation_scope_management
                WHERE scope_identity = ? AND management_key = ?
                """,
                (identity, record.scope_key),
            ).fetchone()
            if row is None:
                if expected_revision != 0:
                    raise StorageConflictError("Observation management revision is stale")
                _insert_management(connection, record)
                return record
            current = _management(row)
            if current.revision != expected_revision:
                raise StorageConflictError("Observation management revision is stale")
            if (
                current.scope != record.scope
                or current.scope_key != record.scope_key
                or current.trace_id != record.trace_id
            ):
                raise StorageIntegrityError("Observation management immutable identity changed")
            if record.updated_at < current.updated_at:
                raise StorageIntegrityError("Observation management updated_at moved backward")
            _update_management(connection, record)
            return record

        return await self._database.transaction(commit)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local observation repository is read-only")


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.CONTROL:
        raise StorageConfigurationError("Local observation repository requires control database")
    database.install_component(
        name="observations",
        version=_COMPONENT_VERSION,
        statements=(
            _CREATE_OBSERVATIONS,
            _CREATE_OBSERVATION_PROJECT_INDEX,
            _CREATE_OBSERVATION_RUN_INDEX,
            _CREATE_OBSERVATION_SESSION_INDEX,
            _CREATE_OBSERVATION_GRAPH_INDEX,
            _CREATE_OBSERVATION_TRACE_INDEX,
            _CREATE_OBSERVATION_CATEGORY_INDEX,
            _CREATE_OBSERVATION_STATUS_INDEX,
            _CREATE_OBSERVATION_SEVERITY_INDEX,
            _CREATE_RESOURCE_LINKS,
            _CREATE_RESOURCE_LOOKUP_INDEX,
            _CREATE_FRAGMENTS,
            _CREATE_MANIFESTS,
            _CREATE_LLM_CALLS,
            _CREATE_LLM_PROVIDER_INDEX,
            _CREATE_LLM_MODEL_INDEX,
            _CREATE_LLM_TYPE_INDEX,
            _CREATE_LLM_MANIFEST_INDEX,
            _CREATE_ATTEMPTS,
            _CREATE_MANAGEMENT,
            _CREATE_MANAGEMENT_TRACE_INDEX,
        ),
    )


def _append_observation(
    connection: sqlite3.Connection, draft: ObservationDraft
) -> ObservationRecord:
    digest = _observation_digest(draft)
    existing = connection.execute(
        "SELECT * FROM local_observations WHERE observation_id = ?",
        (draft.observation_id,),
    ).fetchone()
    if existing is not None:
        if str(existing["content_digest"]) != digest:
            raise StorageIntegrityError(f"Observation identity {draft.observation_id!r} conflicts")
        return _load_observation(connection, existing)
    cursor = connection.execute(
        """
        INSERT INTO local_observations(
            observation_id, scope_identity, tenant_id, project_id, org_id, user_id,
            session_id, run_id, graph_id, node_id, agent_id, scope_key, category,
            name, summary, occurred_at, status, severity, trace_id, turn_id,
            parent_observation_id, caused_by_observation_id, source_event_id,
            attributes_json, payload_fragment_id, retention_class, expires_at,
            schema_version, content_digest
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            draft.observation_id,
            _scope_identity(draft.scope),
            *(_scope_values(draft.scope)),
            draft.category,
            draft.name,
            draft.summary,
            draft.occurred_at.isoformat(),
            draft.status.value,
            draft.severity.value,
            draft.trace_id,
            draft.turn_id,
            draft.parent_observation_id,
            draft.caused_by_observation_id,
            draft.source_event_id,
            _json(draft.attributes),
            draft.payload_fragment_id,
            draft.retention_class,
            _optional_iso(draft.expires_at),
            draft.schema_version,
            digest,
        ),
    )
    sequence = int(cursor.lastrowid)
    for ordinal, link in enumerate(draft.resource_links):
        connection.execute(
            """
            INSERT INTO local_observation_resource_links(
                observation_id, ordinal, resource_key, relation, resource_revision,
                content_hash, slot_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                draft.observation_id,
                ordinal,
                link.resource_key,
                link.relation.value,
                link.resource_revision,
                link.content_hash,
                link.slot_key,
            ),
        )
    return _record_from_draft(draft, sequence)


def _record_from_draft(draft: ObservationDraft, sequence: int) -> ObservationRecord:
    return ObservationRecord(
        observation_id=draft.observation_id,
        category=draft.category,
        name=draft.name,
        summary=draft.summary,
        occurred_at=draft.occurred_at,
        scope=draft.scope,
        cursor=_observation_cursor(sequence),
        status=draft.status,
        severity=draft.severity,
        trace_id=draft.trace_id,
        turn_id=draft.turn_id,
        parent_observation_id=draft.parent_observation_id,
        caused_by_observation_id=draft.caused_by_observation_id,
        source_event_id=draft.source_event_id,
        attributes=draft.attributes,
        resource_links=draft.resource_links,
        payload_fragment_id=draft.payload_fragment_id,
        retention_class=draft.retention_class,
        expires_at=draft.expires_at,
        schema_version=draft.schema_version,
    )


def _load_observation(connection: sqlite3.Connection, row: sqlite3.Row) -> ObservationRecord:
    links = connection.execute(
        """
        SELECT * FROM local_observation_resource_links
        WHERE observation_id = ? ORDER BY ordinal
        """,
        (str(row["observation_id"]),),
    ).fetchall()
    return _observation(row, tuple(_resource_link(link) for link in links))


def _load_observations(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
) -> tuple[ObservationRecord, ...]:
    if not rows:
        return ()
    identities = tuple(str(row["observation_id"]) for row in rows)
    placeholders = ",".join("?" for _ in identities)
    link_rows = connection.execute(
        "SELECT * FROM local_observation_resource_links WHERE observation_id IN ("
        + placeholders
        + ") ORDER BY observation_id, ordinal",
        identities,
    ).fetchall()
    links = _links_by_observation(link_rows)
    return tuple(_observation(row, links.get(str(row["observation_id"]), ())) for row in rows)


def _observation(
    row: sqlite3.Row, resource_links: tuple[ObservationResourceLink, ...]
) -> ObservationRecord:
    try:
        return ObservationRecord(
            observation_id=str(row["observation_id"]),
            category=str(row["category"]),
            name=str(row["name"]),
            summary=str(row["summary"]),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            scope=_scope(row),
            cursor=_observation_cursor(int(row["sequence"])),
            status=ObservationStatus(str(row["status"])),
            severity=ObservationSeverity(str(row["severity"])),
            trace_id=row["trace_id"],
            turn_id=row["turn_id"],
            parent_observation_id=row["parent_observation_id"],
            caused_by_observation_id=row["caused_by_observation_id"],
            source_event_id=row["source_event_id"],
            attributes=_json_object(row["attributes_json"]),
            resource_links=resource_links,
            payload_fragment_id=row["payload_fragment_id"],
            retention_class=str(row["retention_class"]),
            expires_at=_optional_time(row["expires_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local observation row is malformed") from exc


def _resource_link(row: sqlite3.Row) -> ObservationResourceLink:
    try:
        return ObservationResourceLink(
            resource_key=str(row["resource_key"]),
            relation=ObservationResourceRelation(str(row["relation"])),
            resource_revision=row["resource_revision"],
            content_hash=row["content_hash"],
            slot_key=row["slot_key"],
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted observation resource link is malformed") from exc


def _links_by_observation(
    rows: Sequence[sqlite3.Row],
) -> dict[str, tuple[ObservationResourceLink, ...]]:
    grouped: dict[str, list[ObservationResourceLink]] = {}
    for row in rows:
        grouped.setdefault(str(row["observation_id"]), []).append(_resource_link(row))
    return {identity: tuple(links) for identity, links in grouped.items()}


def _store_fragment(
    connection: sqlite3.Connection,
    *,
    content_kind: str,
    value: object,
    created_at: datetime,
) -> str | None:
    if value is None:
        return None
    body = _json(value)
    canonical_hash = hashlib.sha256(body.encode()).hexdigest()
    fragment_id = (
        "obsfrag-" + hashlib.sha256(f"{content_kind}\x00{canonical_hash}".encode()).hexdigest()
    )
    existing = connection.execute(
        "SELECT * FROM local_observation_fragments WHERE fragment_id = ?", (fragment_id,)
    ).fetchone()
    if existing is not None:
        if (
            str(existing["content_kind"]) != content_kind
            or str(existing["canonical_hash"]) != canonical_hash
            or str(existing["body_json"]) != body
        ):
            raise StorageIntegrityError("Observation fragment digest collision")
        return fragment_id
    connection.execute(
        """
        INSERT INTO local_observation_fragments(
            fragment_id, content_kind, canonical_hash, byte_count, body_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            fragment_id,
            content_kind,
            canonical_hash,
            len(body.encode()),
            body,
            created_at.isoformat(),
        ),
    )
    return fragment_id


def _put_manifest(
    connection: sqlite3.Connection,
    *,
    manifest_id: str,
    capture_mode: ObservationCaptureMode,
    request_fragment_id: str | None,
    trace_fragment_id: str | None,
    created_at: datetime,
) -> None:
    values = (
        manifest_id,
        capture_mode.value,
        request_fragment_id,
        trace_fragment_id,
        created_at.isoformat(),
    )
    existing = connection.execute(
        "SELECT * FROM local_observation_manifests WHERE manifest_id = ?", (manifest_id,)
    ).fetchone()
    if existing is not None:
        current = tuple(
            existing[name]
            for name in (
                "manifest_id",
                "capture_mode",
                "request_fragment_id",
                "trace_fragment_id",
                "created_at",
            )
        )
        if current != values:
            raise StorageIntegrityError(f"Observation manifest {manifest_id!r} conflicts")
        return
    connection.execute(
        """
        INSERT INTO local_observation_manifests(
            manifest_id, capture_mode, request_fragment_id, trace_fragment_id, created_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        values,
    )


def _read_fragment(connection: sqlite3.Connection, fragment_id: object) -> object:
    if fragment_id is None:
        return None
    row = connection.execute(
        "SELECT body_json FROM local_observation_fragments WHERE fragment_id = ?",
        (fragment_id,),
    ).fetchone()
    if row is None:
        raise StorageIntegrityError("Observation capture fragment is missing")
    try:
        return json.loads(str(row["body_json"]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Observation capture fragment is malformed") from exc


def _insert_attempt(
    connection: sqlite3.Connection, llm_call_id: str, attempt: LLMCallAttempt
) -> None:
    connection.execute(
        """
        INSERT INTO local_llm_attempts(
            llm_call_id, attempt_number, elapsed_ms, outcome, retryable, status_code,
            error_code, request_id, provider_delay_ms, scheduled_delay_ms,
            rate_limits_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            llm_call_id,
            attempt.attempt_number,
            attempt.elapsed_ms,
            attempt.outcome,
            int(attempt.retryable),
            attempt.status_code,
            attempt.error_code,
            attempt.request_id,
            attempt.provider_delay_ms,
            attempt.scheduled_delay_ms,
            _json(attempt.rate_limits),
        ),
    )


def _attempt(row: sqlite3.Row) -> LLMCallAttempt:
    try:
        rate_limits = _json_array(row["rate_limits_json"])
        if any(not isinstance(value, dict) for value in rate_limits):
            raise TypeError("rate limit entries must be objects")
        return LLMCallAttempt(
            attempt_number=int(row["attempt_number"]),
            elapsed_ms=int(row["elapsed_ms"]),
            outcome=str(row["outcome"]),
            retryable=bool(row["retryable"]),
            status_code=row["status_code"],
            error_code=row["error_code"],
            request_id=row["request_id"],
            provider_delay_ms=row["provider_delay_ms"],
            scheduled_delay_ms=row["scheduled_delay_ms"],
            rate_limits=tuple(rate_limits),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted LLM attempt row is malformed") from exc


def _attempts_by_call(rows: Sequence[sqlite3.Row]) -> dict[str, tuple[LLMCallAttempt, ...]]:
    grouped: dict[str, list[LLMCallAttempt]] = {}
    for row in rows:
        grouped.setdefault(str(row["llm_call_id"]), []).append(_attempt(row))
    return {identity: tuple(attempts) for identity, attempts in grouped.items()}


def _load_llm_record(connection: sqlite3.Connection, row: sqlite3.Row) -> LLMCallRecord:
    observation_row = connection.execute(
        "SELECT * FROM local_observations WHERE observation_id = ?",
        (str(row["observation_id"]),),
    ).fetchone()
    if observation_row is None:
        raise StorageIntegrityError("LLM call observation is missing")
    observation = _load_observation(connection, observation_row)
    attempt_rows = connection.execute(
        """
        SELECT * FROM local_llm_attempts
        WHERE llm_call_id = ? ORDER BY attempt_number
        """,
        (str(row["llm_call_id"]),),
    ).fetchall()
    return _llm_record(row, observation, tuple(_attempt(item) for item in attempt_rows))


def _load_llm_records(
    connection: sqlite3.Connection,
    rows: Sequence[sqlite3.Row],
) -> tuple[LLMCallRecord, ...]:
    if not rows:
        return ()
    observation_ids = tuple(str(row["observation_id"]) for row in rows)
    call_ids = tuple(str(row["llm_call_id"]) for row in rows)
    observation_placeholders = ",".join("?" for _ in observation_ids)
    call_placeholders = ",".join("?" for _ in call_ids)
    observation_rows = connection.execute(
        "SELECT * FROM local_observations WHERE observation_id IN ("
        + observation_placeholders
        + ")",
        observation_ids,
    ).fetchall()
    observations = {
        record.observation_id: record for record in _load_observations(connection, observation_rows)
    }
    attempt_rows = connection.execute(
        "SELECT * FROM local_llm_attempts WHERE llm_call_id IN ("
        + call_placeholders
        + ") ORDER BY llm_call_id, attempt_number",
        call_ids,
    ).fetchall()
    attempts = _attempts_by_call(attempt_rows)
    try:
        return tuple(
            _llm_record(
                row,
                observations[str(row["observation_id"])],
                attempts.get(str(row["llm_call_id"]), ()),
            )
            for row in rows
        )
    except KeyError as exc:
        raise StorageIntegrityError("LLM call observation is missing") from exc


def _llm_record(
    row: sqlite3.Row,
    observation: ObservationRecord,
    attempts: tuple[LLMCallAttempt, ...],
) -> LLMCallRecord:
    try:
        return LLMCallRecord(
            llm_call_id=str(row["llm_call_id"]),
            observation=observation,
            call_type=str(row["call_type"]),
            provider=str(row["provider"]),
            model=str(row["model"]),
            capture_mode=ObservationCaptureMode(str(row["capture_mode"])),
            profile_name=row["profile_name"],
            call_name=row["call_name"],
            request_options=_json_object(row["request_options_json"]),
            usage=_json_object(row["usage_json"]),
            latency_ms=row["latency_ms"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            prompt_manifest_id=row["prompt_manifest_id"],
            request_preview=_json_value(row["request_preview_json"]),
            response_preview=_json_value(row["response_preview_json"]),
            trace_payload_preview=_json_value(row["trace_payload_preview_json"]),
            attempts=attempts,
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local LLM call row is malformed") from exc


def _purge(
    connection: sqlite3.Connection,
    request: ObservationPurgeRequest,
    *,
    execute: bool,
) -> ObservationPurgeResult:
    candidate_ids = _purge_candidates(connection, request)
    if request.target_reclaimed_bytes is not None and candidate_ids:
        full = _purge_accounting(connection, candidate_ids)
        if full.estimated_reclaimed_bytes >= request.target_reclaimed_bytes:
            low, high = 1, len(candidate_ids)
            while low < high:
                middle = (low + high) // 2
                preview = _purge_accounting(connection, candidate_ids[:middle])
                if preview.estimated_reclaimed_bytes >= request.target_reclaimed_bytes:
                    high = middle
                else:
                    low = middle + 1
            candidate_ids = candidate_ids[:low]
    preview = _purge_accounting(connection, candidate_ids)
    if not execute or not candidate_ids:
        return preview

    placeholders = ",".join("?" for _ in candidate_ids)
    manifest_rows = connection.execute(
        """
        SELECT DISTINCT prompt_manifest_id FROM local_llm_calls
        WHERE observation_id IN ("""
        + placeholders
        + ") AND prompt_manifest_id IS NOT NULL",
        candidate_ids,
    ).fetchall()
    manifest_ids = tuple(str(row[0]) for row in manifest_rows)
    fragment_ids = _fragment_ids_for_observations(connection, candidate_ids)
    deleted_observations = connection.execute(
        "DELETE FROM local_observations WHERE observation_id IN (" + placeholders + ")",
        candidate_ids,
    ).rowcount
    deleted_manifests = 0
    if manifest_ids:
        manifest_placeholders = ",".join("?" for _ in manifest_ids)
        deleted_manifests = connection.execute(
            "DELETE FROM local_observation_manifests WHERE manifest_id IN ("
            + manifest_placeholders
            + ") AND NOT EXISTS ("
            "SELECT 1 FROM local_llm_calls l "
            "WHERE l.prompt_manifest_id = local_observation_manifests.manifest_id)",
            manifest_ids,
        ).rowcount
    deleted_fragments = 0
    if fragment_ids:
        fragment_placeholders = ",".join("?" for _ in fragment_ids)
        deleted_fragments = connection.execute(
            "DELETE FROM local_observation_fragments WHERE fragment_id IN ("
            + fragment_placeholders
            + ") AND NOT EXISTS ("
            "SELECT 1 FROM local_observation_manifests m "
            "WHERE m.request_fragment_id = local_observation_fragments.fragment_id "
            "OR m.trace_fragment_id = local_observation_fragments.fragment_id) "
            "AND NOT EXISTS (SELECT 1 FROM local_llm_calls l "
            "WHERE l.response_fragment_id = local_observation_fragments.fragment_id)",
            fragment_ids,
        ).rowcount
    return ObservationPurgeResult(
        dry_run=False,
        matching_traces=preview.matching_traces,
        matching_observations=preview.matching_observations,
        matching_manifests=preview.matching_manifests,
        exclusive_fragment_bytes=preview.exclusive_fragment_bytes,
        shared_fragment_bytes_retained=preview.shared_fragment_bytes_retained,
        estimated_reclaimed_bytes=preview.estimated_reclaimed_bytes,
        deleted_observations=deleted_observations,
        deleted_manifests=deleted_manifests,
        deleted_fragments=deleted_fragments,
    )


def _purge_candidates(
    connection: sqlite3.Connection, request: ObservationPurgeRequest
) -> tuple[str, ...]:
    clauses, values = _scope_filters(request.scope, alias="o")
    if request.categories:
        clauses.append(f"o.category IN ({','.join('?' for _ in request.categories)})")
        values.extend(request.categories)
    if request.trace_id is not None:
        clauses.append("o.trace_id = ?")
        values.append(request.trace_id)
    if request.occurred_before is not None:
        clauses.append("o.occurred_at < ?")
        values.append(request.occurred_before.isoformat())
    if request.expired_before is not None:
        clauses.extend(("o.expires_at IS NOT NULL", "o.expires_at <= ?"))
        values.append(request.expired_before.isoformat())
    clauses.append(
        "NOT EXISTS (SELECT 1 FROM local_observation_scope_management m "
        "WHERE m.trace_id = o.trace_id AND m.pinned = 1 "
        "AND (m.tenant_id IS NULL OR m.tenant_id = o.tenant_id) "
        "AND (m.project_id IS NULL OR m.project_id = o.project_id) "
        "AND (m.org_id IS NULL OR m.org_id = o.org_id) "
        "AND (m.user_id IS NULL OR m.user_id = o.user_id) "
        "AND (m.session_id IS NULL OR m.session_id = o.session_id) "
        "AND (m.run_id IS NULL OR m.run_id = o.run_id) "
        "AND (m.graph_id IS NULL OR m.graph_id = o.graph_id) "
        "AND (m.node_id IS NULL OR m.node_id = o.node_id) "
        "AND (m.agent_id IS NULL OR m.agent_id = o.agent_id) "
        "AND (m.canonical_scope_key IS NULL "
        "OR m.canonical_scope_key = o.scope_key))"
    )
    rows = connection.execute(
        "SELECT o.observation_id FROM local_observations o "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY o.occurred_at ASC, o.sequence ASC LIMIT ?",
        (*values, request.max_observations),
    ).fetchall()
    return tuple(str(row["observation_id"]) for row in rows)


def _purge_accounting(
    connection: sqlite3.Connection, observation_ids: tuple[str, ...]
) -> ObservationPurgeResult:
    if not observation_ids:
        return ObservationPurgeResult(
            dry_run=True,
            matching_traces=0,
            matching_observations=0,
            matching_manifests=0,
            exclusive_fragment_bytes=0,
            shared_fragment_bytes_retained=0,
            estimated_reclaimed_bytes=0,
        )
    placeholders = ",".join("?" for _ in observation_ids)
    trace_row = connection.execute(
        "SELECT COUNT(DISTINCT trace_id) FROM local_observations "
        "WHERE observation_id IN (" + placeholders + ") AND trace_id IS NOT NULL",
        observation_ids,
    ).fetchone()
    manifest_row = connection.execute(
        "SELECT COUNT(DISTINCT prompt_manifest_id) FROM local_llm_calls "
        "WHERE observation_id IN (" + placeholders + ") AND prompt_manifest_id IS NOT NULL",
        observation_ids,
    ).fetchone()
    fragment_ids = _fragment_ids_for_observations(connection, observation_ids)
    selected = set(observation_ids)
    exclusive = shared = 0
    for fragment_id in fragment_ids:
        row = connection.execute(
            "SELECT byte_count FROM local_observation_fragments WHERE fragment_id = ?",
            (fragment_id,),
        ).fetchone()
        if row is None:
            raise StorageIntegrityError("Referenced observation fragment is missing")
        references = _fragment_observation_references(connection, fragment_id)
        if references - selected:
            shared += int(row["byte_count"])
        else:
            exclusive += int(row["byte_count"])
    nonfragment = _nonfragment_bytes(connection, observation_ids) + _manifest_bytes(
        connection,
        observation_ids,
        exclusive=True,
    )
    return ObservationPurgeResult(
        dry_run=True,
        matching_traces=int(trace_row[0]) if trace_row else 0,
        matching_observations=len(observation_ids),
        matching_manifests=int(manifest_row[0]) if manifest_row else 0,
        exclusive_fragment_bytes=exclusive,
        shared_fragment_bytes_retained=shared,
        estimated_reclaimed_bytes=nonfragment + exclusive,
    )


def _fragment_ids_for_observations(
    connection: sqlite3.Connection, observation_ids: tuple[str, ...]
) -> tuple[str, ...]:
    if not observation_ids:
        return ()
    placeholders = ",".join("?" for _ in observation_ids)
    rows = connection.execute(
        """
        SELECT response_fragment_id AS fragment_id FROM local_llm_calls
        WHERE observation_id IN ("""
        + placeholders
        + ") AND response_fragment_id IS NOT NULL "
        "UNION SELECT m.request_fragment_id FROM local_observation_manifests m "
        "JOIN local_llm_calls l ON l.prompt_manifest_id = m.manifest_id "
        "WHERE l.observation_id IN (" + placeholders + ") AND m.request_fragment_id IS NOT NULL "
        "UNION SELECT m.trace_fragment_id FROM local_observation_manifests m "
        "JOIN local_llm_calls l ON l.prompt_manifest_id = m.manifest_id "
        "WHERE l.observation_id IN (" + placeholders + ") AND m.trace_fragment_id IS NOT NULL",
        (*observation_ids, *observation_ids, *observation_ids),
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def _fragment_observation_references(connection: sqlite3.Connection, fragment_id: str) -> set[str]:
    rows = connection.execute(
        """
        SELECT observation_id FROM local_llm_calls WHERE response_fragment_id = ?
        UNION SELECT l.observation_id FROM local_llm_calls l
        JOIN local_observation_manifests m ON m.manifest_id = l.prompt_manifest_id
        WHERE m.request_fragment_id = ? OR m.trace_fragment_id = ?
        """,
        (fragment_id, fragment_id, fragment_id),
    ).fetchall()
    return {str(row[0]) for row in rows}


def _nonfragment_bytes(connection: sqlite3.Connection, observation_ids: tuple[str, ...]) -> int:
    if not observation_ids:
        return 0
    placeholders = ",".join("?" for _ in observation_ids)
    total = 0
    for table, column in (
        ("local_observations", "observation_id"),
        ("local_observation_resource_links", "observation_id"),
        ("local_llm_calls", "observation_id"),
    ):
        rows = connection.execute(
            f"SELECT * FROM {table} WHERE {column} IN ({placeholders})",
            observation_ids,
        ).fetchall()
        total += sum(_row_bytes(row) for row in rows)
    call_rows = connection.execute(
        "SELECT llm_call_id FROM local_llm_calls WHERE observation_id IN (" + placeholders + ")",
        observation_ids,
    ).fetchall()
    call_ids = tuple(str(row[0]) for row in call_rows)
    if call_ids:
        call_placeholders = ",".join("?" for _ in call_ids)
        attempt_rows = connection.execute(
            "SELECT * FROM local_llm_attempts WHERE llm_call_id IN (" + call_placeholders + ")",
            call_ids,
        ).fetchall()
        total += sum(_row_bytes(row) for row in attempt_rows)
    return total


def _manifest_bytes(
    connection: sqlite3.Connection,
    observation_ids: tuple[str, ...],
    *,
    exclusive: bool,
) -> int:
    if not observation_ids:
        return 0
    placeholders = ",".join("?" for _ in observation_ids)
    clauses = [
        "EXISTS (SELECT 1 FROM local_llm_calls selected "
        "WHERE selected.prompt_manifest_id = m.manifest_id "
        f"AND selected.observation_id IN ({placeholders}))"
    ]
    values: tuple[str, ...] = observation_ids
    if exclusive:
        clauses.append(
            "NOT EXISTS (SELECT 1 FROM local_llm_calls retained "
            "WHERE retained.prompt_manifest_id = m.manifest_id "
            f"AND retained.observation_id NOT IN ({placeholders}))"
        )
        values = (*observation_ids, *observation_ids)
    rows = connection.execute(
        "SELECT m.* FROM local_observation_manifests m WHERE " + " AND ".join(clauses),
        values,
    ).fetchall()
    return sum(_row_bytes(row) for row in rows)


def _row_bytes(row: sqlite3.Row) -> int:
    return sum(len(str(value).encode()) for value in row if value is not None)


def _storage_stats(connection: sqlite3.Connection, scope: StorageScope) -> ObservationStorageStats:
    clauses, values = _scope_filters(scope, alias="o")
    rows = connection.execute(
        "SELECT o.observation_id FROM local_observations o WHERE " + " AND ".join(clauses),
        values,
    ).fetchall()
    observation_ids = tuple(str(row[0]) for row in rows)
    if not observation_ids:
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        return ObservationStorageStats(
            observations=0,
            llm_calls=0,
            manifests=0,
            fragments=0,
            fragment_bytes=0,
            logical_bytes=0,
            provider_metrics={"allocated_bytes": page_count * page_size},
        )
    placeholders = ",".join("?" for _ in observation_ids)
    llm_row = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT prompt_manifest_id) FROM local_llm_calls "
        "WHERE observation_id IN (" + placeholders + ")",
        observation_ids,
    ).fetchone()
    fragment_ids = _fragment_ids_for_observations(connection, observation_ids)
    fragment_bytes = 0
    if fragment_ids:
        fragment_placeholders = ",".join("?" for _ in fragment_ids)
        fragment_row = connection.execute(
            "SELECT COALESCE(SUM(byte_count), 0) FROM local_observation_fragments "
            "WHERE fragment_id IN (" + fragment_placeholders + ")",
            fragment_ids,
        ).fetchone()
        fragment_bytes = int(fragment_row[0]) if fragment_row else 0
    page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
    page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
    nonfragment = _nonfragment_bytes(connection, observation_ids) + _manifest_bytes(
        connection,
        observation_ids,
        exclusive=False,
    )
    return ObservationStorageStats(
        observations=len(observation_ids),
        llm_calls=int(llm_row[0]) if llm_row else 0,
        manifests=int(llm_row[1]) if llm_row else 0,
        fragments=len(fragment_ids),
        fragment_bytes=fragment_bytes,
        logical_bytes=nonfragment + fragment_bytes,
        provider_metrics={"allocated_bytes": page_count * page_size},
    )


def _insert_management(
    connection: sqlite3.Connection, record: ObservationScopeManagementRecord
) -> None:
    connection.execute(
        """
        INSERT INTO local_observation_scope_management(
            scope_identity, management_key, tenant_id, project_id, org_id, user_id,
            session_id, run_id, graph_id, node_id, agent_id, canonical_scope_key,
            revision, updated_at, trace_id, pinned, hidden, deleted, label, tags_json,
            retention_class, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _management_values(record),
    )


def _update_management(
    connection: sqlite3.Connection, record: ObservationScopeManagementRecord
) -> None:
    values = _management_values(record)
    connection.execute(
        """
        UPDATE local_observation_scope_management SET
            management_key = ?, tenant_id = ?, project_id = ?, org_id = ?,
            user_id = ?, session_id = ?, run_id = ?, graph_id = ?, node_id = ?,
            agent_id = ?, canonical_scope_key = ?, revision = ?, updated_at = ?,
            trace_id = ?, pinned = ?, hidden = ?, deleted = ?, label = ?,
            tags_json = ?, retention_class = ?, expires_at = ?
        WHERE scope_identity = ? AND management_key = ?
        """,
        (*values[1:], values[0], values[1]),
    )


def _management_values(record: ObservationScopeManagementRecord) -> tuple[object, ...]:
    return (
        _scope_identity(record.scope),
        record.scope_key,
        record.scope.tenant_id,
        record.scope.project_id,
        record.scope.org_id,
        record.scope.user_id,
        record.scope.session_id,
        record.scope.run_id,
        record.scope.graph_id,
        record.scope.node_id,
        record.scope.agent_id,
        record.scope.scope_key,
        record.revision,
        record.updated_at.isoformat(),
        record.trace_id,
        int(record.pinned),
        int(record.hidden),
        int(record.deleted),
        record.label,
        _json(record.tags),
        record.retention_class,
        _optional_iso(record.expires_at),
    )


def _management(row: sqlite3.Row) -> ObservationScopeManagementRecord:
    try:
        return ObservationScopeManagementRecord(
            scope_key=str(row["management_key"]),
            scope=StorageScope(
                tenant_id=row["tenant_id"],
                project_id=row["project_id"],
                org_id=row["org_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                run_id=row["run_id"],
                graph_id=row["graph_id"],
                node_id=row["node_id"],
                agent_id=row["agent_id"],
                scope_key=row["canonical_scope_key"],
            ),
            revision=int(row["revision"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            trace_id=row["trace_id"],
            pinned=bool(row["pinned"]),
            hidden=bool(row["hidden"]),
            deleted=bool(row["deleted"]),
            label=row["label"],
            tags=tuple(_json_array(row["tags_json"])),
            retention_class=str(row["retention_class"]),
            expires_at=_optional_time(row["expires_at"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted observation management row is malformed") from exc


def _scope_values(scope: StorageScope) -> tuple[str | None, ...]:
    return tuple(getattr(scope, name) for name in _SCOPE_COLUMNS)


def _scope(row: sqlite3.Row) -> StorageScope:
    try:
        return StorageScope(**{name: row[name] for name in _SCOPE_COLUMNS})
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted observation scope is malformed") from exc


def _scope_identity(scope: StorageScope) -> str:
    if not scope.as_filter():
        raise StorageConfigurationError("Observation operations require populated scope")
    return _json(scope.as_filter())


def _scope_filters(scope: StorageScope, *, alias: str) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    values: list[object] = []
    for name, value in scope.as_filter().items():
        clauses.append(f"{alias}.{name} = ?")
        values.append(value)
    if not clauses:
        raise StorageConfigurationError("Observation operations require populated scope")
    return clauses, values


def _next_revision(revision: int, expected_revision: int) -> None:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 0
    ):
        raise ValueError("expected_revision must be a non-negative integer")
    if revision != expected_revision + 1:
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
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_value(value: object) -> Any:
    return json.loads(value)


def _json_object(value: object) -> dict[str, Any]:
    parsed = _json_value(value)
    if not isinstance(parsed, dict):
        raise TypeError("persisted JSON value must be an object")
    return parsed


def _json_array(value: object) -> list[Any]:
    parsed = _json_value(value)
    if not isinstance(parsed, list):
        raise TypeError("persisted JSON value must be an array")
    return parsed


def _observation_digest(draft: ObservationDraft) -> str:
    payload = {
        "observation_id": draft.observation_id,
        "category": draft.category,
        "name": draft.name,
        "summary": draft.summary,
        "occurred_at": draft.occurred_at.isoformat(),
        "scope": draft.scope.as_filter(),
        "status": draft.status.value,
        "severity": draft.severity.value,
        "trace_id": draft.trace_id,
        "turn_id": draft.turn_id,
        "parent_observation_id": draft.parent_observation_id,
        "caused_by_observation_id": draft.caused_by_observation_id,
        "source_event_id": draft.source_event_id,
        "attributes": draft.attributes,
        "resource_links": [
            {
                "resource_key": link.resource_key,
                "relation": link.relation.value,
                "resource_revision": link.resource_revision,
                "content_hash": link.content_hash,
                "slot_key": link.slot_key,
            }
            for link in draft.resource_links
        ],
        "payload_fragment_id": draft.payload_fragment_id,
        "retention_class": draft.retention_class,
        "expires_at": _optional_iso(draft.expires_at),
        "schema_version": draft.schema_version,
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def _llm_digest(call: LLMCallDraft) -> str:
    payload = {
        "llm_call_id": call.llm_call_id,
        "observation_digest": _observation_digest(call.observation),
        "call_type": call.call_type,
        "provider": call.provider,
        "model": call.model,
        "capture_mode": call.capture_mode.value,
        "profile_name": call.profile_name,
        "call_name": call.call_name,
        "request_options": call.request_options,
        "usage": call.usage,
        "latency_ms": call.latency_ms,
        "error_type": call.error_type,
        "error_message": call.error_message,
        "prompt_manifest_id": call.prompt_manifest_id,
        "request_preview": call.request_preview,
        "response_preview": call.response_preview,
        "captured_request": call.captured_request,
        "captured_response": call.captured_response,
        "trace_payload": call.trace_payload,
        "attempts": [
            {
                "attempt_number": attempt.attempt_number,
                "elapsed_ms": attempt.elapsed_ms,
                "outcome": attempt.outcome,
                "retryable": attempt.retryable,
                "status_code": attempt.status_code,
                "error_code": attempt.error_code,
                "request_id": attempt.request_id,
                "provider_delay_ms": attempt.provider_delay_ms,
                "scheduled_delay_ms": attempt.scheduled_delay_ms,
                "rate_limits": attempt.rate_limits,
            }
            for attempt in call.attempts
        ],
        "schema_version": call.schema_version,
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def _observation_cursor(sequence: int) -> str:
    return base64.urlsafe_b64encode(f"observation:{sequence}".encode()).decode().rstrip("=")


def _observation_query_fingerprint(query: ObservationQuery) -> str:
    payload = {
        "kind": "observations",
        "scope": query.scope.as_filter(),
        "categories": list(query.categories),
        "statuses": [status.value for status in query.statuses],
        "severities": [severity.value for severity in query.severities],
        "trace_id": query.trace_id,
        "turn_id": query.turn_id,
        "resource_key": query.resource_key,
        "resource_relation": (
            query.resource_relation.value if query.resource_relation is not None else None
        ),
        "occurred_at_or_after": _optional_iso(query.occurred_at_or_after),
        "occurred_at_or_before": _optional_iso(query.occurred_at_or_before),
        "limit": query.page.limit,
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()[:24]


def _llm_query_fingerprint(query: LLMCallQuery) -> str:
    payload = {
        "kind": "llm-calls",
        "scope": query.scope.as_filter(),
        "trace_id": query.trace_id,
        "providers": list(query.providers),
        "models": list(query.models),
        "call_types": list(query.call_types),
        "statuses": [status.value for status in query.statuses],
        "occurred_at_or_after": _optional_iso(query.occurred_at_or_after),
        "occurred_at_or_before": _optional_iso(query.occurred_at_or_before),
        "limit": query.page.limit,
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()[:24]


def _encode_cursor(fingerprint: str, timestamp: str, sequence: int) -> str:
    payload = _json({"fingerprint": fingerprint, "timestamp": timestamp, "sequence": sequence})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> tuple[str, int]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, dict) or set(payload) != {
            "fingerprint",
            "timestamp",
            "sequence",
        }:
            raise ValueError("cursor payload")
        if payload["fingerprint"] != fingerprint:
            raise ValueError("cursor context")
        timestamp = payload["timestamp"]
        sequence = payload["sequence"]
        if (
            not isinstance(timestamp, str)
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
        ):
            raise ValueError("cursor values")
        datetime.fromisoformat(timestamp)
        return timestamp, sequence
    except (
        binascii.Error,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StorageConfigurationError("Invalid or mismatched observation cursor") from exc

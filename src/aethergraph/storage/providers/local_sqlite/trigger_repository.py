"""Transactional local trigger definitions and occurrence claims."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import sqlite3
from typing import Any

from croniter import croniter  # type: ignore[import]
from dateutil.tz import gettz  # type: ignore[import-untyped]

from ...contracts import (
    ClaimedTrigger,
    Page,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
    TriggerClaimRecord,
    TriggerClaimRequest,
    TriggerClaimStatus,
    TriggerKind,
    TriggerQuery,
    TriggerRecord,
    storage_scope_matches_filter,
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
_CREATE_TRIGGERS = """
CREATE TABLE local_triggers (
    trigger_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    run_id TEXT,
    node_id TEXT,
    agent_id TEXT,
    scope_key TEXT,
    kind TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    default_inputs_json TEXT NOT NULL,
    name TEXT,
    origin TEXT NOT NULL,
    cron_expression TEXT,
    interval_seconds INTEGER,
    run_at TEXT,
    event_key TEXT,
    timezone TEXT,
    max_overlap_runs INTEGER,
    catch_up_missed INTEGER NOT NULL CHECK (catch_up_missed IN (0, 1)),
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    last_fired_at TEXT,
    next_fire_at TEXT,
    metadata_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_TRIGGER_PROJECT_INDEX = """
CREATE INDEX ix_local_triggers_project_updated
ON local_triggers(tenant_id, project_id, updated_at DESC, trigger_id DESC)
"""
_CREATE_TRIGGER_GRAPH_INDEX = """
CREATE INDEX ix_local_triggers_graph_updated
ON local_triggers(graph_id, updated_at DESC, trigger_id DESC)
"""
_CREATE_TRIGGER_OWNER_INDEX = """
CREATE INDEX ix_local_triggers_owner_updated
ON local_triggers(org_id, user_id, updated_at DESC, trigger_id DESC)
"""
_CREATE_TRIGGER_EVENT_INDEX = """
CREATE INDEX ix_local_triggers_event_lookup
ON local_triggers(event_key, active, kind, updated_at DESC, trigger_id DESC)
"""
_CREATE_TRIGGER_DUE_INDEX = """
CREATE INDEX ix_local_triggers_due
ON local_triggers(active, next_fire_at, trigger_id)
WHERE kind != 'event' AND next_fire_at IS NOT NULL
"""
_CREATE_CLAIMS = """
CREATE TABLE local_trigger_claims (
    fire_id TEXT PRIMARY KEY,
    trigger_id TEXT NOT NULL,
    tenant_id TEXT,
    project_id TEXT,
    org_id TEXT,
    user_id TEXT,
    session_id TEXT,
    graph_id TEXT NOT NULL,
    agent_id TEXT,
    scope_key TEXT,
    scheduled_for TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts >= 0),
    revision INTEGER NOT NULL CHECK (revision > 0),
    updated_at TEXT NOT NULL,
    worker_id TEXT,
    lease_until TEXT,
    retry_at TEXT,
    run_id TEXT,
    last_error TEXT,
    skip_reason TEXT,
    finished_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0)
)
"""
_CREATE_CLAIM_ELIGIBLE_INDEX = """
CREATE INDEX ix_local_trigger_claims_eligible
ON local_trigger_claims(status, retry_at, lease_until, scheduled_for, fire_id)
"""
_CREATE_CLAIM_SCOPE_INDEX = """
CREATE INDEX ix_local_trigger_claims_scope_updated
ON local_trigger_claims(graph_id, updated_at DESC, fire_id DESC)
"""
_CREATE_CLAIM_TRIGGER_INDEX = """
CREATE INDEX ix_local_trigger_claims_trigger_updated
ON local_trigger_claims(trigger_id, updated_at DESC, fire_id DESC)
"""


class LocalTriggerRepository:
    """Canonical local trigger definitions and atomic occurrence claims."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def create(self, record: TriggerRecord) -> TriggerRecord:
        """Idempotently create one revision-one canonical trigger.

        The definition and its promoted scheduling indexes commit in one local
        transaction. Exact creation retries return the authoritative stored record.

        Examples:
            Create a trigger:
                ```python
                stored = await repository.create(record)
                ```

            Retry exact creation:
                ```python
                assert await repository.create(record) == stored
                ```

        Args:
            record: Complete canonical trigger at revision one.

        Returns:
            TriggerRecord: Authoritative stored definition.

        Notes:
            Conflicting identity reuse raises `StorageIntegrityError`; no upsert occurs.
        """
        self._require_writable()
        if record.revision != 1:
            raise StorageIntegrityError("Initial trigger revision must be one")
        _validate_schedule(record)

        def commit(connection: sqlite3.Connection) -> TriggerRecord:
            row = connection.execute(
                "SELECT * FROM local_triggers WHERE trigger_id = ?", (record.trigger_id,)
            ).fetchone()
            if row is not None:
                current = _trigger(row)
                if current != record:
                    raise StorageIntegrityError(f"Trigger identity {record.trigger_id!r} conflicts")
                return current
            retained = connection.execute(
                "SELECT 1 FROM local_trigger_claims WHERE trigger_id = ? LIMIT 1",
                (record.trigger_id,),
            ).fetchone()
            if retained is not None:
                raise StorageIntegrityError(
                    f"Trigger identity {record.trigger_id!r} has retained receipts"
                )
            _insert_trigger(connection, record)
            return record

        return await self._database.transaction(commit)

    async def get(self, scope: StorageScope, trigger_id: str) -> TriggerRecord | None:
        """Read one exact authorized trigger definition.

        The lookup uses only canonical scope dimensions and never broadens after a
        miss or retries through deprecated application/client identities.

        Examples:
            Read an owned trigger:
                ```python
                trigger = await repository.get(scope, "trigger-1")
                ```

            Detect absence:
                ```python
                assert await repository.get(scope, "missing") is None
                ```

        Args:
            scope: Populated canonical owner or graph scope constraining access.
            trigger_id: Exact stable trigger identity.

        Returns:
            TriggerRecord | None: Authorized current definition or `None`.

        Notes:
            Empty and cross-owner scopes behave as absence without disclosure.
        """
        _nonempty("trigger_id", trigger_id)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_triggers WHERE trigger_id = ?", (trigger_id,)
        )
        if not rows:
            return None
        record = _trigger(rows[0])
        return record if storage_scope_matches_filter(record.scope, scope) else None

    async def compare_and_set(self, record: TriggerRecord, expected_revision: int) -> TriggerRecord:
        """Atomically replace mutable trigger definition state.

        Scheduling configuration, activity, inputs, metadata, and next-fire state
        become visible together at the exact next revision.

        Examples:
            Pause a trigger:
                ```python
                paused = await repository.compare_and_set(record, current.revision)
                ```

            Change a schedule:
                ```python
                changed = await repository.compare_and_set(record, paused.revision)
                ```

        Args:
            record: Complete canonical next trigger revision.
            expected_revision: Exact current revision required for replacement.

        Returns:
            TriggerRecord: Newly committed authoritative definition.

        Notes:
            Identity, creation provenance, schema, and monotonic update time are protected.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)
        _validate_schedule(record)

        def commit(connection: sqlite3.Connection) -> TriggerRecord:
            row = connection.execute(
                "SELECT * FROM local_triggers WHERE trigger_id = ?", (record.trigger_id,)
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(record.trigger_id)
            current = _trigger(row)
            if current.revision != expected_revision:
                raise StorageConflictError("Trigger revision is stale")
            if _trigger_identity(current) != _trigger_identity(record):
                raise StorageIntegrityError("Trigger immutable identity changed")
            if record.updated_at < current.updated_at:
                raise StorageIntegrityError("Trigger updated_at moved backward")
            _update_trigger(connection, record)
            return record

        return await self._database.transaction(commit)

    async def delete(
        self,
        scope: StorageScope,
        trigger_id: str,
        expected_revision: int,
    ) -> bool:
        """Delete an exact trigger while preserving terminal receipts.

        The definition and any leased/retry claims are removed atomically. Delivered
        and skipped receipts remain durable deduplication and audit evidence.

        Examples:
            Delete an owned trigger:
                ```python
                removed = await repository.delete(scope, trigger_id, revision)
                ```

            Detect absence:
                ```python
                assert not await repository.delete(scope, "missing", 1)
                ```

        Args:
            scope: Populated canonical owner or graph scope constraining deletion.
            trigger_id: Exact stable trigger identity.
            expected_revision: Current definition revision required for deletion.

        Returns:
            bool: Whether the exact scoped trigger was deleted.

        Notes:
            Stale revision raises `StorageConflictError`; scope mismatch returns `False`.
        """
        self._require_writable()
        _nonempty("trigger_id", trigger_id)
        _expected_revision(expected_revision)

        def commit(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                "SELECT * FROM local_triggers WHERE trigger_id = ?", (trigger_id,)
            ).fetchone()
            if row is None:
                return False
            current = _trigger(row)
            if not storage_scope_matches_filter(current.scope, scope):
                return False
            if current.revision != expected_revision:
                raise StorageConflictError("Trigger revision is stale")
            connection.execute(
                """
                DELETE FROM local_trigger_claims
                WHERE trigger_id = ? AND status IN ('leased', 'retry')
                """,
                (trigger_id,),
            )
            connection.execute("DELETE FROM local_triggers WHERE trigger_id = ?", (trigger_id,))
            return True

        return await self._database.transaction(commit)

    async def query(self, query: TriggerQuery) -> Page[TriggerRecord]:
        """Query one bounded stable page through promoted trigger indexes.

        Canonical scope plus optional kind, activity, and event-key filters execute
        in SQL before descending update-time and trigger-identity pagination.

        Examples:
            List active triggers:
                ```python
                page = await repository.query(TriggerQuery(scope=scope, active=True))
                ```

            Resolve one event key:
                ```python
                page = await repository.query(TriggerQuery(scope=scope, event_key="paid"))
                ```

        Args:
            query: Exact scope, indexed filters, and opaque page request.

        Returns:
            Page[TriggerRecord]: Matching definitions and optional continuation cursor.

        Notes:
            The cursor is bound to the complete query, including page size.
        """
        clauses, values = _scope_filters(query.scope, alias="t")
        if query.kinds:
            clauses.append(f"t.kind IN ({','.join('?' for _ in query.kinds)})")
            values.extend(kind.value for kind in query.kinds)
        if query.active is not None:
            clauses.append("t.active = ?")
            values.append(int(query.active))
        if query.event_key is not None:
            clauses.append("t.event_key = ?")
            values.append(query.event_key)
        fingerprint = _query_fingerprint(query)
        if query.page.cursor:
            timestamp, identity = _decode_cursor(query.page.cursor, fingerprint)
            clauses.append("(t.updated_at, t.trigger_id) < (?, ?)")
            values.extend((timestamp, identity))
        rows = await self._database.fetch_all(
            "SELECT t.* FROM local_triggers t "
            f"WHERE {' AND '.join(clauses)} "
            "ORDER BY t.updated_at DESC, t.trigger_id DESC LIMIT ?",
            (*values, query.page.limit + 1),
        )
        visible = rows[: query.page.limit]
        next_cursor = None
        if len(rows) > query.page.limit:
            anchor = visible[-1]
            next_cursor = _encode_cursor(
                fingerprint, str(anchor["updated_at"]), str(anchor["trigger_id"])
            )
        return Page(items=tuple(_trigger(row) for row in visible), next_cursor=next_cursor)

    async def claim_due(self, request: TriggerClaimRequest) -> tuple[ClaimedTrigger, ...]:
        """Atomically claim a bounded batch and advance trigger schedules.

        Eligible retries and expired leases are reclaimed before newly due trigger
        definitions. Claim rows, missed-run receipts, and schedule revisions commit
        in one transaction.

        Examples:
            Claim provider-wide due work:
                ```python
                claims = await repository.claim_due(request)
                ```

            Claim an owner partition:
                ```python
                claims = await repository.claim_due(scoped_request)
                ```

        Args:
            request: Worker, clock, bound, optional scope, and catch-up boundary.

        Returns:
            tuple[ClaimedTrigger, ...]: Worker-owned claims up to the request limit.

        Notes:
            `scope=None` is an explicit trusted global scan, never a fallback after a miss.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> tuple[ClaimedTrigger, ...]:
            claimed: list[ClaimedTrigger] = []
            retry_clauses = [
                "((c.status = 'retry' AND c.retry_at <= ?) "
                "OR (c.status = 'leased' AND c.lease_until <= ?))"
            ]
            retry_values: list[object] = [request.now.isoformat(), request.now.isoformat()]
            if request.scope is not None:
                scope_clauses, scope_values = _scope_filters(request.scope, alias="t")
                retry_clauses.extend(scope_clauses)
                retry_values.extend(scope_values)
            retry_rows = connection.execute(
                "SELECT c.* FROM local_trigger_claims c "
                "JOIN local_triggers t ON t.trigger_id = c.trigger_id "
                f"WHERE {' AND '.join(retry_clauses)} "
                "ORDER BY c.scheduled_for ASC, c.fire_id ASC LIMIT ?",
                (*retry_values, request.limit),
            ).fetchall()
            for row in retry_rows:
                current_claim = _claim(row)
                reclaimed = current_claim.status is TriggerClaimStatus.LEASED
                trigger_row = connection.execute(
                    "SELECT * FROM local_triggers WHERE trigger_id = ?",
                    (current_claim.trigger_id,),
                ).fetchone()
                if trigger_row is None:
                    raise StorageIntegrityError("Eligible trigger claim has no definition")
                trigger = _trigger(trigger_row)
                if request.now < current_claim.updated_at:
                    raise StorageIntegrityError("Trigger claim clock moved backward")
                renewed = replace(
                    current_claim,
                    status=TriggerClaimStatus.LEASED,
                    attempts=current_claim.attempts + 1,
                    revision=current_claim.revision + 1,
                    updated_at=request.now,
                    worker_id=request.worker_id,
                    lease_until=request.lease_until,
                    retry_at=None,
                    last_error=None,
                )
                _update_claim(connection, renewed)
                claimed.append(
                    ClaimedTrigger(
                        trigger=trigger,
                        claim=renewed,
                        reclaimed=reclaimed,
                    )
                )

            remaining = request.limit - len(claimed)
            if remaining <= 0:
                return tuple(claimed)
            due_clauses = [
                "t.active = 1",
                "t.kind != 'event'",
                "t.next_fire_at IS NOT NULL",
                "t.next_fire_at <= ?",
            ]
            due_values: list[object] = [request.now.isoformat()]
            if request.scope is not None:
                scope_clauses, scope_values = _scope_filters(request.scope, alias="t")
                due_clauses.extend(scope_clauses)
                due_values.extend(scope_values)
            due_rows = connection.execute(
                "SELECT t.* FROM local_triggers t "
                f"WHERE {' AND '.join(due_clauses)} "
                "ORDER BY t.next_fire_at ASC, t.trigger_id ASC LIMIT ?",
                (*due_values, remaining),
            ).fetchall()
            for row in due_rows:
                trigger = _trigger(row)
                if trigger.updated_at > request.now:
                    raise StorageIntegrityError("Trigger claim clock moved backward")
                scheduled_for = trigger.next_fire_at
                if scheduled_for is None:
                    raise StorageIntegrityError("Due trigger is missing next_fire_at")
                fire_id = _fire_id(trigger.trigger_id, scheduled_for)
                missed = (
                    request.skip_missed_before is not None
                    and scheduled_for < request.skip_missed_before
                    and not trigger.catch_up_missed
                )
                if missed:
                    receipt = TriggerClaimRecord(
                        fire_id=fire_id,
                        trigger_id=trigger.trigger_id,
                        scope=trigger.scope,
                        scheduled_for=scheduled_for,
                        status=TriggerClaimStatus.SKIPPED,
                        attempts=0,
                        revision=1,
                        updated_at=request.now,
                        skip_reason="missed_before_startup",
                        finished_at=request.now,
                    )
                    _insert_claim(connection, receipt)
                    advanced = _advance_trigger(
                        trigger,
                        scheduled_for=scheduled_for,
                        now=request.skip_missed_before,
                    )
                    _update_trigger(connection, advanced)
                    continue
                lease = TriggerClaimRecord(
                    fire_id=fire_id,
                    trigger_id=trigger.trigger_id,
                    scope=trigger.scope,
                    scheduled_for=scheduled_for,
                    status=TriggerClaimStatus.LEASED,
                    attempts=1,
                    revision=1,
                    updated_at=request.now,
                    worker_id=request.worker_id,
                    lease_until=request.lease_until,
                )
                _insert_claim(connection, lease)
                advanced = _advance_trigger(
                    trigger,
                    scheduled_for=scheduled_for,
                    now=request.now,
                )
                _update_trigger(connection, advanced)
                claimed.append(ClaimedTrigger(trigger=advanced, claim=lease))
            return tuple(claimed)

        return await self._database.transaction(commit)

    async def get_claim(self, scope: StorageScope, fire_id: str) -> TriggerClaimRecord | None:
        """Read one exact authorized trigger claim or terminal receipt.

        The read uses copied canonical scope on the occurrence and never changes
        lease ownership, retry eligibility, or the owning trigger definition.

        Examples:
            Inspect a receipt:
                ```python
                receipt = await repository.get_claim(scope, "fire-1")
                ```

            Detect absence:
                ```python
                assert await repository.get_claim(scope, "missing") is None
                ```

        Args:
            scope: Populated canonical owner or graph scope constraining access.
            fire_id: Exact stable occurrence identity.

        Returns:
            TriggerClaimRecord | None: Authorized claim/receipt or `None`.

        Notes:
            Terminal receipts remain readable after their trigger definition is deleted.
        """
        _nonempty("fire_id", fire_id)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_trigger_claims WHERE fire_id = ?", (fire_id,)
        )
        if not rows:
            return None
        record = _claim(rows[0])
        return record if storage_scope_matches_filter(record.scope, scope) else None

    async def compare_and_set_claim(
        self, record: TriggerClaimRecord, expected_revision: int
    ) -> TriggerClaimRecord:
        """Atomically renew, retry, deliver, or skip an active trigger claim.

        The occurrence transition commits at the exact next revision. Delivery also
        advances the current trigger revision and last-fired timestamp atomically.

        Examples:
            Commit delivery:
                ```python
                receipt = await repository.compare_and_set_claim(delivered, lease.revision)
                ```

            Release into retry:
                ```python
                retry = await repository.compare_and_set_claim(failed, lease.revision)
                ```

        Args:
            record: Complete canonical next claim or receipt revision.
            expected_revision: Exact current claim revision required for transition.

        Returns:
            TriggerClaimRecord: Newly committed authoritative claim or receipt.

        Notes:
            Terminal receipts are immutable; lease renewal cannot change worker identity.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)

        def commit(connection: sqlite3.Connection) -> TriggerClaimRecord:
            row = connection.execute(
                "SELECT * FROM local_trigger_claims WHERE fire_id = ?", (record.fire_id,)
            ).fetchone()
            if row is None:
                raise StorageNotFoundError(record.fire_id)
            current = _claim(row)
            if current.revision != expected_revision:
                raise StorageConflictError("Trigger claim revision is stale")
            if _claim_identity(current) != _claim_identity(record):
                raise StorageIntegrityError("Trigger claim immutable identity changed")
            if current.status is not TriggerClaimStatus.LEASED:
                raise StorageConflictError("Only an active trigger claim may transition")
            if record.updated_at < current.updated_at:
                raise StorageIntegrityError("Trigger claim updated_at moved backward")
            if record.status is TriggerClaimStatus.LEASED:
                if record.worker_id != current.worker_id:
                    raise StorageConflictError("Trigger claim worker ownership changed")
                if record.lease_until is None or current.lease_until is None:
                    raise StorageIntegrityError("Trigger claim renewal is malformed")
                if record.lease_until <= current.lease_until:
                    raise StorageIntegrityError("Trigger claim renewal must extend ownership")
            _update_claim(connection, record)
            if record.status is TriggerClaimStatus.DELIVERED:
                trigger_row = connection.execute(
                    "SELECT * FROM local_triggers WHERE trigger_id = ?",
                    (record.trigger_id,),
                ).fetchone()
                if trigger_row is None:
                    raise StorageNotFoundError(record.trigger_id)
                trigger = _trigger(trigger_row)
                fired_at = record.finished_at
                if fired_at is None:
                    raise StorageIntegrityError("Delivered trigger receipt lacks finished_at")
                updated = replace(
                    trigger,
                    revision=trigger.revision + 1,
                    updated_at=max(trigger.updated_at, fired_at),
                    last_fired_at=(
                        max(trigger.last_fired_at, fired_at)
                        if trigger.last_fired_at is not None
                        else fired_at
                    ),
                )
                _update_trigger(connection, updated)
            return record

        return await self._database.transaction(commit)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local trigger repository is read-only")


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.CONTROL:
        raise StorageConfigurationError("Local trigger repository requires control database")
    database.install_component(
        name="triggers",
        version=_COMPONENT_VERSION,
        statements=(
            _CREATE_TRIGGERS,
            _CREATE_TRIGGER_PROJECT_INDEX,
            _CREATE_TRIGGER_GRAPH_INDEX,
            _CREATE_TRIGGER_OWNER_INDEX,
            _CREATE_TRIGGER_EVENT_INDEX,
            _CREATE_TRIGGER_DUE_INDEX,
            _CREATE_CLAIMS,
            _CREATE_CLAIM_ELIGIBLE_INDEX,
            _CREATE_CLAIM_SCOPE_INDEX,
            _CREATE_CLAIM_TRIGGER_INDEX,
        ),
    )


def _insert_trigger(connection: sqlite3.Connection, record: TriggerRecord) -> None:
    connection.execute(
        """
        INSERT INTO local_triggers(
            trigger_id, graph_id, tenant_id, project_id, org_id, user_id,
            session_id, run_id, node_id, agent_id, scope_key, kind, revision,
            created_at, updated_at, default_inputs_json, name, origin,
            cron_expression, interval_seconds, run_at, event_key, timezone,
            max_overlap_runs, catch_up_missed, active, last_fired_at,
            next_fire_at, metadata_json, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        _trigger_values(record),
    )


def _update_trigger(connection: sqlite3.Connection, record: TriggerRecord) -> None:
    values = _trigger_values(record)
    connection.execute(
        """
        UPDATE local_triggers SET
            graph_id = ?, tenant_id = ?, project_id = ?, org_id = ?, user_id = ?,
            session_id = ?, run_id = ?, node_id = ?, agent_id = ?, scope_key = ?,
            kind = ?, revision = ?, created_at = ?, updated_at = ?,
            default_inputs_json = ?, name = ?, origin = ?, cron_expression = ?,
            interval_seconds = ?, run_at = ?, event_key = ?, timezone = ?,
            max_overlap_runs = ?, catch_up_missed = ?, active = ?, last_fired_at = ?,
            next_fire_at = ?, metadata_json = ?, schema_version = ?
        WHERE trigger_id = ?
        """,
        (*values[1:], values[0]),
    )


def _trigger_values(record: TriggerRecord) -> tuple[object, ...]:
    return (
        record.trigger_id,
        record.graph_id,
        record.scope.tenant_id,
        record.scope.project_id,
        record.scope.org_id,
        record.scope.user_id,
        record.scope.session_id,
        record.scope.run_id,
        record.scope.node_id,
        record.scope.agent_id,
        record.scope.scope_key,
        record.kind.value,
        record.revision,
        record.created_at.isoformat(),
        record.updated_at.isoformat(),
        _json(record.default_inputs),
        record.name,
        record.origin,
        record.cron_expression,
        record.interval_seconds,
        _optional_iso(record.run_at),
        record.event_key,
        record.timezone,
        record.max_overlap_runs,
        int(record.catch_up_missed),
        int(record.active),
        _optional_iso(record.last_fired_at),
        _optional_iso(record.next_fire_at),
        _json(record.metadata),
        record.schema_version,
    )


def _trigger(row: sqlite3.Row) -> TriggerRecord:
    try:
        return TriggerRecord(
            trigger_id=str(row["trigger_id"]),
            graph_id=str(row["graph_id"]),
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
                scope_key=row["scope_key"],
            ),
            kind=TriggerKind(str(row["kind"])),
            revision=int(row["revision"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            default_inputs=_json_object(row["default_inputs_json"]),
            name=row["name"],
            origin=str(row["origin"]),
            cron_expression=row["cron_expression"],
            interval_seconds=row["interval_seconds"],
            run_at=_optional_time(row["run_at"]),
            event_key=row["event_key"],
            timezone=row["timezone"],
            max_overlap_runs=row["max_overlap_runs"],
            catch_up_missed=bool(row["catch_up_missed"]),
            active=bool(row["active"]),
            last_fired_at=_optional_time(row["last_fired_at"]),
            next_fire_at=_optional_time(row["next_fire_at"]),
            metadata=_json_object(row["metadata_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local trigger row is malformed") from exc


def _trigger_identity(record: TriggerRecord) -> tuple[object, ...]:
    return (
        record.trigger_id,
        record.graph_id,
        record.scope,
        record.created_at,
        record.origin,
        record.schema_version,
    )


def _insert_claim(connection: sqlite3.Connection, record: TriggerClaimRecord) -> None:
    try:
        connection.execute(
            """
            INSERT INTO local_trigger_claims(
                fire_id, trigger_id, tenant_id, project_id, org_id, user_id,
                session_id, graph_id, agent_id, scope_key, scheduled_for, status,
                attempts, revision, updated_at, worker_id, lease_until, retry_at,
                run_id, last_error, skip_reason, finished_at, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _claim_values(record),
        )
    except sqlite3.IntegrityError as exc:
        raise StorageIntegrityError("Trigger occurrence identity conflicts") from exc


def _update_claim(connection: sqlite3.Connection, record: TriggerClaimRecord) -> None:
    values = _claim_values(record)
    connection.execute(
        """
        UPDATE local_trigger_claims SET
            trigger_id = ?, tenant_id = ?, project_id = ?, org_id = ?, user_id = ?,
            session_id = ?, graph_id = ?, agent_id = ?, scope_key = ?,
            scheduled_for = ?, status = ?, attempts = ?, revision = ?, updated_at = ?,
            worker_id = ?, lease_until = ?, retry_at = ?, run_id = ?, last_error = ?,
            skip_reason = ?, finished_at = ?, schema_version = ?
        WHERE fire_id = ?
        """,
        (*values[1:], values[0]),
    )


def _claim_values(record: TriggerClaimRecord) -> tuple[object, ...]:
    return (
        record.fire_id,
        record.trigger_id,
        record.scope.tenant_id,
        record.scope.project_id,
        record.scope.org_id,
        record.scope.user_id,
        record.scope.session_id,
        record.scope.graph_id,
        record.scope.agent_id,
        record.scope.scope_key,
        record.scheduled_for.isoformat(),
        record.status.value,
        record.attempts,
        record.revision,
        record.updated_at.isoformat(),
        record.worker_id,
        _optional_iso(record.lease_until),
        _optional_iso(record.retry_at),
        record.run_id,
        record.last_error,
        record.skip_reason,
        _optional_iso(record.finished_at),
        record.schema_version,
    )


def _claim(row: sqlite3.Row) -> TriggerClaimRecord:
    try:
        return TriggerClaimRecord(
            fire_id=str(row["fire_id"]),
            trigger_id=str(row["trigger_id"]),
            scope=StorageScope(
                tenant_id=row["tenant_id"],
                project_id=row["project_id"],
                org_id=row["org_id"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                graph_id=row["graph_id"],
                agent_id=row["agent_id"],
                scope_key=row["scope_key"],
            ),
            scheduled_for=datetime.fromisoformat(str(row["scheduled_for"])),
            status=TriggerClaimStatus(str(row["status"])),
            attempts=int(row["attempts"]),
            revision=int(row["revision"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            worker_id=row["worker_id"],
            lease_until=_optional_time(row["lease_until"]),
            retry_at=_optional_time(row["retry_at"]),
            run_id=row["run_id"],
            last_error=row["last_error"],
            skip_reason=row["skip_reason"],
            finished_at=_optional_time(row["finished_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted local trigger claim row is malformed") from exc


def _claim_identity(record: TriggerClaimRecord) -> tuple[object, ...]:
    return (
        record.fire_id,
        record.trigger_id,
        record.scope,
        record.scheduled_for,
        record.attempts,
        record.schema_version,
    )


def _advance_trigger(
    trigger: TriggerRecord,
    *,
    scheduled_for: datetime,
    now: datetime,
) -> TriggerRecord:
    if trigger.kind is TriggerKind.ONE_SHOT:
        next_fire_at = None
        active = False
    else:
        next_fire_at = _next_recurrence(trigger, scheduled_for)
        if not trigger.catch_up_missed:
            while next_fire_at is not None and next_fire_at <= now:
                next_fire_at = _next_recurrence(trigger, next_fire_at)
        active = trigger.active
    return replace(
        trigger,
        revision=trigger.revision + 1,
        updated_at=max(trigger.updated_at, now),
        active=active,
        next_fire_at=next_fire_at,
    )


def _next_recurrence(trigger: TriggerRecord, after: datetime) -> datetime | None:
    if trigger.kind is TriggerKind.INTERVAL:
        if trigger.interval_seconds is None:
            raise StorageIntegrityError("Interval trigger lacks interval_seconds")
        return after + timedelta(seconds=trigger.interval_seconds)
    if trigger.kind is TriggerKind.CRON:
        if trigger.cron_expression is None:
            raise StorageIntegrityError("Cron trigger lacks cron_expression")
        zone = gettz(trigger.timezone or "UTC")
        if zone is None:
            raise StorageIntegrityError("Cron trigger timezone is invalid")
        try:
            local_after = after.astimezone(zone)
            return croniter(trigger.cron_expression, local_after).get_next(datetime).astimezone(UTC)
        except (TypeError, ValueError, KeyError) as exc:
            raise StorageIntegrityError("Cron trigger expression is invalid") from exc
    return None


def _validate_schedule(record: TriggerRecord) -> None:
    zone = gettz(record.timezone or "UTC")
    if zone is None:
        raise StorageConfigurationError("Trigger timezone is invalid")
    if record.kind is TriggerKind.CRON and not croniter.is_valid(record.cron_expression):
        raise StorageConfigurationError("Trigger cron expression is invalid")


def _fire_id(trigger_id: str, scheduled_for: datetime) -> str:
    digest = hashlib.sha256(f"{trigger_id}|{scheduled_for.isoformat()}".encode()).hexdigest()[:24]
    return f"trigfire-{digest}"


def _scope_filters(scope: StorageScope, *, alias: str) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    values: list[object] = []
    for name, value in scope.as_filter().items():
        clauses.append(f"{alias}.{name} = ?")
        values.append(value)
    if not clauses:
        raise StorageConfigurationError("Trigger operations require populated scope")
    return clauses, values


def _next_revision(revision: int, expected_revision: int) -> None:
    _expected_revision(expected_revision)
    if revision != expected_revision + 1:
        raise ValueError("record revision must equal expected_revision plus one")


def _expected_revision(expected_revision: int) -> None:
    if (
        isinstance(expected_revision, bool)
        or not isinstance(expected_revision, int)
        or expected_revision < 1
    ):
        raise ValueError("expected_revision must be a positive integer")


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


def _json_object(value: object) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("persisted JSON value must be an object")
    return parsed


def _query_fingerprint(query: TriggerQuery) -> str:
    payload = {
        "scope": query.scope.as_filter(),
        "kinds": [kind.value for kind in query.kinds],
        "active": query.active,
        "event_key": query.event_key,
        "limit": query.page.limit,
    }
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
        raise StorageConfigurationError("Invalid or mismatched trigger cursor") from exc

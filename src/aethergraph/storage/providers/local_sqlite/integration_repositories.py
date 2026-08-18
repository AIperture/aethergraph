"""Transactional local ingress idempotency and external-session bindings."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import sqlite3
from typing import Any

from ...contracts import (
    ExternalSessionBindingRecord,
    ExternalSessionBindingRequest,
    ExternalSessionBindingResult,
    IngressClaimRecord,
    IngressClaimRequest,
    IngressClaimResult,
    IngressClaimStatus,
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
_CREATE_INGRESS_CLAIMS = """
CREATE TABLE local_ingress_claims (
    deployment_id TEXT NOT NULL,
    integration_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    external_event_id TEXT NOT NULL,
    envelope_digest TEXT NOT NULL,
    digest_algorithm TEXT NOT NULL,
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
    claimed_at TEXT NOT NULL,
    status TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    receipt_json TEXT NOT NULL,
    completed_at TEXT,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    PRIMARY KEY(deployment_id, integration_id, idempotency_key),
    UNIQUE(deployment_id, integration_id, external_event_id)
)
"""
_CREATE_INGRESS_SCOPE_INDEX = """
CREATE INDEX ix_local_ingress_claims_scope_status
ON local_ingress_claims(scope_identity, status, claimed_at)
"""
_CREATE_BINDINGS = """
CREATE TABLE local_external_session_bindings (
    binding_id TEXT PRIMARY KEY,
    route_id TEXT NOT NULL,
    build_id TEXT NOT NULL,
    ag_session_id TEXT NOT NULL,
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
    scope_key TEXT NOT NULL,
    revision INTEGER NOT NULL CHECK (revision > 0),
    created_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version > 0),
    UNIQUE(route_id, scope_identity)
)
"""
_CREATE_BINDING_LOOKUP_INDEX = """
CREATE INDEX ix_local_external_bindings_route_scope
ON local_external_session_bindings(route_id, scope_key, scope_identity)
"""
_CREATE_BINDING_SESSION_INDEX = """
CREATE INDEX ix_local_external_bindings_session
ON local_external_session_bindings(ag_session_id)
"""


class LocalIngressIdempotencyRepository:
    """Canonical local ingress claims with single-assignment receipts."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def claim(self, request: IngressClaimRequest) -> IngressClaimResult:
        """Atomically acquire or inspect one canonical ingress identity.

        Key and external-event uniqueness are enforced under one immediate
        transaction. An exact replay returns the authoritative record without
        changing its original claim timestamp or receipt state.

        Examples:
            Acquire ingress work:
                ```python
                result = await repository.claim(request)
                ```

            Replay a completed ingress:
                ```python
                replay = await repository.claim(request)
                ```

        Args:
            request: Canonical ingress identity, digest, scope, and arrival time.

        Returns:
            IngressClaimResult: New ownership or the existing authoritative record.

        Notes:
            Identity reuse with a different event, digest, algorithm, or scope fails
            with `StorageIntegrityError`; no alternate lookup is attempted.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> IngressClaimResult:
            row = connection.execute(
                """
                SELECT * FROM local_ingress_claims
                WHERE deployment_id = ? AND integration_id = ? AND idempotency_key = ?
                """,
                (request.deployment_id, request.integration_id, request.idempotency_key),
            ).fetchone()
            if row is not None:
                record = _ingress(row)
                if not _claim_matches_request(record, request):
                    raise StorageIntegrityError("Ingress idempotency identity conflicts")
                return IngressClaimResult(record=record, acquired=False)
            external = connection.execute(
                """
                SELECT * FROM local_ingress_claims
                WHERE deployment_id = ? AND integration_id = ? AND external_event_id = ?
                """,
                (request.deployment_id, request.integration_id, request.external_event_id),
            ).fetchone()
            if external is not None:
                raise StorageIntegrityError("Ingress external event identity conflicts")
            record = IngressClaimRecord(
                deployment_id=request.deployment_id,
                integration_id=request.integration_id,
                idempotency_key=request.idempotency_key,
                external_event_id=request.external_event_id,
                envelope_digest=request.envelope_digest,
                digest_algorithm=request.digest_algorithm,
                scope=request.scope,
                claimed_at=request.claimed_at,
                status=IngressClaimStatus.PENDING,
                revision=1,
            )
            _insert_ingress(connection, record)
            return IngressClaimResult(record=record, acquired=True)

        return await self._database.transaction(commit)

    async def get(
        self,
        scope: StorageScope,
        deployment_id: str,
        integration_id: str,
        idempotency_key: str,
    ) -> IngressClaimRecord | None:
        """Read one ingress claim by exact key and canonical scope constraints.

        Every populated caller scope dimension is applied in SQL together with the
        deployment, integration, and idempotency key.

        Examples:
            Read a receipt:
                ```python
                record = await repository.get(scope, deployment_id, integration_id, key)
                ```

            Detect a missing key:
                ```python
                assert await repository.get(scope, deployment_id, integration_id, "missing") is None
                ```

        Args:
            scope: Populated canonical scope constraining access.
            deployment_id: Exact deployment identity.
            integration_id: Exact configured integration identity.
            idempotency_key: Exact provider ingress key.

        Returns:
            IngressClaimRecord | None: Authorized record or `None`.

        Notes:
            External event identity is not a fallback lookup path.
        """
        for name, value in (
            ("deployment_id", deployment_id),
            ("integration_id", integration_id),
            ("idempotency_key", idempotency_key),
        ):
            _nonempty(name, value)
        if not scope.as_filter():
            return None
        clauses, values = _scope_filters(scope)
        rows = await self._database.fetch_all(
            "SELECT * FROM local_ingress_claims WHERE deployment_id = ? "
            "AND integration_id = ? AND idempotency_key = ? AND " + " AND ".join(clauses),
            (deployment_id, integration_id, idempotency_key, *values),
        )
        return _ingress(rows[0]) if rows else None

    async def complete(
        self,
        record: IngressClaimRecord,
        expected_revision: int,
    ) -> IngressClaimRecord:
        """Atomically assign one terminal receipt at the exact next revision.

        The pending claim identity and original timestamp are immutable. Receipt,
        completion time, status, and revision advance together exactly once.

        Examples:
            Complete accepted ingress:
                ```python
                stored = await repository.complete(completed, pending.revision)
                ```

            Complete a stable rejection:
                ```python
                stored = await repository.complete(rejected, pending.revision)
                ```

        Args:
            record: Complete terminal next revision with a non-empty receipt.
            expected_revision: Current pending revision required for completion.

        Returns:
            IngressClaimRecord: Newly committed terminal receipt.

        Notes:
            Missing, stale, or already-completed claims fail without rewriting the
            existing receipt.
        """
        self._require_writable()
        _next_revision(record.revision, expected_revision)
        if record.status is not IngressClaimStatus.COMPLETED:
            raise StorageIntegrityError("Ingress completion requires completed status")

        def commit(connection: sqlite3.Connection) -> IngressClaimRecord:
            row = connection.execute(
                """
                SELECT * FROM local_ingress_claims
                WHERE deployment_id = ? AND integration_id = ? AND idempotency_key = ?
                """,
                (record.deployment_id, record.integration_id, record.idempotency_key),
            ).fetchone()
            if row is None:
                raise StorageNotFoundError("Ingress claim does not exist")
            current = _ingress(row)
            if current.revision != expected_revision:
                raise StorageConflictError("Ingress claim revision is stale")
            if current.status is not IngressClaimStatus.PENDING:
                raise StorageConflictError("Ingress receipt is already assigned")
            if not _claim_identity_equal(current, record):
                raise StorageIntegrityError("Ingress claim immutable identity changed")
            updated = connection.execute(
                """
                UPDATE local_ingress_claims SET
                    status = ?, revision = ?, receipt_json = ?, completed_at = ?
                WHERE deployment_id = ? AND integration_id = ? AND idempotency_key = ?
                  AND revision = ? AND status = ?
                """,
                (
                    record.status.value,
                    record.revision,
                    _json(record.receipt),
                    record.completed_at.isoformat() if record.completed_at else None,
                    record.deployment_id,
                    record.integration_id,
                    record.idempotency_key,
                    expected_revision,
                    IngressClaimStatus.PENDING.value,
                ),
            ).rowcount
            if updated != 1:
                raise StorageConflictError("Ingress claim changed during completion")
            return record

        return await self._database.transaction(commit)

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local ingress idempotency repository is read-only")


class LocalExternalSessionBindingRepository:
    """Canonical local external-session bindings with immutable build pinning."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        _install(database)
        self._database = database
        self._mode = database.mode

    async def get_or_create(
        self,
        request: ExternalSessionBindingRequest,
    ) -> ExternalSessionBindingResult:
        """Resolve or create one exact route and external-scope binding.

        Creation and monotonic last-seen advancement are serialized in one immediate
        transaction. Existing build, session, binding, and scope identity is pinned.

        Examples:
            Resolve a conversation:
                ```python
                result = await repository.get_or_create(request)
                ```

            Detect creation ownership:
                ```python
                if (await repository.get_or_create(request)).created:
                    await create_session(request.ag_session_id)
                ```

        Args:
            request: Candidate route, build, AG session, external scope, and timestamp.

        Returns:
            ExternalSessionBindingResult: Authoritative binding and creation ownership.

        Notes:
            Existing identity mismatch or a regressed clock raises
            `StorageIntegrityError`; no replacement binding is created.
        """
        self._require_writable()
        identity = _scope_identity(request.scope)

        def commit(connection: sqlite3.Connection) -> ExternalSessionBindingResult:
            row = connection.execute(
                """
                SELECT * FROM local_external_session_bindings
                WHERE route_id = ? AND scope_identity = ?
                """,
                (request.route_id, identity),
            ).fetchone()
            if row is not None:
                current = _binding(row)
                if not _binding_matches_request(current, request):
                    raise StorageIntegrityError("External session binding identity conflicts")
                if request.now < current.last_seen_at:
                    raise StorageIntegrityError("External session last_seen_at moved backward")
                if request.now == current.last_seen_at:
                    return ExternalSessionBindingResult(record=current, created=False)
                updated = ExternalSessionBindingRecord(
                    binding_id=current.binding_id,
                    route_id=current.route_id,
                    build_id=current.build_id,
                    ag_session_id=current.ag_session_id,
                    scope=current.scope,
                    revision=current.revision + 1,
                    created_at=current.created_at,
                    last_seen_at=request.now,
                    metadata=current.metadata,
                    schema_version=current.schema_version,
                )
                changed = connection.execute(
                    """
                    UPDATE local_external_session_bindings
                    SET revision = ?, last_seen_at = ?
                    WHERE binding_id = ? AND revision = ?
                    """,
                    (
                        updated.revision,
                        updated.last_seen_at.isoformat(),
                        updated.binding_id,
                        current.revision,
                    ),
                ).rowcount
                if changed != 1:
                    raise StorageConflictError(
                        "External session binding changed during last-seen update"
                    )
                return ExternalSessionBindingResult(record=updated, created=False)
            collision = connection.execute(
                "SELECT * FROM local_external_session_bindings WHERE binding_id = ?",
                (request.binding_id,),
            ).fetchone()
            if collision is not None:
                raise StorageIntegrityError("External binding identity conflicts")
            record = ExternalSessionBindingRecord(
                binding_id=request.binding_id,
                route_id=request.route_id,
                build_id=request.build_id,
                ag_session_id=request.ag_session_id,
                scope=request.scope,
                revision=1,
                created_at=request.now,
                last_seen_at=request.now,
            )
            _insert_binding(connection, record)
            return ExternalSessionBindingResult(record=record, created=True)

        return await self._database.transaction(commit)

    async def get(
        self,
        scope: StorageScope,
        route_id: str,
    ) -> ExternalSessionBindingRecord | None:
        """Read one binding by exact route and canonical external scope.

        The opaque route-authored `scope_key` and every owner dimension are matched
        through the provider's exact canonical scope identity.

        Examples:
            Read a binding:
                ```python
                binding = await repository.get(external_scope, "route-1")
                ```

            Detect an unbound scope:
                ```python
                assert await repository.get(new_scope, "route-1") is None
                ```

        Args:
            scope: Exact canonical external scope containing `scope_key`.
            route_id: Exact manifest route identity.

        Returns:
            ExternalSessionBindingRecord | None: Current binding or `None`.

        Notes:
            Host route and external-identity DTOs are not imported or interpreted.
        """
        _nonempty("route_id", route_id)
        scope.require("scope_key")
        identity = _scope_identity(scope)
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_external_session_bindings
            WHERE route_id = ? AND scope_identity = ?
            """,
            (route_id, identity),
        )
        return _binding(rows[0]) if rows else None

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local external session repository is read-only")


def _install(database: LocalSQLiteDatabase) -> None:
    if database.role is not LocalDatabaseRole.CONTROL:
        raise StorageConfigurationError("Local integration repositories require control database")
    database.install_component(
        name="integration",
        version=_COMPONENT_VERSION,
        statements=(
            _CREATE_INGRESS_CLAIMS,
            _CREATE_INGRESS_SCOPE_INDEX,
            _CREATE_BINDINGS,
            _CREATE_BINDING_LOOKUP_INDEX,
            _CREATE_BINDING_SESSION_INDEX,
        ),
    )


def _insert_ingress(connection: sqlite3.Connection, record: IngressClaimRecord) -> None:
    connection.execute(
        """
        INSERT INTO local_ingress_claims(
            deployment_id, integration_id, idempotency_key, external_event_id,
            envelope_digest, digest_algorithm, scope_identity, tenant_id, project_id,
            org_id, user_id, session_id, run_id, graph_id, node_id, agent_id,
            scope_key, claimed_at, status, revision, receipt_json, completed_at,
            schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.deployment_id,
            record.integration_id,
            record.idempotency_key,
            record.external_event_id,
            record.envelope_digest,
            record.digest_algorithm,
            _scope_identity(record.scope),
            *_scope_values(record.scope),
            record.claimed_at.isoformat(),
            record.status.value,
            record.revision,
            _json(record.receipt),
            record.completed_at.isoformat() if record.completed_at else None,
            record.schema_version,
        ),
    )


def _ingress(row: sqlite3.Row) -> IngressClaimRecord:
    try:
        return IngressClaimRecord(
            deployment_id=str(row["deployment_id"]),
            integration_id=str(row["integration_id"]),
            idempotency_key=str(row["idempotency_key"]),
            external_event_id=str(row["external_event_id"]),
            envelope_digest=str(row["envelope_digest"]),
            digest_algorithm=str(row["digest_algorithm"]),
            scope=_scope(row),
            claimed_at=datetime.fromisoformat(str(row["claimed_at"])),
            status=IngressClaimStatus(str(row["status"])),
            revision=int(row["revision"]),
            receipt=_json_object(row["receipt_json"]),
            completed_at=_optional_time(row["completed_at"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local ingress claim is malformed") from exc


def _claim_matches_request(record: IngressClaimRecord, request: IngressClaimRequest) -> bool:
    return (
        record.deployment_id == request.deployment_id
        and record.integration_id == request.integration_id
        and record.idempotency_key == request.idempotency_key
        and record.external_event_id == request.external_event_id
        and record.envelope_digest == request.envelope_digest
        and record.digest_algorithm == request.digest_algorithm
        and record.scope == request.scope
    )


def _claim_identity_equal(left: IngressClaimRecord, right: IngressClaimRecord) -> bool:
    return (
        left.deployment_id == right.deployment_id
        and left.integration_id == right.integration_id
        and left.idempotency_key == right.idempotency_key
        and left.external_event_id == right.external_event_id
        and left.envelope_digest == right.envelope_digest
        and left.digest_algorithm == right.digest_algorithm
        and left.scope == right.scope
        and left.claimed_at == right.claimed_at
        and left.schema_version == right.schema_version
    )


def _insert_binding(
    connection: sqlite3.Connection,
    record: ExternalSessionBindingRecord,
) -> None:
    connection.execute(
        """
        INSERT INTO local_external_session_bindings(
            binding_id, route_id, build_id, ag_session_id, scope_identity, tenant_id,
            project_id, org_id, user_id, session_id, run_id, graph_id, node_id,
            agent_id, scope_key, revision, created_at, last_seen_at, metadata_json,
            schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.binding_id,
            record.route_id,
            record.build_id,
            record.ag_session_id,
            _scope_identity(record.scope),
            *_scope_values(record.scope),
            record.revision,
            record.created_at.isoformat(),
            record.last_seen_at.isoformat(),
            _json(record.metadata),
            record.schema_version,
        ),
    )


def _binding(row: sqlite3.Row) -> ExternalSessionBindingRecord:
    try:
        return ExternalSessionBindingRecord(
            binding_id=str(row["binding_id"]),
            route_id=str(row["route_id"]),
            build_id=str(row["build_id"]),
            ag_session_id=str(row["ag_session_id"]),
            scope=_scope(row),
            revision=int(row["revision"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            last_seen_at=datetime.fromisoformat(str(row["last_seen_at"])),
            metadata=_json_object(row["metadata_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted external session binding is malformed") from exc


def _binding_matches_request(
    record: ExternalSessionBindingRecord,
    request: ExternalSessionBindingRequest,
) -> bool:
    return (
        record.binding_id == request.binding_id
        and record.route_id == request.route_id
        and record.build_id == request.build_id
        and record.ag_session_id == request.ag_session_id
        and record.scope == request.scope
    )


def _scope_values(scope: StorageScope) -> tuple[str | None, ...]:
    return tuple(getattr(scope, name) for name in _SCOPE_COLUMNS)


def _scope(row: sqlite3.Row) -> StorageScope:
    try:
        return StorageScope(**{name: row[name] for name in _SCOPE_COLUMNS})
    except (TypeError, ValueError, KeyError) as exc:
        raise StorageIntegrityError("Persisted integration scope is malformed") from exc


def _scope_identity(scope: StorageScope) -> str:
    if not scope.as_filter():
        raise StorageConfigurationError("Integration operations require populated scope")
    return _json(scope.as_filter())


def _scope_filters(scope: StorageScope) -> tuple[list[str], list[object]]:
    clauses: list[str] = []
    values: list[object] = []
    for name, value in scope.as_filter().items():
        clauses.append(f"{name} = ?")
        values.append(value)
    if not clauses:
        raise StorageConfigurationError("Integration operations require populated scope")
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


def _json_object(value: object) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("persisted JSON value must be an object")
    return parsed

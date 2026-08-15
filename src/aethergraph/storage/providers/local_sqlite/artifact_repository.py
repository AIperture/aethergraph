"""Canonical artifact metadata, occurrence, and lineage persistence."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import sqlite3

from ...contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    Page,
    PageRequest,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalSQLiteDatabase

_ARTIFACT_COMPONENT_VERSION = 1
_CREATE_ARTIFACTS = """
CREATE TABLE local_artifacts (
    artifact_id TEXT PRIMARY KEY,
    owner_scope_identity TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    hash_algorithm TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    media_type TEXT NOT NULL,
    kind TEXT NOT NULL,
    blob_locator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    preview_locator TEXT,
    original_filename TEXT,
    provider_version TEXT,
    labels_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL
)
"""
_CREATE_ARTIFACT_SCOPE_INDEX = """
CREATE INDEX ix_local_artifacts_scope ON local_artifacts(owner_scope_identity, artifact_id)
"""
_CREATE_OCCURRENCES = """
CREATE TABLE local_artifact_occurrences (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    occurrence_id TEXT NOT NULL UNIQUE,
    artifact_id TEXT NOT NULL REFERENCES local_artifacts(artifact_id),
    scope_identity TEXT NOT NULL,
    action TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    tool_name TEXT,
    tool_version TEXT,
    labels_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL
)
"""
_CREATE_OCCURRENCE_SCOPE_INDEX = """
CREATE INDEX ix_local_occurrences_scope
ON local_artifact_occurrences(scope_identity, sequence)
"""
_CREATE_OCCURRENCE_ARTIFACT_INDEX = """
CREATE INDEX ix_local_occurrences_artifact
ON local_artifact_occurrences(scope_identity, artifact_id, sequence)
"""
_CREATE_RELATIONS = """
CREATE TABLE local_artifact_relations (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    relation_id TEXT NOT NULL UNIQUE,
    source_artifact_id TEXT NOT NULL REFERENCES local_artifacts(artifact_id),
    target_artifact_id TEXT NOT NULL REFERENCES local_artifacts(artifact_id),
    relation_kind TEXT NOT NULL,
    scope_identity TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    schema_version INTEGER NOT NULL
)
"""
_CREATE_RELATION_SOURCE_INDEX = """
CREATE INDEX ix_local_relations_source
ON local_artifact_relations(scope_identity, source_artifact_id, sequence)
"""
_CREATE_RELATION_TARGET_INDEX = """
CREATE INDEX ix_local_relations_target
ON local_artifact_relations(scope_identity, target_artifact_id, sequence)
"""


class LocalArtifactRepository:
    """Immutable artifact metadata with normalized occurrence and lineage tables."""

    def __init__(self, *, database: LocalSQLiteDatabase) -> None:
        self._database = database
        self._mode = database.mode
        database.install_component(
            name="artifacts",
            version=_ARTIFACT_COMPONENT_VERSION,
            statements=(
                _CREATE_ARTIFACTS,
                _CREATE_ARTIFACT_SCOPE_INDEX,
                _CREATE_OCCURRENCES,
                _CREATE_OCCURRENCE_SCOPE_INDEX,
                _CREATE_OCCURRENCE_ARTIFACT_INDEX,
                _CREATE_RELATIONS,
                _CREATE_RELATION_SOURCE_INDEX,
                _CREATE_RELATION_TARGET_INDEX,
            ),
        )

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        """Idempotently commit immutable artifact content metadata.

        The stable artifact ID is globally unique inside the provider. Exact retries
        return the stored record; any changed immutable field fails atomically.

        Examples:
            Commit new metadata:
                ```python
                stored = await repository.put(record)
                ```

            Retry the same record:
                ```python
                assert await repository.put(record) == stored
                ```

        Args:
            record: Complete canonical artifact metadata referencing committed blob content.

        Returns:
            ArtifactRecord: Authoritative immutable metadata.

        Notes:
            Blob bytes, occurrences, and lineage are not duplicated in this row.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> ArtifactRecord:
            existing = connection.execute(
                "SELECT * FROM local_artifacts WHERE artifact_id = ?",
                (record.artifact_id,),
            ).fetchone()
            if existing is not None:
                stored = _artifact(existing)
                if stored != record:
                    raise StorageIntegrityError(
                        f"Artifact identity {record.artifact_id!r} has conflicting metadata"
                    )
                return stored
            connection.execute(
                """
                INSERT INTO local_artifacts(
                    artifact_id, owner_scope_identity, content_hash, hash_algorithm,
                    size_bytes, media_type, kind, blob_locator, created_at,
                    preview_locator, original_filename, provider_version, labels_json,
                    schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.artifact_id,
                    _scope_identity(record.owner_scope),
                    record.content_hash,
                    record.hash_algorithm,
                    record.size_bytes,
                    record.media_type,
                    record.kind,
                    record.blob_locator,
                    record.created_at.isoformat(),
                    record.preview_locator,
                    record.original_filename,
                    record.provider_version,
                    _json(record.labels),
                    record.schema_version,
                ),
            )
            return record

        return await self._database.transaction(commit)

    async def get(
        self,
        scope: StorageScope,
        artifact_id: str,
    ) -> ArtifactRecord | None:
        """Read one exact immutable artifact within owner scope.

        Both stable artifact identity and canonical owner scope must match before any
        metadata is returned.

        Examples:
            Read existing metadata:
                ```python
                artifact = await repository.get(scope, "artifact-1")
                ```

            Detect an unauthorized or missing identity:
                ```python
                assert await repository.get(other_scope, "artifact-1") is None
                ```

        Args:
            scope: Exact canonical artifact owner scope.
            artifact_id: Stable artifact identity.

        Returns:
            ArtifactRecord | None: Matching metadata or `None`.

        Notes:
            This method never hydrates blob content, occurrences, or lineage.
        """
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_artifacts
            WHERE artifact_id = ? AND owner_scope_identity = ?
            """,
            (artifact_id, _scope_identity(scope)),
        )
        return _artifact(rows[0]) if rows else None

    async def record_occurrence(
        self,
        occurrence: ArtifactOccurrence,
    ) -> ArtifactOccurrence:
        """Idempotently commit one authorized artifact occurrence.

        The referenced artifact must exist and every populated owner-scope dimension
        must match the occurrence execution scope.

        Examples:
            Record artifact production:
                ```python
                stored = await repository.record_occurrence(occurrence)
                ```

            Retry an exact occurrence:
                ```python
                assert await repository.record_occurrence(occurrence) == stored
                ```

        Args:
            occurrence: Complete execution-context occurrence record.

        Returns:
            ArtifactOccurrence: Authoritative normalized occurrence.

        Notes:
            Missing or unauthorized artifacts raise `StorageNotFoundError`; conflicting
            duplicate IDs raise `StorageIntegrityError`.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> ArtifactOccurrence:
            existing = connection.execute(
                "SELECT * FROM local_artifact_occurrences WHERE occurrence_id = ?",
                (occurrence.occurrence_id,),
            ).fetchone()
            if existing is not None:
                stored = _occurrence(existing)
                if stored != occurrence:
                    raise StorageIntegrityError(
                        f"Occurrence identity {occurrence.occurrence_id!r} conflicts"
                    )
                return stored
            artifact = connection.execute(
                "SELECT owner_scope_identity FROM local_artifacts WHERE artifact_id = ?",
                (occurrence.artifact_id,),
            ).fetchone()
            if artifact is None or not _scope_authorizes(
                _scope(str(artifact[0])), occurrence.scope
            ):
                raise StorageNotFoundError(occurrence.artifact_id)
            connection.execute(
                """
                INSERT INTO local_artifact_occurrences(
                    occurrence_id, artifact_id, scope_identity, action, occurred_at,
                    tool_name, tool_version, labels_json, metrics_json, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    occurrence.occurrence_id,
                    occurrence.artifact_id,
                    _scope_identity(occurrence.scope),
                    occurrence.action.value,
                    occurrence.occurred_at.isoformat(),
                    occurrence.tool_name,
                    occurrence.tool_version,
                    _json(occurrence.labels),
                    _json(occurrence.metrics),
                    occurrence.schema_version,
                ),
            )
            return occurrence

        return await self._database.transaction(commit)

    async def list_occurrences(
        self,
        scope: StorageScope,
        page: PageRequest,
        artifact_id: str | None = None,
    ) -> Page[ArtifactOccurrence]:
        """List one bounded descending page of exact-scope occurrences.

        Scope and optional artifact identity filter before the provider sequence
        cursor, producing stable pages without offset scans.

        Examples:
            List run occurrences:
                ```python
                page = await repository.list_occurrences(run_scope, PageRequest())
                ```

            List uses of one artifact:
                ```python
                page = await repository.list_occurrences(scope, PageRequest(), artifact_id)
                ```

        Args:
            scope: Exact canonical execution scope.
            page: Bounded opaque cursor request.
            artifact_id: Optional exact artifact identity filter.

        Returns:
            Page[ArtifactOccurrence]: Stable matching occurrences and next cursor.

        Notes:
            Content metadata and blob bytes are not hydrated.
        """
        identity = _scope_identity(scope)
        fingerprint = _fingerprint("occurrence", identity, artifact_id or "")
        clauses = ["scope_identity = ?"]
        values: list[object] = [identity]
        if artifact_id is not None:
            clauses.append("artifact_id = ?")
            values.append(artifact_id)
        if page.cursor is not None:
            clauses.append("sequence < ?")
            values.append(_decode_cursor(page.cursor, fingerprint))
        values.append(page.limit + 1)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_artifact_occurrences WHERE {' AND '.join(clauses)} "
            "ORDER BY sequence DESC LIMIT ?",
            values,
        )
        selected = rows[: page.limit]
        return Page(
            items=tuple(_occurrence(row) for row in selected),
            next_cursor=(
                _encode_cursor(fingerprint, int(selected[-1]["sequence"]))
                if len(rows) > page.limit
                else None
            ),
        )

    async def add_relation(self, relation: ArtifactRelation) -> ArtifactRelation:
        """Idempotently commit one authorized directed lineage edge.

        Both endpoint artifacts must exist in the relation's exact owner scope before
        the normalized edge is inserted.

        Examples:
            Record a derivation edge:
                ```python
                stored = await repository.add_relation(relation)
                ```

            Retry the same edge:
                ```python
                assert await repository.add_relation(relation) == stored
                ```

        Args:
            relation: Complete typed directed lineage relation.

        Returns:
            ArtifactRelation: Authoritative normalized relation.

        Notes:
            Missing or cross-scope endpoints raise `StorageNotFoundError`.
        """
        self._require_writable()

        def commit(connection: sqlite3.Connection) -> ArtifactRelation:
            existing = connection.execute(
                "SELECT * FROM local_artifact_relations WHERE relation_id = ?",
                (relation.relation_id,),
            ).fetchone()
            if existing is not None:
                stored = _relation(existing)
                if stored != relation:
                    raise StorageIntegrityError(
                        f"Relation identity {relation.relation_id!r} conflicts"
                    )
                return stored
            expected_scope = _scope_identity(relation.scope)
            endpoints = connection.execute(
                """
                SELECT artifact_id FROM local_artifacts
                WHERE owner_scope_identity = ? AND artifact_id IN (?, ?)
                """,
                (
                    expected_scope,
                    relation.source_artifact_id,
                    relation.target_artifact_id,
                ),
            ).fetchall()
            if {str(row[0]) for row in endpoints} != {
                relation.source_artifact_id,
                relation.target_artifact_id,
            }:
                raise StorageNotFoundError("artifact lineage endpoint")
            connection.execute(
                """
                INSERT INTO local_artifact_relations(
                    relation_id, source_artifact_id, target_artifact_id, relation_kind,
                    scope_identity, created_at, metadata_json, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.relation_id,
                    relation.source_artifact_id,
                    relation.target_artifact_id,
                    relation.kind.value,
                    expected_scope,
                    relation.created_at.isoformat(),
                    _json(relation.metadata),
                    relation.schema_version,
                ),
            )
            return relation

        return await self._database.transaction(commit)

    async def list_relations(
        self,
        scope: StorageScope,
        artifact_id: str,
        page: PageRequest,
    ) -> Page[ArtifactRelation]:
        """List bounded incoming and outgoing lineage for one exact artifact.

        Exact owner scope and artifact identity filter before descending provider
        sequence pagination.

        Examples:
            Read initial lineage:
                ```python
                page = await repository.list_relations(scope, artifact_id, PageRequest())
                ```

            Continue lineage:
                ```python
                page = await repository.list_relations(scope, artifact_id, next_page)
                ```

        Args:
            scope: Exact canonical endpoint owner scope.
            artifact_id: Artifact whose incoming and outgoing edges are requested.
            page: Bounded opaque cursor request.

        Returns:
            Page[ArtifactRelation]: Stable matching relations and next cursor.

        Notes:
            The query never hydrates endpoint metadata or blob content.
        """
        identity = _scope_identity(scope)
        fingerprint = _fingerprint("relation", identity, artifact_id)
        values: list[object] = [identity, artifact_id, artifact_id]
        cursor_clause = ""
        if page.cursor is not None:
            cursor_clause = " AND sequence < ?"
            values.append(_decode_cursor(page.cursor, fingerprint))
        values.append(page.limit + 1)
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_artifact_relations
            WHERE scope_identity = ?
              AND (source_artifact_id = ? OR target_artifact_id = ?)
            """
            + cursor_clause
            + " ORDER BY sequence DESC LIMIT ?",
            values,
        )
        selected = rows[: page.limit]
        return Page(
            items=tuple(_relation(row) for row in selected),
            next_cursor=(
                _encode_cursor(fingerprint, int(selected[-1]["sequence"]))
                if len(rows) > page.limit
                else None
            ),
        )

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local artifact repository is read-only")


def _artifact(row: sqlite3.Row) -> ArtifactRecord:
    try:
        return ArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            content_hash=str(row["content_hash"]),
            hash_algorithm=str(row["hash_algorithm"]),
            size_bytes=int(row["size_bytes"]),
            media_type=str(row["media_type"]),
            kind=str(row["kind"]),
            blob_locator=str(row["blob_locator"]),
            owner_scope=_scope(str(row["owner_scope_identity"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            preview_locator=row["preview_locator"],
            original_filename=row["original_filename"],
            provider_version=row["provider_version"],
            labels=json.loads(row["labels_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local artifact row is malformed") from exc


def _occurrence(row: sqlite3.Row) -> ArtifactOccurrence:
    try:
        return ArtifactOccurrence(
            occurrence_id=str(row["occurrence_id"]),
            artifact_id=str(row["artifact_id"]),
            scope=_scope(str(row["scope_identity"])),
            action=ArtifactAction(str(row["action"])),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            tool_name=row["tool_name"],
            tool_version=row["tool_version"],
            labels=json.loads(row["labels_json"]),
            metrics=json.loads(row["metrics_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local occurrence row is malformed") from exc


def _relation(row: sqlite3.Row) -> ArtifactRelation:
    try:
        return ArtifactRelation(
            relation_id=str(row["relation_id"]),
            source_artifact_id=str(row["source_artifact_id"]),
            target_artifact_id=str(row["target_artifact_id"]),
            kind=ArtifactRelationKind(str(row["relation_kind"])),
            scope=_scope(str(row["scope_identity"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            metadata=json.loads(row["metadata_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local relation row is malformed") from exc


def _scope_identity(scope: StorageScope) -> str:
    return json.dumps(scope.as_filter(), sort_keys=True, separators=(",", ":"))


def _scope(identity: str) -> StorageScope:
    try:
        payload = json.loads(identity)
        if not isinstance(payload, dict):
            raise TypeError("scope")
        return StorageScope(**payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted artifact scope is malformed") from exc


def _scope_authorizes(owner: StorageScope, operation: StorageScope) -> bool:
    return all(getattr(operation, name) == value for name, value in owner.as_filter().items())


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


def _encode_cursor(fingerprint: str, sequence: int) -> str:
    payload = json.dumps(
        {"fingerprint": fingerprint, "sequence": sequence},
        sort_keys=True,
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, fingerprint: str) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, dict) or set(payload) != {"fingerprint", "sequence"}:
            raise ValueError("cursor payload")
        if payload["fingerprint"] != fingerprint:
            raise ValueError("cursor context")
        sequence = payload["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("cursor sequence")
        return sequence
    except (
        binascii.Error,
        ValueError,
        TypeError,
        KeyError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise StorageConfigurationError("Invalid or mismatched artifact cursor") from exc

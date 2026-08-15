"""Canonical artifact metadata, occurrence, and lineage persistence."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import sqlite3

from ...contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    ArtifactRetentionRecord,
    Page,
    PageRequest,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from .database import LocalSQLiteDatabase

_ARTIFACT_COMPONENT_VERSION = 3
_MAX_ARTIFACT_BATCH = 500
_SCOPE_FIELDS = tuple(StorageScope.__dataclass_fields__)
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
_CREATE_ARTIFACT_KIND_INDEX = """
CREATE INDEX ix_local_artifacts_kind
ON local_artifacts(owner_scope_identity, kind, artifact_id)
"""
_CREATE_ARTIFACT_LABELS = """
CREATE TABLE local_artifact_labels (
    artifact_id TEXT NOT NULL REFERENCES local_artifacts(artifact_id) ON DELETE CASCADE,
    label_key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (artifact_id, label_key)
)
"""
_CREATE_ARTIFACT_LABEL_INDEX = """
CREATE INDEX ix_local_artifact_labels_exact
ON local_artifact_labels(label_key, value_json, artifact_id)
"""
_CREATE_ARTIFACT_TAGS = """
CREATE TABLE local_artifact_tags (
    artifact_id TEXT NOT NULL REFERENCES local_artifacts(artifact_id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (artifact_id, tag)
)
"""
_CREATE_ARTIFACT_TAG_INDEX = """
CREATE INDEX ix_local_artifact_tags_exact
ON local_artifact_tags(tag, artifact_id)
"""
_CREATE_ARTIFACT_RETENTION = """
CREATE TABLE local_artifact_retention (
    artifact_id TEXT PRIMARY KEY REFERENCES local_artifacts(artifact_id) ON DELETE CASCADE,
    owner_scope_identity TEXT NOT NULL,
    pinned INTEGER NOT NULL CHECK (pinned IN (0, 1)),
    revision INTEGER NOT NULL CHECK (revision >= 1),
    updated_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL
)
"""
_CREATE_ARTIFACT_RETENTION_SCOPE_INDEX = """
CREATE INDEX ix_local_artifact_retention_scope
ON local_artifact_retention(owner_scope_identity, artifact_id)
"""
_CREATE_ARTIFACT_RETENTION_PIN_INDEX = """
CREATE INDEX ix_local_artifact_retention_pin
ON local_artifact_retention(owner_scope_identity, pinned, artifact_id)
"""
_CREATE_OCCURRENCES = """
CREATE TABLE local_artifact_occurrences (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    occurrence_id TEXT NOT NULL UNIQUE,
    artifact_id TEXT NOT NULL REFERENCES local_artifacts(artifact_id),
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
ON local_artifact_occurrences(artifact_id, sequence)
"""
_CREATE_OCCURRENCE_SCOPE_DIMENSION_INDEXES = tuple(
    f"CREATE INDEX ix_local_occurrences_{name[:-3]} "
    f"ON local_artifact_occurrences({name}, sequence)"
    for name in _SCOPE_FIELDS
)
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
                _CREATE_ARTIFACT_KIND_INDEX,
                _CREATE_ARTIFACT_LABELS,
                _CREATE_ARTIFACT_LABEL_INDEX,
                _CREATE_ARTIFACT_TAGS,
                _CREATE_ARTIFACT_TAG_INDEX,
                _CREATE_ARTIFACT_RETENTION,
                _CREATE_ARTIFACT_RETENTION_SCOPE_INDEX,
                _CREATE_ARTIFACT_RETENTION_PIN_INDEX,
                _CREATE_OCCURRENCES,
                _CREATE_OCCURRENCE_SCOPE_INDEX,
                _CREATE_OCCURRENCE_ARTIFACT_INDEX,
                *_CREATE_OCCURRENCE_SCOPE_DIMENSION_INDEXES,
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
            connection.executemany(
                """
                INSERT INTO local_artifact_labels(artifact_id, label_key, value_json)
                VALUES (?, ?, ?)
                """,
                ((record.artifact_id, key, _json(value)) for key, value in record.labels.items()),
            )
            connection.executemany(
                "INSERT INTO local_artifact_tags(artifact_id, tag) VALUES (?, ?)",
                ((record.artifact_id, tag) for tag in _artifact_tags(record.labels)),
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

    async def get_many(
        self,
        scope: StorageScope,
        artifact_ids: Sequence[str],
    ) -> tuple[ArtifactRecord | None, ...]:
        """Batch-read immutable artifact metadata in one bounded query.

        Exact owner scope filters before hydration, while duplicate and missing input
        identities retain their original result positions.

        Examples:
            Hydrate a page:
                ```python
                rows = await repository.get_many(scope, artifact_ids)
                ```

            Preserve a missing slot:
                ```python
                assert (await repository.get_many(scope, ("known", "missing")))[1] is None
                ```

        Args:
            scope: Exact canonical artifact owner scope.
            artifact_ids: At most 500 ordered stable artifact identities.

        Returns:
            tuple[ArtifactRecord | None, ...]: One exact result per input position.

        Notes:
            Empty input returns an empty tuple without querying SQLite.
        """
        requested = _batch_artifact_ids(artifact_ids)
        if not requested:
            return ()
        unique = tuple(dict.fromkeys(requested))
        placeholders = ",".join("?" for _item in unique)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_artifacts WHERE owner_scope_identity = ? "
            f"AND artifact_id IN ({placeholders})",
            (_scope_identity(scope), *unique),
        )
        hydrated = {str(row["artifact_id"]): _artifact(row) for row in rows}
        return tuple(hydrated.get(artifact_id) for artifact_id in requested)

    async def get_retention(
        self,
        scope: StorageScope,
        artifact_id: str,
    ) -> ArtifactRetentionRecord | None:
        """Read one exact artifact retention record.

        Retention is scope constrained and remains separate from immutable artifact
        metadata and occurrence history.

        Examples:
            Read current retention:
                ```python
                retention = await repository.get_retention(scope, "artifact-1")
                ```

            Detect no explicit pin:
                ```python
                assert await repository.get_retention(scope, "artifact-1") is None
                ```

        Args:
            scope: Exact canonical artifact owner scope.
            artifact_id: Stable artifact identity.

        Returns:
            ArtifactRetentionRecord | None: Current retention state or `None`.

        Notes:
            A foreign scope is indistinguishable from absence.
        """
        rows = await self._database.fetch_all(
            """
            SELECT * FROM local_artifact_retention
            WHERE artifact_id = ? AND owner_scope_identity = ?
            """,
            (artifact_id, _scope_identity(scope)),
        )
        return _retention(rows[0]) if rows else None

    async def get_retention_many(
        self,
        scope: StorageScope,
        artifact_ids: Sequence[str],
    ) -> tuple[ArtifactRetentionRecord | None, ...]:
        """Batch-read current artifact retention in one bounded query.

        Exact owner scope filters before hydration, while duplicate and missing input
        identities retain their original result positions.

        Examples:
            Hydrate page retention:
                ```python
                rows = await repository.get_retention_many(scope, artifact_ids)
                ```

            Preserve duplicate slots:
                ```python
                rows = await repository.get_retention_many(scope, ("a", "a"))
                assert rows[0] == rows[1]
                ```

        Args:
            scope: Exact canonical artifact owner scope.
            artifact_ids: At most 500 ordered stable artifact identities.

        Returns:
            tuple[ArtifactRetentionRecord | None, ...]: One current state per input position.

        Notes:
            Empty input returns an empty tuple without querying SQLite; missing state
            means unpinned and remains `None`.
        """
        requested = _batch_artifact_ids(artifact_ids)
        if not requested:
            return ()
        unique = tuple(dict.fromkeys(requested))
        placeholders = ",".join("?" for _item in unique)
        rows = await self._database.fetch_all(
            f"SELECT * FROM local_artifact_retention WHERE owner_scope_identity = ? "
            f"AND artifact_id IN ({placeholders})",
            (_scope_identity(scope), *unique),
        )
        hydrated = {str(row["artifact_id"]): _retention(row) for row in rows}
        return tuple(hydrated.get(artifact_id) for artifact_id in requested)

    async def compare_and_set_retention(
        self,
        record: ArtifactRetentionRecord,
        expected_revision: int,
    ) -> ArtifactRetentionRecord:
        """Atomically create or advance mutable artifact retention intent.

        The exact owner scope and next revision are checked in the same transaction
        that writes pin state.

        Examples:
            Create pinned state:
                ```python
                stored = await repository.compare_and_set_retention(record, 0)
                ```

            Advance an existing record:
                ```python
                stored = await repository.compare_and_set_retention(next_record, current.revision)
                ```

        Args:
            record: Complete next retention revision.
            expected_revision: Exact current revision, or zero for creation.

        Returns:
            ArtifactRetentionRecord: Newly committed authoritative state.

        Notes:
            Missing or foreign artifacts raise `StorageNotFoundError`; stale revisions
            raise `StorageConflictError` without mutation.
        """
        self._require_writable()
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise TypeError("expected_revision must be an integer")
        if expected_revision < 0:
            raise ValueError("expected_revision must be non-negative")
        if record.revision != expected_revision + 1:
            raise StorageConflictError("Artifact retention revision is not the exact next revision")

        def commit(connection: sqlite3.Connection) -> ArtifactRetentionRecord:
            owner_identity = _scope_identity(record.scope)
            artifact = connection.execute(
                """
                SELECT artifact_id FROM local_artifacts
                WHERE artifact_id = ? AND owner_scope_identity = ?
                """,
                (record.artifact_id, owner_identity),
            ).fetchone()
            if artifact is None:
                raise StorageNotFoundError(record.artifact_id)
            current = connection.execute(
                "SELECT * FROM local_artifact_retention WHERE artifact_id = ?",
                (record.artifact_id,),
            ).fetchone()
            current_revision = int(current["revision"]) if current is not None else 0
            if current_revision != expected_revision:
                raise StorageConflictError(
                    f"Artifact retention revision conflict: expected {expected_revision}, "
                    f"found {current_revision}"
                )
            if current is not None and record.updated_at < _retention(current).updated_at:
                raise StorageConflictError("Artifact retention updated_at cannot move backward")
            if current is None:
                connection.execute(
                    """
                    INSERT INTO local_artifact_retention(
                        artifact_id, owner_scope_identity, pinned, revision,
                        updated_at, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.artifact_id,
                        owner_identity,
                        int(record.pinned),
                        record.revision,
                        record.updated_at.isoformat(),
                        record.schema_version,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE local_artifact_retention
                    SET pinned = ?, revision = ?, updated_at = ?, schema_version = ?
                    WHERE artifact_id = ? AND owner_scope_identity = ?
                    """,
                    (
                        int(record.pinned),
                        record.revision,
                        record.updated_at.isoformat(),
                        record.schema_version,
                        record.artifact_id,
                        owner_identity,
                    ),
                )
            return record

        return await self._database.transaction(commit)

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
                f"INSERT INTO local_artifact_occurrences("
                f"occurrence_id, artifact_id, scope_identity, {', '.join(_SCOPE_FIELDS)}, "
                "action, occurred_at, tool_name, tool_version, labels_json, metrics_json, "
                f"schema_version) VALUES ({', '.join('?' for _ in range(20))})",
                (
                    occurrence.occurrence_id,
                    occurrence.artifact_id,
                    _scope_identity(occurrence.scope),
                    *(getattr(occurrence.scope, name) for name in _SCOPE_FIELDS),
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

    async def query_occurrences(
        self,
        query: ArtifactOccurrenceQuery,
    ) -> Page[ArtifactOccurrence]:
        """Query one indexed owner-authorized occurrence page.

        Exact immutable content ownership and every populated execution dimension
        constrain the query before content kind, tag, label, pin, and cursor filters.

        Examples:
            Query one run:
                ```python
                page = await repository.query_occurrences(query)
                ```

            Continue an existing filter set:
                ```python
                page = await repository.query_occurrences(replace(query, page=next_page))
                ```

        Args:
            query: Validated owner authorization, partial scope, filters, and page.

        Returns:
            Page[ArtifactOccurrence]: Stable descending occurrence page and cursor.

        Notes:
            Normalized scope, label, and tag indexes avoid JSON extraction and
            deprecated App metadata is never consulted for authorization.
        """
        clauses = ["a.owner_scope_identity = ?"]
        values: list[object] = [_scope_identity(query.owner_scope)]
        for name, value in query.scope.as_filter().items():
            clauses.append(f"o.{name} = ?")
            values.append(value)
        if query.artifact_id is not None:
            clauses.append("o.artifact_id = ?")
            values.append(query.artifact_id)
        if query.kind is not None:
            clauses.append("a.kind = ?")
            values.append(query.kind)
        for index, tag in enumerate(query.tags):
            alias = f"tag_filter_{index}"
            clauses.append(
                f"EXISTS (SELECT 1 FROM local_artifact_tags {alias} "
                f"WHERE {alias}.artifact_id = o.artifact_id AND {alias}.tag = ?)"
            )
            values.append(tag)
        for index, (key, value) in enumerate(query.labels.items()):
            alias = f"label_filter_{index}"
            clauses.append(
                f"EXISTS (SELECT 1 FROM local_artifact_labels {alias} "
                f"WHERE {alias}.artifact_id = o.artifact_id "
                f"AND {alias}.label_key = ? AND {alias}.value_json = ?)"
            )
            values.extend((key, _json(value)))
        if query.pinned is not None:
            clauses.append("COALESCE(r.pinned, 0) = ?")
            values.append(int(query.pinned))

        fingerprint = _artifact_occurrence_query_fingerprint(query)
        if query.page.cursor is not None:
            clauses.append("o.sequence < ?")
            values.append(_decode_cursor(query.page.cursor, fingerprint))
        values.append(query.page.limit + 1)
        rows = await self._database.fetch_all(
            "SELECT o.* FROM local_artifact_occurrences o "
            "JOIN local_artifacts a ON a.artifact_id = o.artifact_id "
            "LEFT JOIN local_artifact_retention r ON r.artifact_id = o.artifact_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY o.sequence DESC LIMIT ?",
            values,
        )
        selected = rows[: query.page.limit]
        return Page(
            items=tuple(_occurrence(row) for row in selected),
            next_cursor=(
                _encode_cursor(fingerprint, int(selected[-1]["sequence"]))
                if len(rows) > query.page.limit
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


def _batch_artifact_ids(artifact_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(artifact_ids, str | bytes | bytearray):
        raise TypeError("artifact_ids must be a sequence of artifact identities")
    requested = tuple(artifact_ids)
    if len(requested) > _MAX_ARTIFACT_BATCH:
        raise StorageConfigurationError(
            f"artifact_ids must contain at most {_MAX_ARTIFACT_BATCH} identities"
        )
    for artifact_id in requested:
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_ids must contain non-empty strings")
    return requested


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


def _retention(row: sqlite3.Row) -> ArtifactRetentionRecord:
    try:
        pinned = int(row["pinned"])
        if pinned not in (0, 1):
            raise ValueError("pinned")
        return ArtifactRetentionRecord(
            artifact_id=str(row["artifact_id"]),
            scope=_scope(str(row["owner_scope_identity"])),
            pinned=bool(pinned),
            revision=int(row["revision"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local artifact retention row is malformed") from exc


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


def _artifact_tags(labels: Mapping[str, object]) -> tuple[str, ...]:
    value = labels.get("tags")
    if isinstance(value, str):
        tags = (item.strip() for item in value.split(","))
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        tags = (str(item).strip() for item in value)
    else:
        return ()
    return tuple(dict.fromkeys(tag for tag in tags if tag))


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


def _artifact_occurrence_query_fingerprint(query: ArtifactOccurrenceQuery) -> str:
    return _fingerprint(
        "occurrence-query",
        _json(
            {
                "owner_scope": query.owner_scope.as_filter(),
                "scope": query.scope.as_filter(),
                "artifact_id": query.artifact_id,
                "kind": query.kind,
                "tags": query.tags,
                "labels": query.labels,
                "pinned": query.pinned,
            }
        ),
    )


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

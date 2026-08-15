"""Exact-mode local search over one provider-owned SQLite role."""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import Mapping, Sequence
from datetime import datetime
import hashlib
import json
import math
import re
import sqlite3
import struct
from time import monotonic
from typing import TypeVar

from aethergraph.contracts.services.llm import EmbeddingClientProtocol

from ...contracts import (
    SearchDocument,
    SearchMode,
    SearchQuery,
    SearchResult,
    StorageCapabilityError,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
    StorageTimeoutError,
)
from .database import LocalSQLiteDatabase

_SEARCH_COMPONENT_VERSION = 1
_MAX_WRITE_BATCH = 1_000
_SQL_ID_BATCH = 400
_MAX_METADATA_FILTERS = 100
_MAX_VECTOR_DIMENSION = 16_384
_T = TypeVar("_T")
_CREATE_DOCUMENTS = """
CREATE TABLE local_search_documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    corpus TEXT NOT NULL,
    item_id TEXT NOT NULL,
    scope_identity TEXT NOT NULL,
    text TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    vector_blob BLOB,
    vector_dimension INTEGER,
    vector_norm REAL,
    indexed_sequence INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    UNIQUE(corpus, scope_identity, item_id),
    CHECK (
        (vector_blob IS NULL AND vector_dimension IS NULL AND vector_norm IS NULL)
        OR
        (vector_blob IS NOT NULL AND vector_dimension > 0 AND vector_norm >= 0.0)
    )
)
"""
_CREATE_DOCUMENT_SCOPE_INDEX = """
CREATE INDEX ix_local_search_scope_time
ON local_search_documents(corpus, scope_identity, occurred_at DESC, document_id DESC)
"""
_CREATE_DOCUMENT_CURSOR_INDEX = """
CREATE INDEX ix_local_search_document_cursor
ON local_search_documents(corpus, indexed_sequence)
"""
_CREATE_METADATA = """
CREATE TABLE local_search_metadata (
    document_id INTEGER NOT NULL
        REFERENCES local_search_documents(document_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY(document_id, key)
)
"""
_CREATE_METADATA_INDEX = """
CREATE INDEX ix_local_search_metadata_value
ON local_search_metadata(key, value_json, document_id)
"""
_CREATE_PROGRESS = """
CREATE TABLE local_search_progress (
    corpus TEXT PRIMARY KEY,
    indexed_sequence INTEGER NOT NULL CHECK (indexed_sequence > 0)
)
"""
_CREATE_FTS = """
CREATE VIRTUAL TABLE local_search_fts USING fts5(
    corpus UNINDEXED,
    scope_identity UNINDEXED,
    item_id UNINDEXED,
    text,
    tokenize='unicode61'
)
"""


class LocalSearchBackend:
    """Transactional local vector and FTS search with exact mode selection."""

    def __init__(
        self,
        *,
        database: LocalSQLiteDatabase,
        embedder: EmbeddingClientProtocol | None,
        max_candidates: int = 10_000,
    ) -> None:
        if database.role.value != "search":
            raise StorageConfigurationError("Local search requires the search database role")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or not 1_000 <= max_candidates <= 100_000
        ):
            raise ValueError("max_candidates must be between 1000 and 100000")
        self._database = database
        self._mode = database.mode
        self._embedder = embedder
        self._max_candidates = max_candidates
        database.install_component(
            name="search",
            version=_SEARCH_COMPONENT_VERSION,
            statements=(
                _CREATE_DOCUMENTS,
                _CREATE_DOCUMENT_SCOPE_INDEX,
                _CREATE_DOCUMENT_CURSOR_INDEX,
                _CREATE_METADATA,
                _CREATE_METADATA_INDEX,
                _CREATE_PROGRESS,
                _CREATE_FTS,
            ),
        )

    async def upsert(self, document: SearchDocument) -> str:
        """Index one canonical document and return its covering corpus cursor.

        Vector generation completes before one transaction updates content, exact
        metadata filters, FTS text, vector data, and freshness state together.

        Examples:
            Index one memory event:
                ```python
                cursor = await search.upsert(document)
                ```

            Retry an unchanged projection:
                ```python
                assert await search.upsert(document) == cursor
                ```

        Args:
            document: Complete canonical searchable projection.

        Returns:
            str: Opaque corpus-bound indexed cursor covering the document.

        Notes:
            Embedding or transaction failure publishes none of the projection.
        """
        result = await self.upsert_many((document,))
        if result is None:  # pragma: no cover - non-empty tuple is invariant
            raise StorageIntegrityError("Search upsert returned no cursor")
        return result

    async def upsert_many(self, documents: tuple[SearchDocument, ...]) -> str | None:
        """Atomically index one bounded same-corpus document batch.

        Different canonical scopes may share a corpus, but all rows in one atomic
        batch use the same exact corpus and unique scoped item identities.

        Examples:
            Index an ordered batch:
                ```python
                cursor = await search.upsert_many((first, second))
                ```

            Skip an empty batch:
                ```python
                assert await search.upsert_many(()) is None
                ```

        Args:
            documents: Immutable same-corpus batch of at most 1000 projections.

        Returns:
            str | None: Covering corpus cursor, or `None` for an empty batch.

        Notes:
            Cross-corpus batches and duplicate scoped identities fail before writes.
        """
        self._require_writable()
        if not documents:
            return None
        if len(documents) > _MAX_WRITE_BATCH:
            raise StorageConfigurationError(
                f"Search write batch exceeds {_MAX_WRITE_BATCH} documents"
            )
        corpus = documents[0].corpus
        if any(document.corpus != corpus for document in documents):
            raise StorageConfigurationError("Atomic search batches require one exact corpus")
        identities = tuple(
            (_scope_identity(document.scope), document.item_id) for document in documents
        )
        if len(set(identities)) != len(identities):
            raise StorageConfigurationError("Search batch contains duplicate scoped identities")
        vectors = await self._embed_documents(documents)

        def commit(connection: sqlite3.Connection) -> int:
            changes: list[tuple[SearchDocument, tuple[float, ...] | None, sqlite3.Row | None]] = []
            for document, vector in zip(documents, vectors, strict=True):
                existing = connection.execute(
                    """
                    SELECT * FROM local_search_documents
                    WHERE corpus = ? AND scope_identity = ? AND item_id = ?
                    """,
                    (document.corpus, _scope_identity(document.scope), document.item_id),
                ).fetchone()
                if existing is not None and _document(existing) == document:
                    if self._embedder is not None and existing["vector_blob"] is None:
                        raise StorageIntegrityError(
                            "Existing search projection lacks its required semantic vector"
                        )
                else:
                    changes.append((document, vector, existing))

            if not changes:
                current = connection.execute(
                    "SELECT indexed_sequence FROM local_search_progress WHERE corpus = ?",
                    (corpus,),
                ).fetchone()
                if current is None:
                    raise StorageIntegrityError("Indexed search documents lack corpus progress")
                return int(current[0])
            sequence = _advance(connection, corpus)
            for document, vector, existing in changes:
                vector_blob, dimension, norm = _encode_vector(vector)
                if existing is None:
                    cursor = connection.execute(
                        """
                        INSERT INTO local_search_documents(
                            corpus, item_id, scope_identity, text, occurred_at,
                            metadata_json, vector_blob, vector_dimension, vector_norm,
                            indexed_sequence, schema_version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document.corpus,
                            document.item_id,
                            _scope_identity(document.scope),
                            document.text,
                            document.occurred_at.isoformat(),
                            _json(document.metadata),
                            vector_blob,
                            dimension,
                            norm,
                            sequence,
                            document.schema_version,
                        ),
                    )
                    document_id = int(cursor.lastrowid)
                else:
                    document_id = int(existing["document_id"])
                    connection.execute(
                        """
                        UPDATE local_search_documents
                        SET text = ?, occurred_at = ?, metadata_json = ?, vector_blob = ?,
                            vector_dimension = ?, vector_norm = ?, indexed_sequence = ?,
                            schema_version = ?
                        WHERE document_id = ?
                        """,
                        (
                            document.text,
                            document.occurred_at.isoformat(),
                            _json(document.metadata),
                            vector_blob,
                            dimension,
                            norm,
                            sequence,
                            document.schema_version,
                            document_id,
                        ),
                    )
                    connection.execute(
                        "DELETE FROM local_search_metadata WHERE document_id = ?",
                        (document_id,),
                    )
                    connection.execute(
                        "DELETE FROM local_search_fts WHERE rowid = ?",
                        (document_id,),
                    )
                _insert_metadata(connection, document_id, document.metadata)
                connection.execute(
                    """
                    INSERT INTO local_search_fts(rowid, corpus, scope_identity, item_id, text)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        document.corpus,
                        _scope_identity(document.scope),
                        document.item_id,
                        document.text,
                    ),
                )
            return sequence

        sequence = await self._database.transaction(commit)
        return _encode_cursor(corpus, sequence)

    async def delete(
        self,
        scope: StorageScope,
        corpus: str,
        item_ids: tuple[str, ...],
    ) -> str | None:
        """Delete exact scoped projections and return covering freshness.

        Missing identities are idempotent and never delete another scope's projection.

        Examples:
            Delete one indexed event:
                ```python
                cursor = await search.delete(scope, "memory", ("event-1",))
                ```

            Skip an empty request:
                ```python
                assert await search.delete(scope, "memory", ()) is None
                ```

        Args:
            scope: Exact canonical owner scope.
            corpus: Exact named corpus.
            item_ids: Bounded immutable item identities.

        Returns:
            str | None: Covering corpus cursor, or `None` for an empty request.

        Notes:
            Authoritative memory events and artifact records are never deleted here.
        """
        self._require_writable()
        _nonempty("corpus", corpus)
        if not item_ids:
            return None
        if len(item_ids) > _MAX_WRITE_BATCH:
            raise StorageConfigurationError(
                f"Search delete batch exceeds {_MAX_WRITE_BATCH} identities"
            )
        if any(not isinstance(item_id, str) or not item_id.strip() for item_id in item_ids):
            raise StorageConfigurationError("Search item identities must be non-empty strings")
        unique_ids = tuple(dict.fromkeys(item_ids))

        def commit(connection: sqlite3.Connection) -> int:
            document_ids: list[int] = []
            for batch in _batches(unique_ids, _SQL_ID_BATCH):
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT document_id FROM local_search_documents
                    WHERE corpus = ? AND scope_identity = ?
                      AND item_id IN ({placeholders})
                    """,
                    (corpus, _scope_identity(scope), *batch),
                ).fetchall()
                document_ids.extend(int(row[0]) for row in rows)
            if not document_ids:
                current = connection.execute(
                    "SELECT indexed_sequence FROM local_search_progress WHERE corpus = ?",
                    (corpus,),
                ).fetchone()
                return int(current[0]) if current is not None else _advance(connection, corpus)
            sequence = _advance(connection, corpus)
            for batch in _batches(tuple(document_ids), _SQL_ID_BATCH):
                document_placeholders = ",".join("?" for _ in batch)
                connection.execute(
                    f"DELETE FROM local_search_fts WHERE rowid IN ({document_placeholders})",
                    batch,
                )
                connection.execute(
                    "DELETE FROM local_search_documents "
                    f"WHERE document_id IN ({document_placeholders})",
                    batch,
                )
            return sequence

        sequence = await self._database.transaction(commit)
        return _encode_cursor(corpus, sequence)

    async def query(self, query: SearchQuery) -> tuple[SearchResult, ...]:
        """Execute one bounded exact-mode search without fallback.

        Scope, time, and arbitrary exact metadata filters are applied before candidate
        bounds. Every returned row reports the exact requested mode.

        Examples:
            Execute semantic search:
                ```python
                rows = await search.query(semantic_query)
                ```

            Execute recent structural search:
                ```python
                rows = await search.query(structural_query)
                ```

        Args:
            query: Exact corpus, mode, scope, filters, bound, and freshness request.

        Returns:
            tuple[SearchResult, ...]: At most `top_k` stable descending results.

        Notes:
            Missing semantic capability raises `StorageCapabilityError`; lexical and
            hybrid requests are never redirected to another mode.
        """
        self._require_mode(query.mode)
        if query.require_indexed_cursor is not None:
            await self.wait_until_indexed(query.corpus, query.require_indexed_cursor, 0.0)
        if query.mode is SearchMode.STRUCTURAL:
            rows = await self._structural(query)
        elif query.mode is SearchMode.LEXICAL:
            rows = await self._lexical(query, query.top_k)
        elif query.mode is SearchMode.SEMANTIC:
            rows = await self._semantic(query, query.top_k)
        elif query.mode is SearchMode.HYBRID:
            rows = await self._hybrid(query)
        else:  # pragma: no cover - canonical enum is exhaustive
            raise StorageConfigurationError(f"Unsupported exact search mode: {query.mode!r}")
        return tuple(
            SearchResult(
                corpus=query.corpus,
                item_id=item_id,
                score=score,
                mode=query.mode,
                metadata=metadata,
            )
            for item_id, score, metadata in rows[: query.top_k]
        )

    async def indexed_cursor(self, corpus: str) -> str | None:
        """Return the latest committed cursor for one exact corpus.

        The cursor identifies durable vector, FTS, metadata, and document state from
        the same transaction.

        Examples:
            Inspect current freshness:
                ```python
                cursor = await search.indexed_cursor("memory")
                ```

            Detect an empty corpus:
                ```python
                assert await search.indexed_cursor("new") is None
                ```

        Args:
            corpus: Exact named corpus.

        Returns:
            str | None: Opaque corpus-bound cursor or `None` before any mutation.

        Notes:
            Services must not parse or compare this provider-owned value themselves.
        """
        _nonempty("corpus", corpus)
        rows = await self._database.fetch_all(
            "SELECT indexed_sequence FROM local_search_progress WHERE corpus = ?",
            (corpus,),
        )
        return _encode_cursor(corpus, int(rows[0][0])) if rows else None

    async def wait_until_indexed(
        self,
        corpus: str,
        cursor: str,
        timeout_seconds: float,
    ) -> str:
        """Wait a bounded interval for one corpus to cover a required cursor.

        Local indexing commits synchronously, while polling permits a concurrent
        writer using the same provider bundle to satisfy the requirement.

        Examples:
            Wait for a concurrent batch:
                ```python
                covered = await search.wait_until_indexed("memory", cursor, 1.0)
                ```

            Perform a nonblocking check:
                ```python
                covered = await search.wait_until_indexed("memory", cursor, 0.0)
                ```

        Args:
            corpus: Exact named corpus.
            cursor: Opaque cursor previously issued for that corpus.
            timeout_seconds: Finite non-negative maximum wait duration.

        Returns:
            str: Current cursor covering the required sequence.

        Notes:
            Invalid/mismatched cursors fail as configuration; expiry raises
            `StorageTimeoutError` and never changes search mode.
        """
        _nonempty("corpus", corpus)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int | float)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be finite and non-negative")
        required = _decode_cursor(cursor, corpus)
        deadline = monotonic() + float(timeout_seconds)
        while True:
            current = await self.indexed_cursor(corpus)
            if current is not None and _decode_cursor(current, corpus) >= required:
                return current
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise StorageTimeoutError(
                    f"Search corpus {corpus!r} did not cover the required cursor"
                )
            await asyncio.sleep(min(0.05, remaining))

    async def _embed_documents(
        self,
        documents: tuple[SearchDocument, ...],
    ) -> tuple[tuple[float, ...] | None, ...]:
        if self._embedder is None:
            return tuple(None for _ in documents)
        vectors = await self._embedder.embed(tuple(document.text for document in documents))
        return _validate_vectors(vectors, len(documents))

    async def _structural(
        self,
        query: SearchQuery,
    ) -> list[tuple[str, float, Mapping[str, object]]]:
        clauses, values = _query_filters(query, alias="d")
        values.append(query.top_k)
        rows = await self._database.fetch_all(
            "SELECT d.* FROM local_search_documents AS d WHERE "
            + " AND ".join(clauses)
            + " ORDER BY d.occurred_at DESC, d.document_id DESC LIMIT ?",
            values,
        )
        return [
            (
                str(row["item_id"]),
                _occurred_score(row),
                _metadata(row),
            )
            for row in rows
        ]

    async def _semantic(
        self,
        query: SearchQuery,
        limit: int,
    ) -> list[tuple[str, float, Mapping[str, object]]]:
        if self._embedder is None:  # protected by _require_mode
            raise StorageCapabilityError("local.sqlite", ("search_semantic",))
        vectors = await self._embedder.embed((query.query,))
        query_vector = _validate_vectors(vectors, 1)[0]
        if query_vector is None:  # pragma: no cover - embedder exists
            raise StorageIntegrityError("Semantic query produced no vector")
        query_norm = math.hypot(*query_vector)
        clauses, values = _query_filters(query, alias="d")
        values.append(self._max_candidates)
        rows = await self._database.fetch_all(
            "SELECT d.* FROM local_search_documents AS d WHERE "
            + " AND ".join(clauses)
            + " ORDER BY d.occurred_at DESC, d.document_id DESC LIMIT ?",
            values,
        )
        scored: list[tuple[str, float, Mapping[str, object]]] = []
        for row in rows:
            vector = _decode_vector(row)
            if len(vector) != len(query_vector):
                raise StorageIntegrityError(
                    "Persisted search vector dimension differs from the active embedder"
                )
            norm = float(row["vector_norm"])
            denominator = query_norm * norm
            score = (
                math.fsum(
                    (left / query_norm) * (right / norm)
                    for left, right in zip(query_vector, vector, strict=True)
                )
                if denominator > 0
                else 0.0
            )
            if not math.isfinite(score):
                raise StorageIntegrityError("Search similarity score is not finite")
            scored.append((str(row["item_id"]), score, _metadata(row)))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    async def _lexical(
        self,
        query: SearchQuery,
        limit: int,
    ) -> list[tuple[str, float, Mapping[str, object]]]:
        expression = _fts_expression(query.query)
        if not expression:
            return []
        clauses, values = _query_filters(query, alias="d")
        clauses.insert(0, "local_search_fts MATCH ?")
        values.insert(0, expression)
        values.append(limit)
        rows = await self._database.fetch_all(
            """
            SELECT d.*, bm25(local_search_fts) AS lexical_rank
            FROM local_search_fts
            JOIN local_search_documents AS d ON d.document_id = local_search_fts.rowid
            WHERE
            """
            + " AND ".join(clauses)
            + " ORDER BY lexical_rank ASC, d.document_id DESC LIMIT ?",
            values,
        )
        return [(str(row["item_id"]), -float(row["lexical_rank"]), _metadata(row)) for row in rows]

    async def _hybrid(
        self,
        query: SearchQuery,
    ) -> list[tuple[str, float, Mapping[str, object]]]:
        channel_limit = min(self._max_candidates, max(query.top_k * 3, query.top_k))
        semantic = await self._semantic(query, channel_limit)
        lexical = await self._lexical(query, channel_limit)
        semantic_scores = {item_id: score for item_id, score, _metadata_value in semantic}
        lexical_scores = {item_id: score for item_id, score, _metadata_value in lexical}
        normalized_semantic = _normalize(semantic_scores)
        normalized_lexical = _normalize(lexical_scores)
        metadata = {
            item_id: item_metadata for item_id, _score, item_metadata in (*semantic, *lexical)
        }
        combined = [
            (
                item_id,
                0.6 * normalized_semantic.get(item_id, 0.0)
                + 0.4 * normalized_lexical.get(item_id, 0.0),
                metadata[item_id],
            )
            for item_id in semantic_scores.keys() | lexical_scores.keys()
        ]
        combined.sort(key=lambda item: (-item[1], item[0]))
        return combined[: query.top_k]

    def _require_mode(self, mode: SearchMode) -> None:
        if mode is SearchMode.SEMANTIC and self._embedder is None:
            raise StorageCapabilityError("local.sqlite", ("search_semantic",))
        if mode is SearchMode.HYBRID and self._embedder is None:
            raise StorageCapabilityError("local.sqlite", ("search_hybrid",))

    def _require_writable(self) -> None:
        if self._mode is StorageOpenMode.READ_ONLY:
            raise StorageReadOnlyError("Local search backend is read-only")


def _query_filters(query: SearchQuery, *, alias: str) -> tuple[list[str], list[object]]:
    if len(query.metadata) > _MAX_METADATA_FILTERS:
        raise StorageConfigurationError(
            f"Search query exceeds {_MAX_METADATA_FILTERS} exact metadata filters"
        )
    clauses = [f"{alias}.corpus = ?", f"{alias}.scope_identity = ?"]
    values: list[object] = [query.corpus, _scope_identity(query.scope)]
    if query.occurred_at_min is not None:
        clauses.append(f"{alias}.occurred_at >= ?")
        values.append(query.occurred_at_min.isoformat())
    if query.occurred_at_max is not None:
        clauses.append(f"{alias}.occurred_at <= ?")
        values.append(query.occurred_at_max.isoformat())
    for index, (key, value) in enumerate(sorted(query.metadata.items())):
        metadata_alias = f"m{index}"
        clauses.append(
            "EXISTS (SELECT 1 FROM local_search_metadata AS "
            f"{metadata_alias} WHERE {metadata_alias}.document_id = {alias}.document_id "
            f"AND {metadata_alias}.key = ? AND {metadata_alias}.value_json = ?)"
        )
        values.extend((key, _json(value)))
    return clauses, values


def _insert_metadata(
    connection: sqlite3.Connection,
    document_id: int,
    metadata: Mapping[str, object],
) -> None:
    connection.executemany(
        "INSERT INTO local_search_metadata(document_id, key, value_json) VALUES (?, ?, ?)",
        ((document_id, key, _json(value)) for key, value in sorted(metadata.items())),
    )


def _advance(connection: sqlite3.Connection, corpus: str) -> int:
    row = connection.execute(
        "SELECT indexed_sequence FROM local_search_progress WHERE corpus = ?",
        (corpus,),
    ).fetchone()
    sequence = int(row[0]) + 1 if row is not None else 1
    connection.execute(
        """
        INSERT INTO local_search_progress(corpus, indexed_sequence) VALUES (?, ?)
        ON CONFLICT(corpus) DO UPDATE SET indexed_sequence = excluded.indexed_sequence
        """,
        (corpus, sequence),
    )
    return sequence


def _document(row: sqlite3.Row) -> SearchDocument:
    try:
        return SearchDocument(
            corpus=str(row["corpus"]),
            item_id=str(row["item_id"]),
            text=str(row["text"]),
            scope=_scope(str(row["scope_identity"])),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
            metadata=json.loads(row["metadata_json"]),
            schema_version=int(row["schema_version"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local search document is malformed") from exc


def _metadata(row: sqlite3.Row) -> Mapping[str, object]:
    try:
        value = json.loads(row["metadata_json"])
        if not isinstance(value, dict):
            raise TypeError("metadata")
        return value
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted local search metadata is malformed") from exc


def _validate_vectors(
    vectors: Sequence[Sequence[float]],
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        if len(vectors) != expected_count:
            raise StorageIntegrityError("Embedding result count does not match search input")
        normalized: list[tuple[float, ...]] = []
        dimension: int | None = None
        for vector in vectors:
            values = tuple(float(value) for value in vector)
            if not values or any(not math.isfinite(value) for value in values):
                raise StorageIntegrityError("Embedding vectors must be non-empty and finite")
            if len(values) > _MAX_VECTOR_DIMENSION:
                raise StorageIntegrityError(
                    f"Embedding vector exceeds {_MAX_VECTOR_DIMENSION} dimensions"
                )
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                raise StorageIntegrityError("Embedding batch contains inconsistent dimensions")
            normalized.append(values)
        return tuple(normalized)
    except StorageIntegrityError:
        raise
    except (TypeError, ValueError, OverflowError) as exc:
        raise StorageIntegrityError("Embedding result is not a valid vector batch") from exc


def _encode_vector(
    vector: tuple[float, ...] | None,
) -> tuple[bytes | None, int | None, float | None]:
    if vector is None:
        return None, None, None
    return (
        struct.pack(f"<{len(vector)}d", *vector),
        len(vector),
        math.hypot(*vector),
    )


def _decode_vector(row: sqlite3.Row) -> tuple[float, ...]:
    try:
        dimension = int(row["vector_dimension"])
        payload = bytes(row["vector_blob"])
        norm = float(row["vector_norm"])
        if dimension < 1 or len(payload) != dimension * 8 or not math.isfinite(norm) or norm < 0:
            raise ValueError("vector encoding")
        vector = struct.unpack(f"<{dimension}d", payload)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("vector values")
        if not math.isclose(math.hypot(*vector), norm, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("vector norm")
        return vector
    except (TypeError, ValueError, struct.error) as exc:
        raise StorageIntegrityError("Persisted local search vector is malformed") from exc


def _occurred_score(row: sqlite3.Row) -> float:
    try:
        value = datetime.fromisoformat(str(row["occurred_at"])).timestamp()
        if not math.isfinite(value):
            raise ValueError("occurred_at")
        return value
    except (TypeError, ValueError, OverflowError) as exc:
        raise StorageIntegrityError("Persisted local search occurrence time is malformed") from exc


def _scope_identity(scope: StorageScope) -> str:
    return json.dumps(scope.as_filter(), sort_keys=True, separators=(",", ":"))


def _scope(identity: str) -> StorageScope:
    try:
        payload = json.loads(identity)
        if not isinstance(payload, dict):
            raise TypeError("scope")
        return StorageScope(**payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StorageIntegrityError("Persisted search scope is malformed") from exc


def _json(value: object) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _batches(values: tuple[_T, ...], size: int) -> tuple[tuple[_T, ...], ...]:
    return tuple(values[index : index + size] for index in range(0, len(values), size))


def _fts_expression(text: str) -> str:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _normalize(scores: Mapping[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum - minimum < 1e-12:
        return {key: 0.5 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def _encode_cursor(corpus: str, sequence: int) -> str:
    payload = json.dumps(
        {"corpus": hashlib.sha256(corpus.encode()).hexdigest()[:24], "sequence": sequence},
        sort_keys=True,
        separators=(",", ":"),
    )
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str, corpus: str) -> int:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
        if not isinstance(payload, dict) or set(payload) != {"corpus", "sequence"}:
            raise ValueError("cursor payload")
        if payload["corpus"] != hashlib.sha256(corpus.encode()).hexdigest()[:24]:
            raise ValueError("cursor corpus")
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
        raise StorageConfigurationError("Invalid or mismatched search cursor") from exc


def _nonempty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StorageConfigurationError(f"{name} must be a non-empty string")

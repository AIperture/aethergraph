"""Canonical artifact content, metadata, occurrence, retention, and search service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    ArtifactRepository,
    ArtifactRetentionRecord,
    BlobStore,
    Page,
    PageRequest,
    RunRepository,
    SearchBackend,
    SearchDocument,
    SearchMode,
    SearchQuery,
    SearchResult,
    SessionRepository,
    StorageCapacityError,
    StorageNotFoundError,
    StorageScope,
)

_ARTIFACT_CORPUS = "artifact"
_DEFAULT_READ_LIMIT = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ArtifactCommitReceipt:
    """One coherent canonical artifact write result and its projection cursor."""

    record: ArtifactRecord
    occurrence: ArtifactOccurrence
    retention: ArtifactRetentionRecord | None
    indexed_cursor: str


class CanonicalArtifactFacade:
    """Provider-neutral artifact facade bound to explicit owner and execution scopes."""

    def __init__(
        self,
        *,
        blobs: BlobStore,
        artifacts: ArtifactRepository,
        search: SearchBackend,
        runs: RunRepository,
        sessions: SessionRepository,
        owner_scope: StorageScope,
        execution_scope: StorageScope,
        tool_name: str | None = None,
        tool_version: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        artifact_id_factory: Callable[[], str] = lambda: f"artifact-{uuid4().hex}",
        occurrence_id_factory: Callable[[], str] = lambda: f"occurrence-{uuid4().hex}",
    ) -> None:
        """Compose the canonical artifact service from one storage bundle.

        Owner and execution scope are supplied independently so content authorization
        is never inferred from deprecated App/client labels or a physical URI.

        Examples:
            Bind a runtime artifact service:
                ```python
                facade = CanonicalArtifactFacade(
                    blobs=bundle.blobs,
                    artifacts=bundle.artifacts,
                    search=bundle.search,
                    runs=bundle.runs,
                    sessions=bundle.sessions,
                    owner_scope=owner_scope,
                    execution_scope=run_scope,
                )
                ```

            Bind tool provenance explicitly:
                ```python
                facade = CanonicalArtifactFacade(
                    blobs=blobs,
                    artifacts=metadata,
                    search=search,
                    runs=runs,
                    sessions=sessions,
                    owner_scope=owner,
                    execution_scope=execution,
                    tool_name="reporter",
                    tool_version="1.0",
                )
                ```

        Args:
            blobs: Canonical immutable streaming content store.
            artifacts: Canonical metadata, occurrence, lineage, and retention repository.
            search: Canonical exact-mode search projection.
            runs: Canonical run repository for idempotent occurrence counters.
            sessions: Canonical session repository for idempotent occurrence counters.
            owner_scope: Stable canonical artifact-content owner scope.
            execution_scope: Exact canonical occurrence scope.
            tool_name: Optional producing Tool name.
            tool_version: Optional producing Tool version.
            clock: UTC wall-clock source for records and projections.
            artifact_id_factory: Stable artifact identity factory.
            occurrence_id_factory: Stable occurrence identity factory.

        Returns:
            None: The inactive-until-S9 canonical service is ready without I/O.

        Notes:
            Every populated owner dimension must match execution scope. App/client
            identity and physical workspace paths are absent.
        """
        owner_dimensions = owner_scope.as_filter()
        if not owner_dimensions:
            raise ValueError("owner_scope must contain at least one canonical dimension")
        if any(getattr(execution_scope, name) != value for name, value in owner_dimensions.items()):
            raise ValueError("execution_scope must contain every exact owner_scope dimension")
        self._blobs = blobs
        self._artifacts = artifacts
        self._search = search
        self._runs = runs
        self._sessions = sessions
        self.owner_scope = owner_scope
        self.execution_scope = execution_scope
        self.tool_name = tool_name
        self.tool_version = tool_version
        self._clock = clock
        self._artifact_id_factory = artifact_id_factory
        self._occurrence_id_factory = occurrence_id_factory

    async def write(
        self,
        chunks: AsyncIterable[bytes],
        *,
        kind: str,
        media_type: str,
        original_filename: str | None = None,
        content_labels: Mapping[str, Any] | None = None,
        occurrence_labels: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        search_text: str = "",
        pinned: bool = False,
        artifact_id: str | None = None,
        occurrence_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> ArtifactCommitReceipt:
        """Stream, record, project, and count one canonical artifact occurrence.

        Content commits through the blob store before immutable metadata, normalized
        occurrence, optional retention, search projection, and idempotent counters.

        Examples:
            Write generated bytes:
                ```python
                receipt = await facade.write(
                    chunks(),
                    kind="report",
                    media_type="text/plain",
                )
                ```

            Retry with stable identities:
                ```python
                receipt = await facade.write(
                    chunks(),
                    kind="report",
                    media_type="text/plain",
                    artifact_id="artifact-1",
                    occurrence_id="occurrence-1",
                    occurred_at=started_at,
                )
                ```

        Args:
            chunks: Asynchronous immutable byte stream consumed exactly once.
            kind: Exact canonical artifact kind.
            media_type: Exact canonical media type; MIME aliases are not accepted.
            original_filename: Optional descriptive filename without path authority.
            content_labels: Optional immutable content metadata.
            occurrence_labels: Optional execution-occurrence metadata.
            metrics: Optional finite occurrence metrics.
            search_text: Optional content-safe searchable projection text.
            pinned: Whether to create explicit pinned retention intent.
            artifact_id: Optional stable idempotency identity for metadata.
            occurrence_id: Optional stable idempotency identity for occurrence/counters.
            occurred_at: Optional stable UTC occurrence time for exact retries.

        Returns:
            ArtifactCommitReceipt: Canonical records and covering search cursor.

        Notes:
            Any projection or counter failure remains visible and never selects a
            legacy store. Stable caller identities and occurrence time make an exact
            retry idempotent.
        """
        if not isinstance(pinned, bool):
            raise TypeError("pinned must be a boolean")
        now = occurred_at or self._now()
        blob = await self._blobs.put(self.owner_scope, chunks)
        record = ArtifactRecord(
            artifact_id=artifact_id or self._artifact_id_factory(),
            content_hash=blob.content_hash,
            hash_algorithm=blob.hash_algorithm,
            size_bytes=blob.size_bytes,
            media_type=media_type,
            kind=kind,
            blob_locator=blob.blob_locator,
            owner_scope=self.owner_scope,
            created_at=now,
            original_filename=original_filename,
            provider_version=blob.provider_version,
            labels=dict(content_labels or {}),
        )
        record = await self._artifacts.put(record)
        occurrence = ArtifactOccurrence(
            occurrence_id=occurrence_id or self._occurrence_id_factory(),
            artifact_id=record.artifact_id,
            scope=self.execution_scope,
            action=ArtifactAction.PRODUCED,
            occurred_at=now,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            labels=dict(occurrence_labels or {}),
            metrics=dict(metrics or {}),
        )
        occurrence = await self._artifacts.record_occurrence(occurrence)
        retention = await self.pin(record.artifact_id, True) if pinned else None
        indexed_cursor = await self._search.upsert(
            _search_document(record, occurrence, search_text=search_text)
        )
        if self.execution_scope.run_id is not None:
            await self._runs.record_artifact(
                self.execution_scope,
                self.execution_scope.run_id,
                occurrence.occurrence_id,
                occurrence.occurred_at,
            )
        if self.execution_scope.session_id is not None:
            await self._sessions.record_artifact(
                self.execution_scope,
                self.execution_scope.session_id,
                occurrence.occurrence_id,
                occurrence.occurred_at,
            )
        return ArtifactCommitReceipt(
            record=record,
            occurrence=occurrence,
            retention=retention,
            indexed_cursor=indexed_cursor,
        )

    async def save_file(
        self,
        path: str | Path,
        *,
        kind: str,
        media_type: str,
        original_filename: str | None = None,
        content_labels: Mapping[str, Any] | None = None,
        occurrence_labels: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        pinned: bool = False,
        cleanup: bool = False,
        artifact_id: str | None = None,
        occurrence_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> ArtifactCommitReceipt:
        """Stream one existing local file through canonical artifact persistence.

        The path is only an ingestion source and is never persisted as a provider
        locator or authorization field.

        Examples:
            Save a report:
                ```python
                receipt = await facade.save_file(
                    "report.txt",
                    kind="report",
                    media_type="text/plain",
                )
                ```

            Consume a staged file after success:
                ```python
                receipt = await facade.save_file(
                    staged,
                    kind="dataset",
                    media_type="text/csv",
                    cleanup=True,
                )
                ```

        Args:
            path: Existing local source file.
            kind: Exact canonical artifact kind.
            media_type: Exact canonical media type.
            original_filename: Optional descriptive filename; defaults to source name.
            content_labels: Optional immutable content metadata.
            occurrence_labels: Optional execution-occurrence metadata.
            metrics: Optional finite occurrence metrics.
            pinned: Whether to create explicit pinned retention intent.
            cleanup: Delete only the exact source file after complete success.
            artifact_id: Optional stable artifact identity.
            occurrence_id: Optional stable occurrence identity.
            occurred_at: Optional stable UTC occurrence time for exact retries.

        Returns:
            ArtifactCommitReceipt: Canonical records and covering search cursor.

        Notes:
            Cleanup never runs after failure and never removes a directory.
        """
        source = Path(path).resolve(strict=True)
        if not source.is_file():
            raise ValueError("path must identify an existing file")
        filename = original_filename or source.name
        receipt = await self.write(
            _file_chunks(source),
            kind=kind,
            media_type=media_type,
            original_filename=filename,
            content_labels=content_labels,
            occurrence_labels=occurrence_labels,
            metrics=metrics,
            search_text=_searchable_description(kind, filename, content_labels),
            pinned=pinned,
            artifact_id=artifact_id,
            occurrence_id=occurrence_id,
            occurred_at=occurred_at,
        )
        if cleanup:
            await asyncio.to_thread(source.unlink)
        return receipt

    async def save_text(
        self,
        payload: str,
        *,
        kind: str = "text",
        original_filename: str | None = None,
        content_labels: Mapping[str, Any] | None = None,
        occurrence_labels: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        pinned: bool = False,
        artifact_id: str | None = None,
        occurrence_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> ArtifactCommitReceipt:
        """Persist UTF-8 text as one canonical artifact occurrence.

        Text is streamed directly to the provider and supplied to the named artifact
        search projection without creating a caller-visible durable file.

        Examples:
            Save plain text:
                ```python
                receipt = await facade.save_text("hello")
                ```

            Save a named report:
                ```python
                receipt = await facade.save_text(
                    report,
                    kind="report",
                    original_filename="report.txt",
                    pinned=True,
                )
                ```

        Args:
            payload: Text content encoded as UTF-8.
            kind: Exact canonical artifact kind.
            original_filename: Optional descriptive filename.
            content_labels: Optional immutable content metadata.
            occurrence_labels: Optional execution-occurrence metadata.
            metrics: Optional finite occurrence metrics.
            pinned: Whether to create explicit pinned retention intent.
            artifact_id: Optional stable artifact identity.
            occurrence_id: Optional stable occurrence identity.
            occurred_at: Optional stable UTC occurrence time for exact retries.

        Returns:
            ArtifactCommitReceipt: Canonical records and covering search cursor.

        Notes:
            The method accepts only `str`; encoding failures propagate directly.
        """
        if not isinstance(payload, str):
            raise TypeError("payload must be a string")
        return await self.write(
            _single_chunk(payload.encode("utf-8")),
            kind=kind,
            media_type="text/plain; charset=utf-8",
            original_filename=original_filename,
            content_labels=content_labels,
            occurrence_labels=occurrence_labels,
            metrics=metrics,
            search_text=payload,
            pinned=pinned,
            artifact_id=artifact_id,
            occurrence_id=occurrence_id,
            occurred_at=occurred_at,
        )

    async def save_json(
        self,
        payload: Mapping[str, Any],
        *,
        kind: str = "json",
        original_filename: str | None = None,
        content_labels: Mapping[str, Any] | None = None,
        occurrence_labels: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        pinned: bool = False,
        artifact_id: str | None = None,
        occurrence_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> ArtifactCommitReceipt:
        """Persist deterministic JSON object content as a canonical artifact.

        Serialization is UTF-8, key-sorted, compact, and rejects non-finite numbers so
        exact caller identities can be retried with identical content metadata.

        Examples:
            Save a JSON result:
                ```python
                receipt = await facade.save_json({"status": "ok"})
                ```

            Save a named pinned manifest:
                ```python
                receipt = await facade.save_json(
                    manifest,
                    kind="manifest",
                    original_filename="manifest.json",
                    pinned=True,
                )
                ```

        Args:
            payload: JSON-compatible object mapping.
            kind: Exact canonical artifact kind.
            original_filename: Optional descriptive filename.
            content_labels: Optional immutable content metadata.
            occurrence_labels: Optional execution-occurrence metadata.
            metrics: Optional finite occurrence metrics.
            pinned: Whether to create explicit pinned retention intent.
            artifact_id: Optional stable artifact identity.
            occurrence_id: Optional stable occurrence identity.
            occurred_at: Optional stable UTC occurrence time for exact retries.

        Returns:
            ArtifactCommitReceipt: Canonical records and covering search cursor.

        Notes:
            Arrays remain values inside the required top-level mapping.
        """
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        encoded = json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return await self.write(
            _single_chunk(encoded),
            kind=kind,
            media_type="application/json",
            original_filename=original_filename,
            content_labels=content_labels,
            occurrence_labels=occurrence_labels,
            metrics=metrics,
            search_text=encoded.decode("utf-8"),
            pinned=pinned,
            artifact_id=artifact_id,
            occurrence_id=occurrence_id,
            occurred_at=occurred_at,
        )

    async def get(self, artifact_id: str) -> ArtifactRecord | None:
        """Read immutable metadata for one exact owner-scoped artifact.

        The repository lookup never broadens scope or hydrates content implicitly.

        Examples:
            Read metadata:
                ```python
                artifact = await facade.get("artifact-1")
                ```

            Detect absence:
                ```python
                assert await facade.get("missing") is None
                ```

        Args:
            artifact_id: Exact stable artifact identity.

        Returns:
            ArtifactRecord | None: Matching immutable metadata or `None`.

        Notes:
            Retention, occurrence, lineage, and bytes use their focused methods.
        """
        return await self._artifacts.get(self.owner_scope, artifact_id)

    def read(self, artifact_id: str) -> AsyncIterator[bytes]:
        """Stream exact owner-authorized artifact content.

        Metadata authorization is resolved before the provider blob iterator exposes
        any bytes.

        Examples:
            Stream content:
                ```python
                async for chunk in facade.read("artifact-1"):
                    consume(chunk)
                ```

            Collect a small artifact:
                ```python
                data = b"".join([chunk async for chunk in facade.read("artifact-1")])
                ```

        Args:
            artifact_id: Exact stable artifact identity.

        Returns:
            AsyncIterator[bytes]: Bounded provider chunks in content order.

        Notes:
            Missing or unauthorized identities raise `StorageNotFoundError`.
        """
        return self._read(artifact_id)

    async def load_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int = _DEFAULT_READ_LIMIT,
    ) -> bytes:
        """Hydrate one explicitly bounded artifact into memory.

        Metadata size is checked before streaming and accumulated size is checked
        again while reading to fail closed on inconsistent provider content.

        Examples:
            Load a small artifact:
                ```python
                data = await facade.load_bytes("artifact-1")
                ```

            Apply a tighter bound:
                ```python
                data = await facade.load_bytes("artifact-1", max_bytes=1024)
                ```

        Args:
            artifact_id: Exact stable artifact identity.
            max_bytes: Positive maximum bytes permitted in memory.

        Returns:
            bytes: Complete authorized artifact content.

        Notes:
            Larger content raises `StorageCapacityError`; it is not truncated.
        """
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        record = await self.get(artifact_id)
        if record is None:
            raise StorageNotFoundError(artifact_id)
        if record.size_bytes > max_bytes:
            raise StorageCapacityError("Artifact exceeds the explicit hydration bound")
        chunks: list[bytes] = []
        total = 0
        async for chunk in self._blobs.read(self.owner_scope, record.blob_locator):
            total += len(chunk)
            if total > max_bytes:
                raise StorageCapacityError("Artifact exceeds the explicit hydration bound")
            chunks.append(chunk)
        return b"".join(chunks)

    async def pin(self, artifact_id: str, pinned: bool = True) -> ArtifactRetentionRecord:
        """Create or advance exact revisioned artifact retention intent.

        Repeating the already-current boolean state returns the current record without
        rewriting immutable content or advancing its retention revision.

        Examples:
            Pin content:
                ```python
                retention = await facade.pin("artifact-1")
                ```

            Unpin content:
                ```python
                retention = await facade.pin("artifact-1", False)
                ```

        Args:
            artifact_id: Exact stable artifact identity.
            pinned: Desired explicit retention state.

        Returns:
            ArtifactRetentionRecord: Current authoritative retention state.

        Notes:
            Concurrent conflicting changes raise the repository's typed CAS conflict.
        """
        if not isinstance(pinned, bool):
            raise TypeError("pinned must be a boolean")
        if await self.get(artifact_id) is None:
            raise StorageNotFoundError(artifact_id)
        current = await self._artifacts.get_retention(self.owner_scope, artifact_id)
        if current is not None and current.pinned is pinned:
            return current
        expected_revision = current.revision if current is not None else 0
        return await self._artifacts.compare_and_set_retention(
            ArtifactRetentionRecord(
                artifact_id=artifact_id,
                scope=self.owner_scope,
                pinned=pinned,
                revision=expected_revision + 1,
                updated_at=self._now(),
            ),
            expected_revision,
        )

    async def get_retention(self, artifact_id: str) -> ArtifactRetentionRecord | None:
        """Read exact owner-scoped mutable retention intent.

        Retention is queried independently from immutable metadata and content
        hydration.

        Examples:
            Read explicit pin state:
                ```python
                retention = await facade.get_retention("artifact-1")
                ```

            Detect no explicit retention state:
                ```python
                assert await facade.get_retention("artifact-1") is None
                ```

        Args:
            artifact_id: Exact stable artifact identity.

        Returns:
            ArtifactRetentionRecord | None: Current retention state or `None`.

        Notes:
            Absence means unpinned by default; no label fallback is consulted.
        """
        return await self._artifacts.get_retention(self.owner_scope, artifact_id)

    async def add_relation(
        self,
        *,
        relation_id: str,
        source_artifact_id: str,
        target_artifact_id: str,
        kind: ArtifactRelationKind,
        metadata: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> ArtifactRelation:
        """Record one typed directed lineage edge in exact owner scope.

        Both endpoints must already exist under the bound canonical owner before the
        repository commits the normalized relation.

        Examples:
            Record derivation:
                ```python
                relation = await facade.add_relation(
                    relation_id="relation-1",
                    source_artifact_id="source",
                    target_artifact_id="result",
                    kind=ArtifactRelationKind.DERIVED_FROM,
                )
                ```

            Record an explicit reference:
                ```python
                relation = await facade.add_relation(
                    relation_id="relation-2",
                    source_artifact_id="report",
                    target_artifact_id="dataset",
                    kind=ArtifactRelationKind.REFERENCES,
                    metadata={"section": "inputs"},
                )
                ```

        Args:
            relation_id: Stable lineage idempotency identity.
            source_artifact_id: Directed source artifact identity.
            target_artifact_id: Directed target artifact identity.
            kind: Exact canonical relation kind.
            metadata: Optional relation-specific JSON metadata.
            created_at: Optional stable UTC relation time for exact retries.

        Returns:
            ArtifactRelation: Authoritative normalized lineage edge.

        Notes:
            Cross-scope and missing endpoints fail closed; no content is hydrated.
        """
        return await self._artifacts.add_relation(
            ArtifactRelation(
                relation_id=relation_id,
                source_artifact_id=source_artifact_id,
                target_artifact_id=target_artifact_id,
                kind=kind,
                scope=self.owner_scope,
                created_at=created_at or self._now(),
                metadata=dict(metadata or {}),
            )
        )

    async def list_relations(
        self,
        artifact_id: str,
        page: PageRequest | None = None,
    ) -> Page[ArtifactRelation]:
        """List one bounded owner-scoped page of artifact lineage.

        Incoming and outgoing edges are returned without metadata or content
        hydration.

        Examples:
            Read initial lineage:
                ```python
                page = await facade.list_relations("artifact-1")
                ```

            Continue a page:
                ```python
                page = await facade.list_relations("artifact-1", next_page)
                ```

        Args:
            artifact_id: Exact artifact identity whose lineage is requested.
            page: Optional bounded opaque cursor request.

        Returns:
            Page[ArtifactRelation]: Matching lineage and continuation cursor.

        Notes:
            Cursor context remains bound to exact owner and artifact identity.
        """
        return await self._artifacts.list_relations(
            self.owner_scope,
            artifact_id,
            page or PageRequest(),
        )

    async def list_occurrences(
        self,
        page: PageRequest | None = None,
        *,
        artifact_id: str | None = None,
    ) -> Page[ArtifactOccurrence]:
        """List one exact execution-scope cursor page of artifact occurrences.

        Scope and optional artifact identity filter before provider pagination.

        Examples:
            List recent occurrences:
                ```python
                page = await facade.list_occurrences()
                ```

            List one artifact's occurrences:
                ```python
                page = await facade.list_occurrences(artifact_id="artifact-1")
                ```

        Args:
            page: Optional bounded opaque cursor request.
            artifact_id: Optional exact artifact identity filter.

        Returns:
            Page[ArtifactOccurrence]: Matching occurrences and continuation cursor.

        Notes:
            Metadata and bytes are never hydrated by this method.
        """
        return await self._artifacts.list_occurrences(
            self.execution_scope,
            page or PageRequest(),
            artifact_id,
        )

    async def search(
        self,
        *,
        query: str = "",
        mode: SearchMode = SearchMode.STRUCTURAL,
        top_k: int = 10,
        metadata: Mapping[str, Any] | None = None,
        require_indexed_cursor: str | None = None,
    ) -> tuple[SearchResult, ...]:
        """Search the canonical artifact projection in one explicit mode.

        Exact owner scope, metadata, result bound, mode, and optional freshness cursor
        are passed directly to the provider search backend.

        Examples:
            List structurally indexed artifacts:
                ```python
                hits = await facade.search()
                ```

            Search artifact text lexically:
                ```python
                hits = await facade.search(query="report", mode=SearchMode.LEXICAL)
                ```

        Args:
            query: Search text; structural mode may use an empty value.
            mode: Exact required search mode with no fallback.
            top_k: Positive bounded result count.
            metadata: Optional exact canonical metadata filters.
            require_indexed_cursor: Optional covering artifact search cursor.

        Returns:
            tuple[SearchResult, ...]: Stable provider-ranked artifact hits.

        Notes:
            Semantic capability failure propagates without relabeling lexical results.
        """
        return await self._search.query(
            SearchQuery(
                corpus=_ARTIFACT_CORPUS,
                mode=mode,
                scope=self.owner_scope,
                query=query,
                top_k=top_k,
                metadata=dict(metadata or {}),
                require_indexed_cursor=require_indexed_cursor,
            )
        )

    async def _read(self, artifact_id: str) -> AsyncIterator[bytes]:
        record = await self.get(artifact_id)
        if record is None:
            raise StorageNotFoundError(artifact_id)
        async for chunk in self._blobs.read(self.owner_scope, record.blob_locator):
            yield chunk

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("artifact clock must return a timezone-aware UTC datetime")
        return value


async def _single_chunk(payload: bytes) -> AsyncIterator[bytes]:
    yield payload


async def _file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while chunk := await asyncio.to_thread(handle.read, chunk_size):
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)


def _searchable_description(
    kind: str,
    filename: str | None,
    labels: Mapping[str, Any] | None,
) -> str:
    parts = [kind]
    if filename:
        parts.append(filename)
    if labels:
        parts.extend(f"{key}: {value}" for key, value in labels.items())
    return " ".join(parts)


def _search_document(
    record: ArtifactRecord,
    occurrence: ArtifactOccurrence,
    *,
    search_text: str,
) -> SearchDocument:
    metadata: dict[str, Any] = {
        "kind": record.kind,
        "media_type": record.media_type,
        "content_hash": record.content_hash,
        "occurrence_id": occurrence.occurrence_id,
    }
    if record.original_filename is not None:
        metadata["original_filename"] = record.original_filename
    return SearchDocument(
        corpus=_ARTIFACT_CORPUS,
        item_id=record.artifact_id,
        text=search_text
        or _searchable_description(record.kind, record.original_filename, record.labels),
        scope=record.owner_scope,
        occurred_at=occurrence.occurred_at,
        metadata=metadata,
    )

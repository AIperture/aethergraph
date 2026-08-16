"""Canonical artifact content, metadata, occurrence, retention, and search service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO
from uuid import uuid4

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactMetricOrder,
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactOrphanCleanupResult,
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
    StorageIntegrityError,
    StorageNotFoundError,
    StorageScope,
)

from ._directory_archive import extract_directory_archive, write_directory_archive
from .public_projection import project_public_artifact

_ARTIFACT_CORPUS = "artifact"
_DEFAULT_READ_LIMIT = 64 * 1024 * 1024
_DIRECTORY_MEDIA_TYPE = "application/vnd.aethergraph.directory+tar"
_DIRECTORY_FORMAT_LABEL = "_aethergraph_directory_format"
_DIRECTORY_ENTRY_COUNT_LABEL = "_aethergraph_directory_entry_count"
_DIRECTORY_FILE_COUNT_LABEL = "_aethergraph_directory_file_count"
_DIRECTORY_TOTAL_BYTES_LABEL = "_aethergraph_directory_total_bytes"
_DIRECTORY_FORMAT = "tar.v1"
_DEFAULT_DIRECTORY_ENTRIES = 10_000
_DEFAULT_DIRECTORY_BYTES = 1024 * 1024 * 1024
_DEFAULT_DIRECTORY_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_PUBLIC_SEARCH_HYDRATION = 500
_DEPRECATED_IDENTITY_LABELS = frozenset({"app_id", "application_id", "client_id"})


@dataclass(frozen=True, slots=True)
class ArtifactCommitReceipt:
    """One coherent canonical artifact write result and its projection cursor."""

    record: ArtifactRecord
    occurrence: ArtifactOccurrence
    retention: ArtifactRetentionRecord | None
    indexed_cursor: str


@dataclass(frozen=True, slots=True)
class PublicArtifactSearchHit:
    """One ranked canonical search result hydrated to the frozen public Artifact."""

    artifact: Artifact
    score: float
    mode: SearchMode


class CanonicalArtifactWriter:
    """Accumulate one canonical artifact stream in service-owned transient staging."""

    def __init__(self, path: Path, handle: BinaryIO) -> None:
        self._path = path
        self._handle = handle
        self._lock = asyncio.Lock()
        self._closed = False
        self._content_labels: dict[str, Any] = {}
        self._occurrence_labels: dict[str, Any] = {}
        self._metrics: dict[str, float] = {}
        self._receipt: ArtifactCommitReceipt | None = None

    @property
    def receipt(self) -> ArtifactCommitReceipt | None:
        """Return the canonical commit receipt after writer finalization.

        The value remains absent while the context is open and after any failed or
        cancelled production attempt.

        Examples:
            Inspect a committed writer:
                ```python
                async with facade.writer(kind="binary") as writer:
                    await writer.write(b"data")
                assert writer.receipt is not None
                ```

            Inspect an active writer:
                ```python
                async with facade.writer(kind="binary") as writer:
                    assert writer.receipt is None
                ```

        Args:
            None.

        Returns:
            ArtifactCommitReceipt | None: Complete receipt after successful exit, or
            `None` before finalization and after failure.

        Notes:
            The receipt never exposes the transient staging path.
        """
        return self._receipt

    async def write(self, chunk: bytes) -> None:
        """Append one immutable byte chunk to transient artifact staging.

        The chunk is written off the event-loop thread and becomes durable only when
        the surrounding facade writer context exits successfully.

        Examples:
            Write one chunk:
                ```python
                await writer.write(b"first")
                ```

            Write a sequence:
                ```python
                for chunk in chunks:
                    await writer.write(chunk)
                ```

        Args:
            chunk: Exact bytes to append in call order.

        Returns:
            None: The transient staging file has accepted the complete chunk.

        Notes:
            Non-bytes input and writes after context exit fail explicitly.
        """
        if not isinstance(chunk, bytes):
            raise TypeError("chunk must be bytes")
        async with self._lock:
            self._ensure_open()
            await asyncio.to_thread(self._handle.write, chunk)

    def add_labels(self, labels: Mapping[str, Any]) -> None:
        """Merge immutable content labels into the pending artifact record.

        Labels remain separate from execution occurrence labels and are validated by
        the canonical record when the surrounding context commits.

        Examples:
            Add one label:
                ```python
                writer.add_labels({"category": "report"})
                ```

            Extend labels incrementally:
                ```python
                writer.add_labels({"stage": "draft"})
                writer.add_labels({"reviewed": True})
                ```

        Args:
            labels: JSON-compatible immutable content labels to merge.

        Returns:
            None: The pending content-label mapping is updated in place.

        Notes:
            Later values replace earlier values for the same exact key.
        """
        self._ensure_open()
        if not isinstance(labels, Mapping):
            raise TypeError("labels must be a mapping")
        self._content_labels.update(labels)

    def add_occurrence_labels(self, labels: Mapping[str, Any]) -> None:
        """Merge execution labels into the pending artifact occurrence.

        These labels describe this production event and do not modify immutable
        content metadata.

        Examples:
            Add a workflow stage:
                ```python
                writer.add_occurrence_labels({"stage": "final"})
                ```

            Extend occurrence metadata:
                ```python
                writer.add_occurrence_labels({"attempt": 1})
                writer.add_occurrence_labels({"review": "accepted"})
                ```

        Args:
            labels: JSON-compatible occurrence labels to merge.

        Returns:
            None: The pending occurrence-label mapping is updated in place.

        Notes:
            Later values replace earlier values for the same exact key.
        """
        self._ensure_open()
        if not isinstance(labels, Mapping):
            raise TypeError("labels must be a mapping")
        self._occurrence_labels.update(labels)

    def add_metrics(self, metrics: Mapping[str, float]) -> None:
        """Merge finite numeric metrics into the pending artifact occurrence.

        Metrics are recorded on the production occurrence and validated by the
        canonical record before any repository mutation.

        Examples:
            Add one metric:
                ```python
                writer.add_metrics({"rows": 10.0})
                ```

            Replace a pending metric:
                ```python
                writer.add_metrics({"quality": 0.8})
                writer.add_metrics({"quality": 0.9})
                ```

        Args:
            metrics: Finite numeric occurrence metrics to merge.

        Returns:
            None: The pending metric mapping is updated in place.

        Notes:
            Later values replace earlier values for the same exact key.
        """
        self._ensure_open()
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        self._metrics.update(metrics)

    async def _close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            await asyncio.to_thread(self._handle.close)
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("artifact writer is closed")


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

    async def stage_path(self, suffix: str = "") -> str:
        """Allocate one service-owned transient file for artifact production.

        The returned path is outside durable provider identity and exists only as a
        local producer convenience until it is ingested or explicitly removed.

        Examples:
            Stage a text file:
                ```python
                path = await facade.stage_path(".txt")
                ```

            Stage an extension-free file:
                ```python
                path = await facade.stage_path()
                ```

        Args:
            suffix: Optional filename suffix without path separators or drive syntax.

        Returns:
            str: Absolute path to an existing empty transient file.

        Notes:
            Callers own unused staging paths; durable records never contain this path.
        """
        normalized = _staging_suffix(suffix)
        return await asyncio.to_thread(_create_staging_file, normalized)

    async def stage_dir(self, suffix: str = "") -> str:
        """Allocate one service-owned transient directory for artifact production.

        The directory is local staging only and does not establish artifact ownership,
        authorization, or a provider blob locator.

        Examples:
            Stage a directory:
                ```python
                directory = await facade.stage_dir()
                ```

            Add a descriptive suffix:
                ```python
                directory = await facade.stage_dir("_frames")
                ```

        Args:
            suffix: Optional directory suffix without path separators or drive syntax.

        Returns:
            str: Absolute path to an existing empty transient directory.

        Notes:
            Callers own unused staging directories and must remove them explicitly.
        """
        normalized = _staging_suffix(suffix)
        return await asyncio.to_thread(
            tempfile.mkdtemp,
            normalized,
            "aethergraph-artifact-",
        )

    @asynccontextmanager
    async def writer(
        self,
        *,
        kind: str,
        media_type: str = "application/octet-stream",
        planned_ext: str | None = None,
        original_filename: str | None = None,
        content_labels: Mapping[str, Any] | None = None,
        occurrence_labels: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        pinned: bool = False,
        artifact_id: str | None = None,
        occurrence_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> AsyncIterator[CanonicalArtifactWriter]:
        """Stage and commit one incrementally produced canonical artifact.

        Bytes spool to a transient file with bounded memory use. Successful context
        exit commits through the same blob, metadata, occurrence, search, and counter
        path as direct streams; failure removes staging without a fallback write.

        Examples:
            Stream binary output:
                ```python
                async with facade.writer(kind="binary") as writer:
                    await writer.write(b"payload")
                ```

            Attach normalized metadata:
                ```python
                async with facade.writer(
                    kind="report",
                    media_type="text/plain",
                    planned_ext=".txt",
                ) as writer:
                    writer.add_labels({"category": "evidence"})
                    writer.add_occurrence_labels({"stage": "final"})
                    await writer.write(b"report")
                receipt = writer.receipt
                ```

        Args:
            kind: Exact canonical artifact kind.
            media_type: Exact canonical media type.
            planned_ext: Optional transient staging suffix.
            original_filename: Optional descriptive filename without path authority.
            content_labels: Optional initial immutable content metadata.
            occurrence_labels: Optional initial execution-occurrence metadata.
            metrics: Optional initial finite occurrence metrics.
            pinned: Whether to create explicit pinned retention intent.
            artifact_id: Optional stable artifact identity.
            occurrence_id: Optional stable occurrence identity.
            occurred_at: Optional stable UTC occurrence time for exact retries.

        Returns:
            AsyncIterator[CanonicalArtifactWriter]: Writer whose receipt is populated
            only after successful context exit.

        Notes:
            The transient path is never exposed by the writer or persisted. Exact
            retries still require caller-supplied stable identities and time.
        """
        staged = Path(await self.stage_path(planned_ext or ""))
        handle = await asyncio.to_thread(staged.open, "wb")
        stream = CanonicalArtifactWriter(staged, handle)
        try:
            if content_labels:
                stream.add_labels(content_labels)
            if occurrence_labels:
                stream.add_occurrence_labels(occurrence_labels)
            if metrics:
                stream.add_metrics(metrics)
            yield stream
            await stream._close()
            receipt = await self.write(
                _file_chunks(staged),
                kind=kind,
                media_type=media_type,
                original_filename=original_filename,
                content_labels=stream._content_labels,
                occurrence_labels=stream._occurrence_labels,
                metrics=stream._metrics,
                search_text=_searchable_description(
                    kind,
                    original_filename,
                    stream._content_labels,
                ),
                pinned=pinned,
                artifact_id=artifact_id,
                occurrence_id=occurrence_id,
                occurred_at=occurred_at,
            )
            stream._receipt = receipt
        finally:
            await stream._close()
            await asyncio.to_thread(staged.unlink, missing_ok=True)

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
        resolved_artifact_id = artifact_id or self._artifact_id_factory()
        resolved_occurrence_id = occurrence_id or self._occurrence_id_factory()
        resolved_content_labels = _canonical_artifact_labels("content_labels", content_labels)
        resolved_occurrence_labels = _canonical_artifact_labels(
            "occurrence_labels",
            occurrence_labels,
        )
        record = ArtifactRecord(
            artifact_id=resolved_artifact_id,
            content_hash="pending",
            hash_algorithm="pending",
            size_bytes=0,
            media_type=media_type,
            kind=kind,
            blob_locator="pending",
            owner_scope=self.owner_scope,
            created_at=now,
            original_filename=original_filename,
            labels=resolved_content_labels,
        )
        occurrence = ArtifactOccurrence(
            occurrence_id=resolved_occurrence_id,
            artifact_id=resolved_artifact_id,
            scope=self.execution_scope,
            action=ArtifactAction.PRODUCED,
            occurred_at=now,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            labels=resolved_occurrence_labels,
            metrics=dict(metrics or {}),
        )
        _search_document(record, occurrence, search_text=search_text)
        blob = await self._blobs.put(self.owner_scope, chunks)
        record = replace(
            record,
            content_hash=blob.content_hash,
            hash_algorithm=blob.hash_algorithm,
            size_bytes=blob.size_bytes,
            blob_locator=blob.blob_locator,
            provider_version=blob.provider_version,
        )
        record = await self._artifacts.put(record)
        occurrence = await self._artifacts.record_occurrence(occurrence)
        retention = await self.pin(record.artifact_id, True) if pinned else None
        indexed_cursor = await self._search.upsert(
            _search_document(record, occurrence, search_text=search_text)
        )
        if self.execution_scope.run_id is not None:
            await self._runs.record_artifact(
                self.execution_scope,
                self.execution_scope.run_id,
                record.artifact_id,
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

    async def save_directory(
        self,
        path: str | Path,
        *,
        kind: str = "directory",
        original_filename: str | None = None,
        content_labels: Mapping[str, Any] | None = None,
        occurrence_labels: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        pinned: bool = False,
        cleanup: bool = False,
        max_entries: int = _DEFAULT_DIRECTORY_ENTRIES,
        max_total_bytes: int = _DEFAULT_DIRECTORY_BYTES,
        artifact_id: str | None = None,
        occurrence_id: str | None = None,
        occurred_at: datetime | None = None,
    ) -> ArtifactCommitReceipt:
        """Package and persist one directory as deterministic canonical tar content.

        Regular files and empty directories are sorted by POSIX relative path and
        archived with normalized ownership, modes, and timestamps. Links and special
        files fail closed before provider persistence.

        Examples:
            Save a generated directory:
                ```python
                receipt = await facade.save_directory("build-output")
                ```

            Consume explicit staging after success:
                ```python
                receipt = await facade.save_directory(
                    staged_directory,
                    kind="dataset",
                    cleanup=True,
                    pinned=True,
                )
                ```

        Args:
            path: Existing non-linked local source directory.
            kind: Exact canonical artifact kind.
            original_filename: Optional descriptive archive filename.
            content_labels: Optional immutable content metadata.
            occurrence_labels: Optional execution-occurrence metadata.
            metrics: Optional finite occurrence metrics.
            pinned: Whether to create explicit pinned retention intent.
            cleanup: Delete only the exact source directory after complete success.
            max_entries: Positive maximum combined file and directory entries.
            max_total_bytes: Positive maximum total regular-file source bytes.
            artifact_id: Optional stable artifact identity.
            occurrence_id: Optional stable occurrence identity.
            occurred_at: Optional stable UTC occurrence time for exact retries.

        Returns:
            ArtifactCommitReceipt: Canonical directory content and occurrence records.

        Notes:
            Directory bytes use canonical `tar.v1`; no source path, file timestamp,
            platform ownership, or executable bit enters durable identity.
        """
        _positive_bound("max_entries", max_entries)
        _positive_bound("max_total_bytes", max_total_bytes)
        requested_source = Path(path)
        is_junction = getattr(requested_source, "is_junction", lambda: False)
        if requested_source.is_symlink() or is_junction():
            raise ValueError("path must identify an existing non-linked directory")
        source = requested_source.resolve(strict=True)
        if not source.is_dir():
            raise ValueError("path must identify an existing non-linked directory")
        source_identity = source.stat(follow_symlinks=False)
        staged = Path(await self.stage_path(".tar"))
        try:
            stats = await asyncio.to_thread(
                write_directory_archive,
                source,
                staged,
                max_entries=max_entries,
                max_total_bytes=max_total_bytes,
            )
            labels = dict(content_labels or {})
            labels[_DIRECTORY_FORMAT_LABEL] = _DIRECTORY_FORMAT
            labels[_DIRECTORY_ENTRY_COUNT_LABEL] = stats.entry_count
            labels[_DIRECTORY_FILE_COUNT_LABEL] = stats.file_count
            labels[_DIRECTORY_TOTAL_BYTES_LABEL] = stats.total_bytes
            filename = original_filename or f"{source.name}.tar"
            receipt = await self.write(
                _file_chunks(staged),
                kind=kind,
                media_type=_DIRECTORY_MEDIA_TYPE,
                original_filename=filename,
                content_labels=labels,
                occurrence_labels=occurrence_labels,
                metrics=metrics,
                search_text=_searchable_description(kind, filename, labels),
                pinned=pinned,
                artifact_id=artifact_id,
                occurrence_id=occurrence_id,
                occurred_at=occurred_at,
            )
        finally:
            await asyncio.to_thread(staged.unlink, missing_ok=True)
        if cleanup:
            current_identity = source.stat(follow_symlinks=False)
            if not os.path.samestat(source_identity, current_identity):
                raise StorageIntegrityError("Directory source identity changed before cleanup")
            await asyncio.to_thread(shutil.rmtree, source)
        return receipt

    async def materialize_directory(
        self,
        artifact_id: str,
        destination: str | Path,
        *,
        max_entries: int = _DEFAULT_DIRECTORY_ENTRIES,
        max_total_bytes: int = _DEFAULT_DIRECTORY_BYTES,
        max_archive_bytes: int = _DEFAULT_DIRECTORY_ARCHIVE_BYTES,
    ) -> str:
        """Safely materialize one bounded canonical directory artifact.

        Metadata authorization and canonical format are checked before the archive is
        streamed to transient staging. Extraction rejects links, special files,
        absolute/traversal paths, duplicates, and file-as-parent collisions.

        Examples:
            Materialize with defaults:
                ```python
                path = await facade.materialize_directory("artifact-1", "output")
                ```

            Apply tighter resource bounds:
                ```python
                path = await facade.materialize_directory(
                    "artifact-1",
                    "output",
                    max_entries=100,
                    max_total_bytes=10_000_000,
                )
                ```

        Args:
            artifact_id: Exact stable directory artifact identity.
            destination: New local directory path beneath an existing parent.
            max_entries: Positive maximum combined file and directory entries.
            max_total_bytes: Positive maximum total extracted regular-file bytes.
            max_archive_bytes: Positive maximum transient archive bytes.

        Returns:
            str: Absolute path to the completely materialized new directory.

        Notes:
            Existing destinations are never merged or overwritten. Any extraction
            failure removes only the new destination created by this call.
        """
        _positive_bound("max_entries", max_entries)
        _positive_bound("max_total_bytes", max_total_bytes)
        _positive_bound("max_archive_bytes", max_archive_bytes)
        record = await self.get(artifact_id)
        if record is None:
            raise StorageNotFoundError(artifact_id)
        if (
            record.media_type != _DIRECTORY_MEDIA_TYPE
            or record.labels.get(_DIRECTORY_FORMAT_LABEL) != _DIRECTORY_FORMAT
        ):
            raise StorageIntegrityError("Artifact is not a canonical directory archive")
        if record.size_bytes > max_archive_bytes:
            raise StorageCapacityError("Directory archive exceeds the explicit archive bound")

        target = Path(destination).resolve(strict=False)
        staged = Path(await self.stage_path(".tar"))
        try:
            total = await _write_chunks_to_file(
                self._blobs.read(self.owner_scope, record.blob_locator),
                staged,
                max_bytes=max_archive_bytes,
            )
            if total != record.size_bytes:
                raise StorageIntegrityError("Directory archive size does not match metadata")
            await asyncio.to_thread(
                extract_directory_archive,
                staged,
                target,
                max_entries=max_entries,
                max_total_bytes=max_total_bytes,
            )
        finally:
            await asyncio.to_thread(staged.unlink, missing_ok=True)
        return str(target)

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

    def project_commit(
        self,
        receipt: ArtifactCommitReceipt,
        *,
        deprecated_app_id: str | None = None,
    ) -> Artifact:
        """Project one canonical commit receipt into the frozen public Artifact DTO.

        The immutable record, exact production occurrence, and optional retention
        revision are combined only at this public response boundary.

        Examples:
            Project a new artifact for an agent response:
                ```python
                artifact = facade.project_commit(receipt)
                ```

            Retain explicit deprecated App response metadata:
                ```python
                artifact = facade.project_commit(
                    receipt,
                    deprecated_app_id=request.app_id,
                )
                ```

        Args:
            receipt: Complete commit result returned by this canonical service shape.
            deprecated_app_id: Optional deprecated response-only App metadata; never
                inferred and never used for authorization or lookup.

        Returns:
            Artifact: Frozen public DTO with a stable AG content route and no provider
            blob locator.

        Notes:
            Projection performs no provider I/O and never reconstructs `client_id`.
        """
        if not isinstance(receipt, ArtifactCommitReceipt):
            raise TypeError("receipt must be an ArtifactCommitReceipt")
        return project_public_artifact(
            receipt.record,
            occurrence=receipt.occurrence,
            retention=receipt.retention,
            deprecated_app_id=deprecated_app_id,
        )

    async def get_public(
        self,
        artifact_id: str,
        *,
        scope: StorageScope | None = None,
        deprecated_app_id: str | None = None,
    ) -> Artifact | None:
        """Read one occurrence-backed frozen public Artifact in canonical scope.

        The provider filters exact content ownership, the requested partial execution
        scope, and artifact identity before selecting the newest occurrence. Content
        and current retention then hydrate through one bounded public page.

        Examples:
            Read in the facade execution scope:
                ```python
                artifact = await facade.get_public("artifact-1")
                ```

            Read one run-scoped occurrence:
                ```python
                artifact = await facade.get_public(
                    "artifact-1",
                    scope=StorageScope(run_id="run-1"),
                )
                ```

        Args:
            artifact_id: Exact stable artifact identity.
            scope: Optional partial canonical occurrence filter; defaults to the
                facade's exact execution scope.
            deprecated_app_id: Optional deprecated response-only App metadata; never
                inferred and never used for authorization or lookup.

        Returns:
            Artifact | None: Newest authorized occurrence projection or `None` when
            no matching occurrence exists.

        Notes:
            An immutable content row without a matching authorized occurrence is not
            exposed through this occurrence-backed public method.
        """
        page = await self.query_public_artifacts(
            PageRequest(limit=1),
            scope=scope,
            artifact_id=artifact_id,
            deprecated_app_id=deprecated_app_id,
        )
        return page.items[0] if page.items else None

    async def get_many(
        self,
        artifact_ids: Sequence[str],
    ) -> tuple[ArtifactRecord | None, ...]:
        """Batch-read canonical artifact metadata in bound owner scope.

        The provider resolves one bounded ordered request so occurrence-page hydration
        never degrades into caller-managed single-record loops.

        Examples:
            Hydrate artifact identities:
                ```python
                records = await facade.get_many(("artifact-1", "artifact-2"))
                ```

            Preserve a missing slot:
                ```python
                records = await facade.get_many(("artifact-1", "missing"))
                assert records[1] is None
                ```

        Args:
            artifact_ids: Bounded ordered identities; duplicates are preserved.

        Returns:
            tuple[ArtifactRecord | None, ...]: One exact owner-scoped result per slot.

        Notes:
            No content, occurrence, lineage, or retention state is hydrated.
        """
        return await self._artifacts.get_many(self.owner_scope, artifact_ids)

    async def get_occurrences_many(
        self,
        occurrence_ids: Sequence[str],
    ) -> tuple[ArtifactOccurrence | None, ...]:
        """Batch-read search occurrences authorized by this facade's owner.

        One provider call joins each requested occurrence to immutable content under
        the exact facade owner while retaining duplicates and missing result slots.

        Examples:
            Hydrate search occurrence identities:
                ```python
                rows = await facade.get_occurrences_many(("occurrence-1", "occurrence-2"))
                ```

            Detect a stale projection:
                ```python
                assert await facade.get_occurrences_many(("missing",)) == (None,)
                ```

        Args:
            occurrence_ids: Bounded ordered occurrence identities; duplicates are allowed.

        Returns:
            tuple[ArtifactOccurrence | None, ...]: Authorized occurrence or missing slot.

        Notes:
            Deprecated App/client metadata is never used to authorize the batch.
        """
        return await self._artifacts.get_occurrences_many(self.owner_scope, occurrence_ids)

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

    async def get_retention_many(
        self,
        artifact_ids: Sequence[str],
    ) -> tuple[ArtifactRetentionRecord | None, ...]:
        """Batch-read current retention in bound canonical owner scope.

        The provider resolves one bounded ordered request independently from immutable
        content and execution occurrence records.

        Examples:
            Hydrate current pin state:
                ```python
                retention = await facade.get_retention_many(("artifact-1", "artifact-2"))
                ```

            Preserve duplicate slots:
                ```python
                rows = await facade.get_retention_many(("artifact-1", "artifact-1"))
                assert rows[0] == rows[1]
                ```

        Args:
            artifact_ids: Bounded ordered identities; duplicates are preserved.

        Returns:
            tuple[ArtifactRetentionRecord | None, ...]: One current state per input slot.

        Notes:
            Missing state remains `None` and means unpinned by default.
        """
        return await self._artifacts.get_retention_many(self.owner_scope, artifact_ids)

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

    async def query_occurrences(
        self,
        page: PageRequest | None = None,
        *,
        scope: StorageScope | None = None,
        artifact_id: str | None = None,
        kind: str | None = None,
        tags: tuple[str, ...] = (),
        labels: Mapping[str, Any] | None = None,
        pinned: bool | None = None,
        metric: str | None = None,
        metric_order: ArtifactMetricOrder | None = None,
    ) -> Page[ArtifactOccurrence]:
        """Query a bounded owner-authorized artifact occurrence page.

        The facade supplies exact immutable-content ownership while the caller may
        select partial canonical execution dimensions. All content and retention
        filters run in the repository before cursor pagination.

        Examples:
            Query the facade execution scope:
                ```python
                page = await facade.query_occurrences(kind="report")
                ```

            Query one run with indexed filters:
                ```python
                page = await facade.query_occurrences(
                    scope=StorageScope(run_id="run-1"),
                    tags=("final",),
                    pinned=True,
                )
                ```

        Args:
            page: Optional bounded opaque cursor request, capped at 500 records.
            scope: Optional partial canonical occurrence filter; defaults to the
                facade's exact execution scope.
            artifact_id: Optional exact immutable artifact identity.
            kind: Optional exact immutable artifact kind.
            tags: Immutable unique content-tag intersection filter.
            labels: Optional exact immutable content-label filters.
            pinned: Optional current retention-state filter.
            metric: Optional exact occurrence metric key used for indexed ranking.
            metric_order: Required maximum/minimum ranking direction with `metric`.

        Returns:
            Page[ArtifactOccurrence]: Stable matching occurrences and continuation cursor.

        Notes:
            Deprecated App/client metadata and blob locators are absent from the
            query. This method does not hydrate content metadata or bytes.
        """
        return await self._artifacts.query_occurrences(
            ArtifactOccurrenceQuery(
                owner_scope=self.owner_scope,
                scope=scope or self.execution_scope,
                page=page or PageRequest(),
                artifact_id=artifact_id,
                kind=kind,
                tags=tags,
                labels=dict(labels or {}),
                pinned=pinned,
                metric=metric,
                metric_order=metric_order,
            )
        )

    async def query_public_artifacts(
        self,
        page: PageRequest | None = None,
        *,
        scope: StorageScope | None = None,
        artifact_id: str | None = None,
        kind: str | None = None,
        tags: tuple[str, ...] = (),
        labels: Mapping[str, Any] | None = None,
        pinned: bool | None = None,
        metric: str | None = None,
        metric_order: ArtifactMetricOrder | None = None,
        deprecated_app_id: str | None = None,
    ) -> Page[Artifact]:
        """Hydrate one frozen public Artifact page from canonical records.

        One filtered occurrence query is followed by bounded batch metadata and
        retention reads. The one-way public projection combines those records only
        at the response boundary and preserves the repository cursor unchanged.

        Examples:
            Hydrate recent public artifacts:
                ```python
                page = await facade.query_public_artifacts(kind="report")
                ```

            Supply deprecated App response metadata explicitly:
                ```python
                page = await facade.query_public_artifacts(
                    scope=StorageScope(run_id="run-1"),
                    deprecated_app_id=request.app_id,
                )
                ```

        Args:
            page: Optional bounded opaque cursor request, capped at 500 records.
            scope: Optional partial canonical occurrence filter; defaults to the
                facade's exact execution scope.
            artifact_id: Optional exact immutable artifact identity.
            kind: Optional exact immutable artifact kind.
            tags: Immutable unique content-tag intersection filter.
            labels: Optional exact immutable content-label filters.
            pinned: Optional current retention-state filter.
            metric: Optional exact occurrence metric key used for indexed ranking.
            metric_order: Required maximum/minimum ranking direction with `metric`.
            deprecated_app_id: Optional deprecated response-only App metadata; never
                inferred and never used for authorization.

        Returns:
            Page[Artifact]: Frozen public DTOs with the exact provider cursor, or an
            empty page when no authorized occurrences match.

        Notes:
            Hydration performs no single-record loop, exposes no blob locator, and
            never reconstructs deprecated `client_id` metadata.
        """
        requested_page = page or PageRequest()
        if requested_page.limit > _MAX_PUBLIC_SEARCH_HYDRATION:
            raise ValueError(
                "public Artifact page limit must be between 1 and "
                f"{_MAX_PUBLIC_SEARCH_HYDRATION} for hydration"
            )
        occurrences = await self.query_occurrences(
            requested_page,
            scope=scope,
            artifact_id=artifact_id,
            kind=kind,
            tags=tags,
            labels=labels,
            pinned=pinned,
            metric=metric,
            metric_order=metric_order,
        )
        artifact_ids = tuple(item.artifact_id for item in occurrences.items)
        records, retention = await asyncio.gather(
            self.get_many(artifact_ids),
            self.get_retention_many(artifact_ids),
        )
        projected: list[Artifact] = []
        for occurrence, record, retention_record in zip(
            occurrences.items,
            records,
            retention,
            strict=True,
        ):
            if record is None:
                raise StorageIntegrityError(
                    "Artifact occurrence references missing authorized content"
                )
            projected.append(
                project_public_artifact(
                    record,
                    occurrence=occurrence,
                    retention=retention_record,
                    deprecated_app_id=deprecated_app_id,
                )
            )
        return Page(items=tuple(projected), next_cursor=occurrences.next_cursor)

    async def reconcile_orphans(
        self,
        *,
        older_than: datetime,
        limit: int = 100,
    ) -> ArtifactOrphanCleanupResult:
        """Reconcile a bounded page of expired unreferenced artifact blobs.

        The facade fixes cleanup to its exact immutable-content owner scope and
        delegates the atomic reference recheck, durable tombstones, and physical
        deduplication behavior to the coherent provider.

        Examples:
            Reconcile after an explicit grace period:
                ```python
                result = await facade.reconcile_orphans(older_than=cutoff)
                ```

            Drain bounded cleanup pages:
                ```python
                result = await facade.reconcile_orphans(
                    older_than=cutoff,
                    limit=50,
                )
                ```

        Args:
            older_than: Exclusive timezone-aware UTC blob last-touch cutoff.
            limit: Positive maximum eligible scoped blobs examined, at most 500.

        Returns:
            ArtifactOrphanCleanupResult: Bounded cleanup counts, physical bytes
            freed, and whether another maintenance page remains.

        Notes:
            This method never derives scope from deprecated App/client metadata and
            never performs facade-level check-then-delete logic.
        """
        return await self._blobs.reconcile_artifact_orphans(
            self.owner_scope,
            older_than=older_than,
            limit=limit,
        )

    async def search(
        self,
        *,
        query: str = "",
        mode: SearchMode = SearchMode.STRUCTURAL,
        top_k: int = 10,
        tags: tuple[str, ...] = (),
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
            tags: Immutable unique tag-intersection filters applied before ranking.
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
                tags=tags,
                metadata=dict(metadata or {}),
                require_indexed_cursor=require_indexed_cursor,
            )
        )

    async def search_public_artifacts(
        self,
        *,
        query: str,
        mode: SearchMode,
        top_k: int = 10,
        tags: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        require_indexed_cursor: str | None = None,
        deprecated_app_id: str | None = None,
    ) -> tuple[PublicArtifactSearchHit, ...]:
        """Search and batch-hydrate ranked frozen public Artifacts.

        The exact-mode search runs first, then one bounded occurrence read and bounded
        content/retention reads hydrate results in provider rank order. Missing or
        mismatched canonical projection references fail as storage-integrity errors.

        Examples:
            Run lexical artifact search:
                ```python
                hits = await facade.search_public_artifacts(
                    query="migration report",
                    mode=SearchMode.LEXICAL,
                )
                ```

            Require a covering search cursor:
                ```python
                hits = await facade.search_public_artifacts(
                    query="design",
                    mode=SearchMode.SEMANTIC,
                    require_indexed_cursor=receipt.indexed_cursor,
                )
                ```

        Args:
            query: Search text; structural mode may use an empty value.
            mode: Exact required search mode with no fallback.
            top_k: Positive bounded result count.
            tags: Immutable unique tag-intersection filters applied before ranking.
            metadata: Optional exact canonical search-projection filters.
            require_indexed_cursor: Optional covering artifact search cursor.
            deprecated_app_id: Optional deprecated response-only App metadata.

        Returns:
            tuple[PublicArtifactSearchHit, ...]: Ranked hydrated public results.

        Notes:
            Search projection metadata never authorizes content. Owner-scoped
            occurrence hydration does, and deprecated App/client metadata never does.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if not 1 <= top_k <= _MAX_PUBLIC_SEARCH_HYDRATION:
            raise ValueError(
                f"top_k must be between 1 and {_MAX_PUBLIC_SEARCH_HYDRATION} for hydration"
            )
        results = await self.search(
            query=query,
            mode=mode,
            top_k=top_k,
            tags=tags,
            metadata=metadata,
            require_indexed_cursor=require_indexed_cursor,
        )
        occurrence_ids: list[str] = []
        for result in results:
            occurrence_id = result.metadata.get("occurrence_id")
            if not isinstance(occurrence_id, str) or not occurrence_id.strip():
                raise StorageIntegrityError(
                    "Artifact search result lacks a canonical occurrence identity"
                )
            occurrence_ids.append(occurrence_id)

        artifact_ids = tuple(result.item_id for result in results)
        occurrences, records, retention = await asyncio.gather(
            self.get_occurrences_many(tuple(occurrence_ids)),
            self.get_many(artifact_ids),
            self.get_retention_many(artifact_ids),
        )
        hydrated: list[PublicArtifactSearchHit] = []
        for result, occurrence, record, retention_record in zip(
            results,
            occurrences,
            records,
            retention,
            strict=True,
        ):
            if occurrence is None or record is None:
                raise StorageIntegrityError(
                    "Artifact search result references missing authorized records"
                )
            if occurrence.artifact_id != result.item_id:
                raise StorageIntegrityError(
                    "Artifact search result occurrence references different content"
                )
            hydrated.append(
                PublicArtifactSearchHit(
                    artifact=project_public_artifact(
                        record,
                        occurrence=occurrence,
                        retention=retention_record,
                        deprecated_app_id=deprecated_app_id,
                    ),
                    score=result.score,
                    mode=result.mode,
                )
            )
        return tuple(hydrated)

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


async def _write_chunks_to_file(
    chunks: AsyncIterable[bytes],
    path: Path,
    *,
    max_bytes: int,
) -> int:
    handle = await asyncio.to_thread(path.open, "wb")
    total = 0
    try:
        async for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise StorageIntegrityError("Blob provider yielded a non-bytes chunk")
            total += len(chunk)
            if total > max_bytes:
                raise StorageCapacityError("Directory archive exceeds the explicit archive bound")
            await asyncio.to_thread(handle.write, chunk)
    finally:
        await asyncio.to_thread(handle.close)
    return total


def _staging_suffix(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("staging suffix must be a string")
    if len(value) > 128:
        raise ValueError("staging suffix must not exceed 128 characters")
    if "\x00" in value or "/" in value or "\\" in value or ":" in value:
        raise ValueError("staging suffix must not contain path or drive syntax")
    return value


def _create_staging_file(suffix: str) -> str:
    descriptor, path = tempfile.mkstemp(prefix="aethergraph-artifact-", suffix=suffix)
    os.close(descriptor)
    return path


def _positive_bound(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _canonical_artifact_labels(
    name: str,
    labels: Mapping[str, Any] | None,
) -> dict[str, Any]:
    normalized = dict(labels or {})
    forbidden = sorted(_DEPRECATED_IDENTITY_LABELS.intersection(normalized))
    if forbidden:
        raise ValueError(f"{name} contains deprecated identity labels: {forbidden}")
    return normalized


def _searchable_description(
    kind: str,
    filename: str | None,
    labels: Mapping[str, Any] | None,
) -> str:
    parts = [kind]
    if filename:
        parts.append(filename)
    if labels:
        parts.extend(
            f"{key}: {value}"
            for key, value in labels.items()
            if key not in _DEPRECATED_IDENTITY_LABELS
        )
    return " ".join(parts)


def _search_document(
    record: ArtifactRecord,
    occurrence: ArtifactOccurrence,
    *,
    search_text: str,
) -> SearchDocument:
    metadata: dict[str, Any] = {
        key: value for key, value in record.labels.items() if key not in _DEPRECATED_IDENTITY_LABELS
    }
    metadata.update(
        {
            "kind": record.kind,
            "media_type": record.media_type,
            "content_hash": record.content_hash,
            "occurrence_id": occurrence.occurrence_id,
        }
    )
    if record.original_filename is not None:
        metadata["original_filename"] = record.original_filename
    return SearchDocument(
        corpus=_ARTIFACT_CORPUS,
        item_id=record.artifact_id,
        text=search_text
        or _searchable_description(record.kind, record.original_filename, record.labels),
        scope=record.owner_scope,
        occurred_at=occurrence.occurred_at,
        tags=_artifact_tags(record.labels),
        metadata=metadata,
    )


def _artifact_tags(labels: Mapping[str, Any]) -> tuple[str, ...]:
    value = labels.get("tags")
    if isinstance(value, str):
        tags = (item.strip() for item in value.split(","))
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        tags = (str(item).strip() for item in value)
    else:
        return ()
    return tuple(dict.fromkeys(tag for tag in tags if tag))

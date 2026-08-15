from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import inspect
from typing import get_type_hints

import pytest
from storage_conformance.suite import (
    check_artifact_repository_conformance,
    check_blob_store_conformance,
    check_event_store_conformance,
    check_search_backend_conformance,
    check_state_store_conformance,
)

from aethergraph.storage.contracts import (
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRepository,
    ArtifactRetentionRecord,
    BlobHead,
    BlobRange,
    BlobStore,
    BlobWriteResult,
    EventDraft,
    EventQuery,
    EventRecord,
    EventStore,
    FrozenJson,
    Page,
    PageRequest,
    SearchBackend,
    SearchDocument,
    SearchMode,
    SearchQuery,
    SearchResult,
    SortDirection,
    StateHistoryQuery,
    StateRecord,
    StateStore,
    StorageBundle,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageScope,
)


def _scope_key(scope: StorageScope) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(scope.as_filter().items()))


class _EventStore:
    def __init__(self) -> None:
        self.rows: list[EventRecord] = []

    async def append(self, event: EventDraft) -> EventRecord:
        existing = next((row for row in self.rows if row.event_id == event.event_id), None)
        if existing is not None:
            return existing
        row = EventRecord(
            **{name: getattr(event, name) for name in event.__dataclass_fields__},
            cursor=f"cursor-{len(self.rows) + 1}",
        )
        self.rows.append(row)
        return row

    async def append_many(self, events: tuple[EventDraft, ...]) -> tuple[EventRecord, ...]:
        return tuple([await self.append(event) for event in events])

    async def get(self, scope: StorageScope, event_id: str) -> EventRecord | None:
        return next(
            (
                row
                for row in self.rows
                if row.event_id == event_id and _scope_key(row.scope) == _scope_key(scope)
            ),
            None,
        )

    async def query(self, query: EventQuery) -> Page[EventRecord]:
        rows = [row for row in self.rows if _scope_key(row.scope) == _scope_key(query.scope)]
        if query.kinds:
            rows = [row for row in rows if row.kind in query.kinds]
        rows.sort(key=lambda row: int(row.cursor.split("-")[-1]), reverse=True)
        if query.order is SortDirection.ASCENDING:
            rows.reverse()
        start = int(query.page.cursor.split("-")[-1]) if query.page.cursor else 0
        selected = tuple(rows[start : start + query.page.limit])
        next_index = start + len(selected)
        next_cursor = f"page-{next_index}" if next_index < len(rows) else None
        return Page(items=selected, next_cursor=next_cursor)


class _StateStore:
    def __init__(self) -> None:
        self.current: dict[tuple, StateRecord] = {}
        self.revisions: dict[tuple, list[StateRecord]] = {}

    @staticmethod
    def _key(scope: StorageScope, namespace: str, key: str) -> tuple:
        return (_scope_key(scope), namespace, key)

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
    ) -> StateRecord | None:
        return self.current.get(self._key(scope, namespace, key))

    async def get_many(
        self,
        scope: StorageScope,
        namespace: str,
        keys: tuple[str, ...],
    ) -> tuple[StateRecord | None, ...]:
        return tuple([await self.get(scope, namespace, key) for key in keys])

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
        value: FrozenJson,
        metadata: Mapping[str, FrozenJson],
    ) -> StateRecord:
        storage_key = self._key(scope, namespace, key)
        current = self.current.get(storage_key)
        actual = current.revision if current else 0
        if actual != expected_revision:
            raise StorageConflictError(f"expected revision {expected_revision}, found {actual}")
        record = StateRecord(
            namespace=namespace,
            key=key,
            value=value,
            revision=actual + 1,
            scope=scope,
            updated_at=datetime.now(UTC),
            metadata=metadata,
        )
        self.current[storage_key] = record
        self.revisions.setdefault(storage_key, []).append(record)
        return record

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
    ) -> bool:
        storage_key = self._key(scope, namespace, key)
        current = self.current.get(storage_key)
        if current is None:
            if expected_revision == 0:
                return False
            raise StorageConflictError("state is absent")
        if current.revision != expected_revision:
            raise StorageConflictError("state revision changed")
        del self.current[storage_key]
        return True

    async def history(self, query: StateHistoryQuery) -> Page[StateRecord]:
        rows = list(self.revisions.get(self._key(query.scope, query.namespace, query.key), []))
        rows.sort(key=lambda row: row.revision, reverse=True)
        if query.order is SortDirection.ASCENDING:
            rows.reverse()
        start = int(query.page.cursor.split("-")[-1]) if query.page.cursor else 0
        selected = tuple(rows[start : start + query.page.limit])
        next_index = start + len(selected)
        cursor = f"page-{next_index}" if next_index < len(rows) else None
        return Page(items=selected, next_cursor=cursor)


class _BlobStore:
    def __init__(self) -> None:
        self.rows: dict[tuple, tuple[bytes, BlobWriteResult]] = {}

    async def put(
        self,
        scope: StorageScope,
        chunks,
        expected_hash: str | None = None,
        hash_algorithm: str = "sha256",
    ) -> BlobWriteResult:
        content = b"".join([chunk async for chunk in chunks])
        digest = hashlib.new(hash_algorithm, content).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise StorageIntegrityError("content hash mismatch")
        result = BlobWriteResult(
            blob_locator=f"blob:{digest}",
            content_hash=digest,
            hash_algorithm=hash_algorithm,
            size_bytes=len(content),
            provider_version="1",
        )
        self.rows[(_scope_key(scope), result.blob_locator)] = (content, result)
        return result

    async def _content(self, scope: StorageScope, blob_locator: str) -> bytes:
        try:
            return self.rows[(_scope_key(scope), blob_locator)][0]
        except KeyError as exc:
            raise StorageNotFoundError(blob_locator) from exc

    async def read(self, scope: StorageScope, blob_locator: str, byte_range=None):
        content = await self._content(scope, blob_locator)
        if byte_range is not None:
            content = content[byte_range.start : byte_range.end]
        yield content

    async def head(self, scope: StorageScope, blob_locator: str) -> BlobHead | None:
        row = self.rows.get((_scope_key(scope), blob_locator))
        if row is None:
            return None
        result = row[1]
        return BlobHead(
            blob_locator=result.blob_locator,
            size_bytes=result.size_bytes,
            content_hash=result.content_hash,
            hash_algorithm=result.hash_algorithm,
            provider_version=result.provider_version,
        )

    async def delete(
        self,
        scope: StorageScope,
        blob_locator: str,
        provider_version: str | None = None,
    ) -> bool:
        key = (_scope_key(scope), blob_locator)
        row = self.rows.get(key)
        if row is None:
            return False
        if provider_version is not None and row[1].provider_version != provider_version:
            raise StorageConflictError("blob version changed")
        del self.rows[key]
        return True


class _ArtifactRepository:
    def __init__(self, blobs: _BlobStore) -> None:
        self.blobs = blobs
        self.artifacts: dict[tuple, ArtifactRecord] = {}
        self.retention: dict[tuple, ArtifactRetentionRecord] = {}
        self.occurrences: list[ArtifactOccurrence] = []
        self.relations: list[ArtifactRelation] = []

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        blob = await self.blobs.head(record.owner_scope, record.blob_locator)
        if blob is None:
            raise StorageNotFoundError(record.blob_locator)
        if (
            blob.content_hash,
            blob.hash_algorithm,
            blob.size_bytes,
            blob.provider_version,
        ) != (
            record.content_hash,
            record.hash_algorithm,
            record.size_bytes,
            record.provider_version,
        ):
            raise StorageIntegrityError("artifact metadata conflicts with scoped blob")
        key = (_scope_key(record.owner_scope), record.artifact_id)
        existing = self.artifacts.get(key)
        if existing is not None and existing != record:
            raise StorageIntegrityError("artifact metadata conflicts")
        self.artifacts[key] = record
        return record

    async def get(self, scope: StorageScope, artifact_id: str) -> ArtifactRecord | None:
        return self.artifacts.get((_scope_key(scope), artifact_id))

    async def get_many(
        self,
        scope: StorageScope,
        artifact_ids,
    ) -> tuple[ArtifactRecord | None, ...]:
        return tuple([await self.get(scope, artifact_id) for artifact_id in artifact_ids])

    async def get_retention(
        self,
        scope: StorageScope,
        artifact_id: str,
    ) -> ArtifactRetentionRecord | None:
        return self.retention.get((_scope_key(scope), artifact_id))

    async def get_retention_many(
        self,
        scope: StorageScope,
        artifact_ids,
    ) -> tuple[ArtifactRetentionRecord | None, ...]:
        return tuple([await self.get_retention(scope, artifact_id) for artifact_id in artifact_ids])

    async def compare_and_set_retention(
        self,
        record: ArtifactRetentionRecord,
        expected_revision: int,
    ) -> ArtifactRetentionRecord:
        key = (_scope_key(record.scope), record.artifact_id)
        if await self.get(record.scope, record.artifact_id) is None:
            raise StorageNotFoundError(record.artifact_id)
        current = self.retention.get(key)
        current_revision = current.revision if current is not None else 0
        if current_revision != expected_revision or record.revision != expected_revision + 1:
            raise StorageConflictError("artifact retention revision conflict")
        self.retention[key] = record
        return record

    async def record_occurrence(self, occurrence: ArtifactOccurrence) -> ArtifactOccurrence:
        owner = StorageScope(
            tenant_id=occurrence.scope.tenant_id,
            project_id=occurrence.scope.project_id,
        )
        if await self.get(owner, occurrence.artifact_id) is None:
            raise StorageNotFoundError(occurrence.artifact_id)
        existing = next(
            (row for row in self.occurrences if row.occurrence_id == occurrence.occurrence_id),
            None,
        )
        if existing is not None and existing != occurrence:
            raise StorageIntegrityError("occurrence conflicts")
        if existing is None:
            self.occurrences.append(occurrence)
        return occurrence

    async def list_occurrences(
        self,
        scope: StorageScope,
        page: PageRequest,
        artifact_id: str | None = None,
    ) -> Page[ArtifactOccurrence]:
        rows = [
            row
            for row in self.occurrences
            if _scope_key(row.scope) == _scope_key(scope)
            and (artifact_id is None or row.artifact_id == artifact_id)
        ]
        start = int(page.cursor.split("-")[-1]) if page.cursor else 0
        selected = tuple(rows[start : start + page.limit])
        next_index = start + len(selected)
        cursor = f"page-{next_index}" if next_index < len(rows) else None
        return Page(items=selected, next_cursor=cursor)

    async def query_occurrences(
        self,
        query: ArtifactOccurrenceQuery,
    ) -> Page[ArtifactOccurrence]:
        rows: list[ArtifactOccurrence] = []
        owner_key = _scope_key(query.owner_scope)
        for occurrence in reversed(self.occurrences):
            if any(
                getattr(occurrence.scope, name) != value
                for name, value in query.scope.as_filter().items()
            ):
                continue
            if query.artifact_id is not None and occurrence.artifact_id != query.artifact_id:
                continue
            artifact = self.artifacts.get((owner_key, occurrence.artifact_id))
            if artifact is None or (query.kind is not None and artifact.kind != query.kind):
                continue
            tag_value = artifact.labels.get("tags", ())
            tags = (
                tuple(item.strip() for item in tag_value.split(",") if item.strip())
                if isinstance(tag_value, str)
                else tuple(str(item) for item in tag_value)
            )
            if any(tag not in tags for tag in query.tags):
                continue
            if any(artifact.labels.get(key) != value for key, value in query.labels.items()):
                continue
            retention = self.retention.get((owner_key, occurrence.artifact_id))
            pinned = retention.pinned if retention is not None else False
            if query.pinned is not None and pinned is not query.pinned:
                continue
            rows.append(occurrence)
        start = int(query.page.cursor.split("-")[-1]) if query.page.cursor else 0
        selected = tuple(rows[start : start + query.page.limit])
        next_index = start + len(selected)
        cursor = f"page-{next_index}" if next_index < len(rows) else None
        return Page(items=selected, next_cursor=cursor)

    async def add_relation(self, relation: ArtifactRelation) -> ArtifactRelation:
        for artifact_id in (relation.source_artifact_id, relation.target_artifact_id):
            if await self.get(relation.scope, artifact_id) is None:
                raise StorageNotFoundError(artifact_id)
        existing = next(
            (row for row in self.relations if row.relation_id == relation.relation_id),
            None,
        )
        if existing is not None and existing != relation:
            raise StorageIntegrityError("relation conflicts")
        if existing is None:
            self.relations.append(relation)
        return relation

    async def list_relations(
        self,
        scope: StorageScope,
        artifact_id: str,
        page: PageRequest,
    ) -> Page[ArtifactRelation]:
        rows = [
            row
            for row in self.relations
            if _scope_key(row.scope) == _scope_key(scope)
            and artifact_id in (row.source_artifact_id, row.target_artifact_id)
        ]
        return Page(items=tuple(rows[: page.limit]))


class _SearchBackend:
    def __init__(self) -> None:
        self.documents: dict[tuple, SearchDocument] = {}
        self.cursors: dict[str, int] = {}

    def _advance(self, corpus: str) -> str:
        self.cursors[corpus] = self.cursors.get(corpus, 0) + 1
        return f"cursor-{self.cursors[corpus]}"

    async def upsert(self, document: SearchDocument) -> str:
        self.documents[(_scope_key(document.scope), document.corpus, document.item_id)] = document
        return self._advance(document.corpus)

    async def upsert_many(self, documents: tuple[SearchDocument, ...]) -> str | None:
        cursor = None
        for document in documents:
            cursor = await self.upsert(document)
        return cursor

    async def delete(
        self,
        scope: StorageScope,
        corpus: str,
        item_ids: tuple[str, ...],
    ) -> str | None:
        if not item_ids:
            return None
        for item_id in item_ids:
            self.documents.pop((_scope_key(scope), corpus, item_id), None)
        return self._advance(corpus)

    async def query(self, query: SearchQuery) -> tuple[SearchResult, ...]:
        rows = [
            document
            for (scope_key, corpus, _item_id), document in self.documents.items()
            if scope_key == _scope_key(query.scope) and corpus == query.corpus
        ]
        if query.mode is not SearchMode.STRUCTURAL:
            terms = set(query.query.lower().split())
            rows = [row for row in rows if terms & set(row.text.lower().split())]
        rows = rows[: query.top_k]
        return tuple(
            SearchResult(
                corpus=row.corpus,
                item_id=row.item_id,
                score=1.0,
                mode=query.mode,
                metadata=row.metadata,
            )
            for row in rows
        )

    async def indexed_cursor(self, corpus: str) -> str | None:
        current = self.cursors.get(corpus)
        return f"cursor-{current}" if current is not None else None

    async def wait_until_indexed(
        self,
        corpus: str,
        cursor: str,
        timeout_seconds: float,
    ) -> str:
        current = await self.indexed_cursor(corpus)
        if current is None or int(current.split("-")[-1]) < int(cursor.split("-")[-1]):
            raise TimeoutError
        return current


@pytest.mark.asyncio
async def test_fake_event_store_passes_shared_conformance_suite() -> None:
    await check_event_store_conformance(_EventStore())


@pytest.mark.asyncio
async def test_fake_state_store_passes_shared_conformance_suite() -> None:
    await check_state_store_conformance(_StateStore())


@pytest.mark.asyncio
async def test_fake_blob_store_passes_shared_conformance_suite() -> None:
    await check_blob_store_conformance(_BlobStore())


@pytest.mark.asyncio
async def test_fake_artifact_repository_passes_shared_conformance_suite() -> None:
    blobs = _BlobStore()
    await check_artifact_repository_conformance(_ArtifactRepository(blobs), blobs)


@pytest.mark.asyncio
async def test_fake_search_backend_passes_shared_conformance_suite() -> None:
    await check_search_backend_conformance(_SearchBackend())


def test_query_and_blob_range_validation_is_explicit() -> None:
    scope = StorageScope(tenant_id="tenant-1")
    query = EventQuery(
        scope=scope,
        page=PageRequest(limit=25),
        kinds=("memory.event",),
        order=SortDirection.ASCENDING,
    )

    assert replace(query, page=PageRequest(cursor="opaque")).page.cursor == "opaque"
    assert BlobRange(start=0, end=10).end == 10
    with pytest.raises(ValueError, match="greater"):
        BlobRange(start=10, end=10)
    with pytest.raises(ValueError, match="duplicates"):
        EventQuery(scope=scope, tags=("memory", "memory"))


def test_bundle_exposes_only_typed_canonical_high_frequency_store_fields() -> None:
    hints = get_type_hints(StorageBundle)

    assert hints["events"] is EventStore
    assert hints["memory_events"] is EventStore
    assert hints["state"] is StateStore
    assert hints["blobs"] is BlobStore
    assert hints["artifacts"] is ArtifactRepository
    assert hints["search"] is SearchBackend
    assert "ext_services" not in hints
    assert "stores" not in hints


def test_public_protocol_docstrings_follow_required_section_order() -> None:
    protocols = (EventStore, StateStore, BlobStore, ArtifactRepository, SearchBackend)
    required = ("Examples:", "Args:", "Returns:", "Notes:")

    for protocol in protocols:
        for name, member in inspect.getmembers(protocol, inspect.isfunction):
            if name.startswith("_"):
                continue
            docstring = inspect.getdoc(member) or ""
            positions = tuple(docstring.find(section) for section in required)
            assert all(position >= 0 for position in positions), (protocol.__name__, name)
            assert positions == tuple(sorted(positions)), (protocol.__name__, name)

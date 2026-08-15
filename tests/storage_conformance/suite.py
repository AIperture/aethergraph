from __future__ import annotations

from datetime import UTC, datetime
import hashlib

import pytest

from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactMetricOrder,
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    ArtifactRepository,
    ArtifactRetentionRecord,
    BlobRange,
    BlobStore,
    EventDraft,
    EventQuery,
    EventStore,
    PageRequest,
    SearchBackend,
    SearchDocument,
    SearchMode,
    SearchQuery,
    StateHistoryQuery,
    StateStore,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)


def event(event_id: str, scope: StorageScope, *, topic: str = "memory") -> EventDraft:
    return EventDraft(
        event_id=event_id,
        occurred_at=NOW,
        scope=scope,
        kind="memory.event",
        topic=topic,
        tags=("memory",),
        payload={"id": event_id},
    )


async def check_event_store_conformance(store: EventStore) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1", session_id="s-1")
    other_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-2",
        session_id="s-1",
    )

    first = await store.append(event("event-1", scope))
    batch = await store.append_many((event("event-2", scope), event("event-3", scope)))

    assert first.event_id == "event-1"
    assert [row.event_id for row in batch] == ["event-2", "event-3"]
    assert len({first.cursor, *(row.cursor for row in batch)}) == 3
    assert await store.get(scope, "event-1") == first
    assert await store.get(other_scope, "event-1") is None
    assert await store.append_many(()) == ()

    page_one = await store.query(EventQuery(scope=scope, page=PageRequest(limit=2)))
    assert [row.event_id for row in page_one.items] == ["event-3", "event-2"]
    assert page_one.next_cursor is not None
    page_two = await store.query(
        EventQuery(
            scope=scope,
            page=PageRequest(limit=2, cursor=page_one.next_cursor),
        )
    )
    assert [row.event_id for row in page_two.items] == ["event-1"]
    assert page_two.next_cursor is None


async def check_state_store_conformance(store: StateStore) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1", run_id="run-1")
    other_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-2",
        run_id="run-1",
    )

    assert await store.get(scope, "agent", "writer") is None
    created = await store.compare_and_set(
        scope,
        "agent",
        "writer",
        0,
        {"count": 1},
        {"reason": "create"},
    )
    assert created.revision == 1
    assert await store.get(scope, "agent", "writer") == created
    assert await store.get(other_scope, "agent", "writer") is None

    with pytest.raises(StorageConflictError):
        await store.compare_and_set(scope, "agent", "writer", 0, {"count": 2}, {})
    updated = await store.compare_and_set(
        scope,
        "agent",
        "writer",
        1,
        {"count": 2},
        {"reason": "advance"},
    )
    assert updated.revision == 2

    hydrated = await store.get_many(scope, "agent", ("writer", "missing"))
    assert hydrated == (updated, None)
    history = await store.history(
        StateHistoryQuery(
            scope=scope,
            namespace="agent",
            key="writer",
            page=PageRequest(limit=1),
        )
    )
    assert history.items == (updated,)
    assert history.next_cursor is not None

    with pytest.raises(StorageConflictError):
        await store.delete(scope, "agent", "writer", 1)
    assert await store.delete(scope, "agent", "writer", 2) is True
    assert await store.delete(scope, "agent", "writer", 0) is False


async def check_blob_store_conformance(store: BlobStore) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_scope = StorageScope(tenant_id="tenant-1", project_id="project-2")
    content = b"canonical artifact content"
    digest = hashlib.sha256(content).hexdigest()

    async def chunks():
        yield content[:10]
        yield content[10:]

    stored = await store.put(scope, chunks(), expected_hash=digest)
    assert stored.content_hash == digest
    assert stored.size_bytes == len(content)
    assert await store.head(scope, stored.blob_locator) is not None
    assert await store.head(other_scope, stored.blob_locator) is None
    assert b"".join([chunk async for chunk in store.read(scope, stored.blob_locator)]) == content
    assert (
        b"".join(
            [
                chunk
                async for chunk in store.read(
                    scope,
                    stored.blob_locator,
                    BlobRange(start=3, end=12),
                )
            ]
        )
        == content[3:12]
    )

    async def wrong_chunks():
        yield b"wrong"

    with pytest.raises(StorageIntegrityError):
        await store.put(scope, wrong_chunks(), expected_hash=digest)
    assert (
        await store.delete(
            scope,
            stored.blob_locator,
            provider_version=stored.provider_version,
        )
        is True
    )
    assert await store.delete(scope, stored.blob_locator) is False


async def check_artifact_repository_conformance(
    repository: ArtifactRepository,
    blobs: BlobStore,
) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    run_scope = StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        run_id="run-1",
        session_id="session-1",
    )
    other_scope = StorageScope(tenant_id="tenant-1", project_id="project-2")

    async def content(payload: bytes):
        yield payload

    source_blob = await blobs.put(scope, content(b"source-data"))
    target_blob = await blobs.put(scope, content(b"target-artifact-data"))
    source = ArtifactRecord(
        artifact_id="artifact-source",
        content_hash=source_blob.content_hash,
        hash_algorithm=source_blob.hash_algorithm,
        size_bytes=source_blob.size_bytes,
        media_type="text/plain",
        kind="source",
        blob_locator=source_blob.blob_locator,
        owner_scope=scope,
        created_at=NOW,
        provider_version=source_blob.provider_version,
    )
    target = ArtifactRecord(
        artifact_id="artifact-target",
        content_hash=target_blob.content_hash,
        hash_algorithm=target_blob.hash_algorithm,
        size_bytes=target_blob.size_bytes,
        media_type="application/json",
        kind="result",
        blob_locator=target_blob.blob_locator,
        owner_scope=scope,
        created_at=NOW,
        provider_version=target_blob.provider_version,
        labels={"tags": ("final", "report"), "scope_id": "scope-1"},
    )

    assert await repository.put(source) == source
    assert await repository.put(source) == source
    assert await repository.put(target) == target
    assert await repository.get(scope, target.artifact_id) == target
    assert await repository.get(other_scope, target.artifact_id) is None
    assert await repository.get_many(
        scope,
        (target.artifact_id, "missing", target.artifact_id),
    ) == (target, None, target)
    assert await repository.get_many(other_scope, (target.artifact_id,)) == (None,)

    pinned = ArtifactRetentionRecord(
        artifact_id=target.artifact_id,
        scope=scope,
        pinned=True,
        revision=1,
        updated_at=NOW,
    )
    assert await repository.get_retention(scope, target.artifact_id) is None
    assert await repository.compare_and_set_retention(pinned, 0) == pinned
    assert await repository.get_retention(scope, target.artifact_id) == pinned
    assert await repository.get_retention(other_scope, target.artifact_id) is None
    assert await repository.get_retention_many(
        scope,
        (target.artifact_id, "missing", target.artifact_id),
    ) == (pinned, None, pinned)
    assert await repository.get_retention_many(other_scope, (target.artifact_id,)) == (None,)
    with pytest.raises(StorageConflictError):
        await repository.compare_and_set_retention(pinned, 0)
    unpinned = ArtifactRetentionRecord(
        artifact_id=target.artifact_id,
        scope=scope,
        pinned=False,
        revision=2,
        updated_at=NOW,
    )
    assert await repository.compare_and_set_retention(unpinned, 1) == unpinned
    with pytest.raises(StorageNotFoundError):
        await repository.compare_and_set_retention(
            ArtifactRetentionRecord(
                artifact_id="missing",
                scope=scope,
                pinned=True,
                revision=1,
                updated_at=NOW,
            ),
            0,
        )

    occurrences = tuple(
        ArtifactOccurrence(
            occurrence_id=f"occurrence-{index}",
            artifact_id=target.artifact_id,
            scope=run_scope,
            action=ArtifactAction.PRODUCED if index == 0 else ArtifactAction.CONSUMED,
            occurred_at=NOW,
            metrics={"quality": (0.2, 0.9, 0.9)[index]},
        )
        for index in range(3)
    )
    for occurrence in occurrences:
        assert await repository.record_occurrence(occurrence) == occurrence
    assert await repository.get_occurrences_many(
        scope,
        ("occurrence-1", "missing", "occurrence-1"),
    ) == (occurrences[1], None, occurrences[1])
    assert await repository.get_occurrences_many(
        other_scope,
        ("occurrence-1",),
    ) == (None,)
    page_one = await repository.list_occurrences(run_scope, PageRequest(limit=2))
    assert len(page_one.items) == 2
    assert page_one.next_cursor is not None
    page_two = await repository.list_occurrences(
        run_scope,
        PageRequest(limit=2, cursor=page_one.next_cursor),
    )
    assert len(page_two.items) == 1
    filtered = await repository.query_occurrences(
        ArtifactOccurrenceQuery(
            owner_scope=scope,
            scope=StorageScope(run_id="run-1"),
            kind="result",
            tags=("final",),
            labels={"scope_id": "scope-1"},
            pinned=False,
        )
    )
    assert filtered.items == tuple(reversed(occurrences))
    ranked = await repository.query_occurrences(
        ArtifactOccurrenceQuery(
            owner_scope=scope,
            scope=StorageScope(run_id="run-1"),
            page=PageRequest(limit=2),
            metric="quality",
            metric_order=ArtifactMetricOrder.MAXIMUM,
        )
    )
    assert ranked.items == (occurrences[2], occurrences[1])
    assert ranked.next_cursor is not None
    ranked_tail = await repository.query_occurrences(
        ArtifactOccurrenceQuery(
            owner_scope=scope,
            scope=StorageScope(run_id="run-1"),
            page=PageRequest(limit=2, cursor=ranked.next_cursor),
            metric="quality",
            metric_order=ArtifactMetricOrder.MAXIMUM,
        )
    )
    assert ranked_tail.items == (occurrences[0],)
    assert (
        await repository.query_occurrences(
            ArtifactOccurrenceQuery(
                owner_scope=scope,
                scope=StorageScope(session_id="session-1"),
                pinned=True,
            )
        )
    ).items == ()
    assert (
        await repository.query_occurrences(
            ArtifactOccurrenceQuery(
                owner_scope=other_scope,
                scope=StorageScope(run_id="run-1"),
            )
        )
    ).items == ()

    missing = ArtifactOccurrence(
        occurrence_id="occurrence-missing",
        artifact_id="missing",
        scope=run_scope,
        action=ArtifactAction.ATTACHED,
        occurred_at=NOW,
    )
    with pytest.raises(StorageNotFoundError):
        await repository.record_occurrence(missing)

    relation = ArtifactRelation(
        relation_id="relation-1",
        source_artifact_id=source.artifact_id,
        target_artifact_id=target.artifact_id,
        kind=ArtifactRelationKind.DERIVED_FROM,
        scope=scope,
        created_at=NOW,
    )
    assert await repository.add_relation(relation) == relation
    lineage = await repository.list_relations(scope, target.artifact_id, PageRequest())
    assert lineage.items == (relation,)


async def check_search_backend_conformance(search: SearchBackend) -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_scope = StorageScope(tenant_id="tenant-1", project_id="project-2")
    first = SearchDocument(
        corpus="memory",
        item_id="event-1",
        text="canonical storage migration",
        scope=scope,
        occurred_at=NOW,
    )
    second = SearchDocument(
        corpus="memory",
        item_id="event-2",
        text="provider contract conformance",
        scope=scope,
        occurred_at=NOW,
    )
    hidden = SearchDocument(
        corpus="memory",
        item_id="event-hidden",
        text="canonical storage migration",
        scope=other_scope,
        occurred_at=NOW,
    )

    first_cursor = await search.upsert(first)
    batch_cursor = await search.upsert_many((second, hidden))
    assert batch_cursor is not None and batch_cursor != first_cursor
    assert await search.upsert_many(()) is None
    assert await search.indexed_cursor("memory") == batch_cursor
    assert await search.wait_until_indexed("memory", first_cursor, 0.0) == batch_cursor

    structural = await search.query(
        SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=scope, top_k=10)
    )
    assert {row.item_id for row in structural} == {"event-1", "event-2"}
    assert all(row.mode is SearchMode.STRUCTURAL for row in structural)
    for mode in (SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID):
        rows = await search.query(
            SearchQuery(
                corpus="memory",
                mode=mode,
                scope=scope,
                query="canonical migration",
                top_k=10,
            )
        )
        assert rows
        assert all(row.mode is mode for row in rows)

    delete_cursor = await search.delete(scope, "memory", ("event-1",))
    assert delete_cursor is not None
    remaining = await search.query(
        SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=scope, top_k=10)
    )
    assert {row.item_id for row in remaining} == {"event-2"}

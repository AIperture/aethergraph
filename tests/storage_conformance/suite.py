from __future__ import annotations

from dataclasses import replace
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
    ArtifactSearchProjectionIntent,
    BlobRange,
    BlobStore,
    EventDraft,
    EventQuery,
    EventStore,
    LLMCallDraft,
    LLMCallLifecycleStatus,
    ObservationCaptureMode,
    ObservationDraft,
    ObservationLLMSummaryQuery,
    ObservationRepository,
    ObservationStatus,
    ObservationTraceSummaryQuery,
    PageRequest,
    SearchBackend,
    SearchDocument,
    SearchMode,
    SearchProjectionStatus,
    SearchQuery,
    StateHistoryQuery,
    StateStore,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)


async def seed_observation_summary_conformance(
    observations: ObservationRepository,
) -> None:
    scope = StorageScope(project_id="project-1", run_id="summary-run-1")
    await observations.append_many(
        (
            ObservationDraft(
                observation_id="conformance-trace-a",
                category="trace",
                name="span-a",
                summary="failed span",
                occurred_at=NOW,
                scope=scope,
                trace_id="trace-a",
                producer="runner",
                status=ObservationStatus.ERROR,
                attributes={"duration_ms": 7, "error": {"type": "RuntimeError"}},
            ),
            ObservationDraft(
                observation_id="conformance-trace-b",
                category="trace",
                name="span-b",
                summary="successful span",
                occurred_at=NOW,
                scope=scope,
                trace_id="trace-b",
                attributes={"duration_ms": 5},
            ),
        )
    )
    for call_id, model, usage, error_type in (
        ("conformance-a", "model-a", {"input_tokens": 2, "output_tokens": 1}, None),
        (
            "conformance-b",
            "model-b",
            {"prompt_tokens": 3, "completion_tokens": 2},
            "ProviderError",
        ),
    ):
        completed = LLMCallDraft(
            llm_call_id=call_id,
            observation=ObservationDraft(
                observation_id=f"conformance-llm-{call_id}",
                category="llm",
                name="chat",
                summary="LLM call",
                occurred_at=NOW,
                scope=scope,
                trace_id=f"llm-{call_id}",
                status=(
                    ObservationStatus.ERROR if error_type is not None else ObservationStatus.OK
                ),
            ),
            call_type="chat",
            provider="external-test",
            model=model,
            capture_mode=ObservationCaptureMode.OFF,
            lifecycle_status=(
                LLMCallLifecycleStatus.FAILED
                if error_type is not None
                else LLMCallLifecycleStatus.COMPLETED
            ),
            usage=usage,
            error_type=error_type,
            error_message="failed" if error_type is not None else None,
        )
        started = replace(
            completed,
            observation=replace(
                completed.observation,
                status=ObservationStatus.PENDING,
            ),
            lifecycle_status=LLMCallLifecycleStatus.IN_PROGRESS,
            usage={},
            error_type=None,
            error_message=None,
        )
        await observations.begin_llm_call(started)
        await observations.finish_llm_call(call_id, completed)


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
    owner_scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    assert await store.get(owner_scope, "event-1") == first
    assert await store.get_many(owner_scope, ("event-3", "missing", "event-1")) == (
        batch[1],
        first,
    )
    assert await store.get_many(other_scope, ("event-1",)) == ()
    assert await store.get_many(owner_scope, ()) == ()
    assert await store.append_many(()) == ()

    owner_page = await store.query(EventQuery(scope=owner_scope, page=PageRequest(limit=10)))
    assert [row.event_id for row in owner_page.items] == ["event-3", "event-2", "event-1"]

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


async def check_observation_summary_conformance(
    observations: ObservationRepository,
) -> None:
    scope = StorageScope(project_id="project-1", run_id="summary-run-1")
    trace = await observations.summarize_traces(
        ObservationTraceSummaryQuery(
            scope=scope,
            occurred_at_or_after=NOW,
            occurred_at_or_before=NOW,
            trace_id_limit=1,
            failing_service_limit=1,
        )
    )
    llm = await observations.summarize_llm_calls(
        ObservationLLMSummaryQuery(
            scope=scope,
            occurred_at_or_after=NOW,
            occurred_at_or_before=NOW,
            model_limit=1,
        )
    )

    assert trace.span_count == 2
    assert trace.error_count == 1
    assert trace.total_duration_ms == 12
    assert trace.trace_id_count == 2
    assert trace.trace_ids == ("trace-a",)
    assert trace.trace_ids_truncated
    assert dict(trace.top_failing_services) == {"runner": 1}
    assert trace.latest_error_at == NOW
    assert llm.total_calls == 2
    assert llm.total_prompt_tokens == 5
    assert llm.total_completion_tokens == 3
    assert llm.total_tokens == 8
    assert llm.error_count == 1
    assert llm.model_count == 2
    assert dict(llm.by_model) == {"model-a": 1}
    assert llm.by_model_truncated


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

    committed_blob = await blobs.put(scope, content(b"atomic-production"))
    committed = ArtifactRecord(
        artifact_id="artifact-committed",
        content_hash=committed_blob.content_hash,
        hash_algorithm=committed_blob.hash_algorithm,
        size_bytes=committed_blob.size_bytes,
        media_type="text/plain",
        kind="atomic",
        blob_locator=committed_blob.blob_locator,
        owner_scope=scope,
        created_at=NOW,
        provider_version=committed_blob.provider_version,
    )
    committed_occurrence = ArtifactOccurrence(
        occurrence_id="occurrence-committed",
        artifact_id=committed.artifact_id,
        scope=scope,
        action=ArtifactAction.PRODUCED,
        occurred_at=NOW,
    )
    committed_retention = ArtifactRetentionRecord(
        artifact_id=committed.artifact_id,
        scope=scope,
        pinned=True,
        revision=1,
        updated_at=NOW,
    )
    committed_document = SearchDocument(
        corpus="artifact",
        item_id=committed.artifact_id,
        text="atomic production",
        scope=scope,
        occurred_at=NOW,
        metadata={"occurrence_id": committed_occurrence.occurrence_id},
    )
    committed_intent = ArtifactSearchProjectionIntent(
        intent_id="artifact-search:occurrence-committed",
        artifact_id=committed.artifact_id,
        occurrence_id=committed_occurrence.occurrence_id,
        owner_scope=scope,
        document=committed_document,
        status=SearchProjectionStatus.PENDING,
        revision=1,
        attempts=0,
        updated_at=NOW,
    )
    first_commit = await repository.commit_production(
        committed,
        committed_occurrence,
        committed_retention,
        committed_intent,
    )
    retry_commit = await repository.commit_production(
        committed,
        committed_occurrence,
        committed_retention,
        committed_intent,
    )
    assert first_commit[:-1] == retry_commit[:-1]
    assert first_commit[-1] is True and retry_commit[-1] is False
    assert await repository.get_search_intent(scope, committed_intent.intent_id) == committed_intent
    failed_intent = replace(
        committed_intent,
        status=SearchProjectionStatus.FAILED,
        revision=2,
        attempts=1,
        diagnostic="RuntimeError: search projection failed",
    )
    assert await repository.compare_and_set_search_intent(failed_intent, 1) == failed_intent

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
        tags=("canonical", "shared"),
    )
    second = SearchDocument(
        corpus="memory",
        item_id="event-2",
        text="provider contract conformance",
        scope=scope,
        occurred_at=NOW,
        tags=("contract", "shared"),
    )
    hidden = SearchDocument(
        corpus="memory",
        item_id="event-hidden",
        text="canonical storage migration",
        scope=other_scope,
        occurred_at=NOW,
        tags=("canonical", "shared"),
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
    tagged = await search.query(
        SearchQuery(
            corpus="memory",
            mode=SearchMode.STRUCTURAL,
            scope=scope,
            tags=("shared", "canonical"),
        )
    )
    assert [row.item_id for row in tagged] == ["event-1"]
    for mode in (SearchMode.SEMANTIC, SearchMode.LEXICAL, SearchMode.HYBRID):
        rows = await search.query(
            SearchQuery(
                corpus="memory",
                mode=mode,
                scope=scope,
                query="canonical migration",
                top_k=10,
                tags=("shared", "canonical"),
            )
        )
        assert rows
        assert [row.item_id for row in rows] == ["event-1"]
        assert all(row.mode is mode for row in rows)

    delete_cursor = await search.delete(scope, "memory", ("event-1",))
    assert delete_cursor is not None
    remaining = await search.query(
        SearchQuery(corpus="memory", mode=SearchMode.STRUCTURAL, scope=scope, top_k=10)
    )
    assert {row.item_id for row in remaining} == {"event-2"}

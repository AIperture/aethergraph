from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta

import pytest

from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactMetricOrder,
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactOrphanCleanupResult,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    ArtifactRetentionRecord,
    EventDraft,
    EventRecord,
    Page,
    PageRequest,
    SearchDocument,
    SearchMode,
    SearchQuery,
    SearchResult,
    StateRecord,
    StorageScope,
)

NOW = datetime(2026, 8, 14, 12, tzinfo=UTC)
SCOPE = StorageScope(
    tenant_id="tenant-1",
    project_id="project-1",
    session_id="session-1",
    run_id="run-1",
    graph_id="graph-1",
    node_id="node-1",
    agent_id="agent-1",
)


def test_event_record_is_immutable_utc_scoped_and_has_no_legacy_aliases() -> None:
    payload = {"nested": {"values": [1, 2]}}
    record = EventRecord(
        event_id="event-1",
        cursor="cursor-1",
        occurred_at=NOW,
        scope=SCOPE,
        kind="memory",
        stage="planning",
        topic="tool.call",
        tags=("memory", "tool"),
        payload=payload,
        metrics={"tokens": 12},
    )
    payload["nested"]["values"].append(3)

    assert record.payload["nested"]["values"] == (1, 2)
    assert record.metrics["tokens"] == 12.0
    assert "tool" not in {item.name for item in fields(EventDraft)}
    assert "embedding" not in {item.name for item in fields(EventDraft)}
    assert "app_id" not in {item.name for item in fields(EventDraft)}
    with pytest.raises(FrozenInstanceError):
        record.cursor = "other"  # type: ignore[misc]


def test_event_and_state_records_reject_naive_time_invalid_json_and_revision() -> None:
    with pytest.raises(ValueError, match="UTC"):
        EventDraft(
            event_id="event-1",
            occurred_at=datetime(2026, 8, 14),
            scope=SCOPE,
            kind="memory",
        )
    with pytest.raises(TypeError, match="JSON-compatible"):
        EventDraft(
            event_id="event-1",
            occurred_at=NOW,
            scope=SCOPE,
            kind="memory",
            payload={"invalid": object()},
        )
    with pytest.raises(ValueError, match="positive"):
        StateRecord(
            namespace="agent",
            key="writer",
            value={},
            revision=0,
            scope=SCOPE,
            updated_at=NOW,
        )


def test_state_record_deep_freezes_value_and_metadata() -> None:
    value = {"items": [{"id": "a"}]}
    record = StateRecord(
        namespace="agent",
        key="writer",
        value=value,
        revision=2,
        scope=SCOPE,
        updated_at=NOW,
        metadata={"reason": "checkpoint"},
    )
    value["items"].append({"id": "b"})

    assert record.value["items"] == ({"id": "a"},)
    assert record.metadata["reason"] == "checkpoint"


def test_artifact_records_separate_content_occurrence_and_lineage() -> None:
    artifact = ArtifactRecord(
        artifact_id="artifact-1",
        content_hash="abc123",
        hash_algorithm="sha256",
        size_bytes=42,
        media_type="text/plain",
        kind="report",
        blob_locator="blob:abc123",
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        created_at=NOW,
        original_filename="report.txt",
    )
    occurrence = ArtifactOccurrence(
        occurrence_id="occurrence-1",
        artifact_id=artifact.artifact_id,
        scope=SCOPE,
        action=ArtifactAction.PRODUCED,
        occurred_at=NOW,
        tool_name="reporter",
        metrics={"quality": 0.9},
    )
    relation = ArtifactRelation(
        relation_id="relation-1",
        source_artifact_id="artifact-source",
        target_artifact_id=artifact.artifact_id,
        kind=ArtifactRelationKind.DERIVED_FROM,
        scope=SCOPE,
        created_at=NOW,
    )
    retention = ArtifactRetentionRecord(
        artifact_id=artifact.artifact_id,
        scope=artifact.owner_scope,
        pinned=True,
        revision=1,
        updated_at=NOW,
    )

    artifact_fields = {item.name for item in fields(ArtifactRecord)}
    occurrence_fields = {item.name for item in fields(ArtifactOccurrence)}
    retention_fields = {item.name for item in fields(ArtifactRetentionRecord)}
    assert {"media_type", "size_bytes", "blob_locator"} <= artifact_fields
    assert not {"mime", "mimetype", "bytes", "uri", "app_id"} & artifact_fields
    assert not {"content_hash", "size_bytes", "media_type", "blob_locator"} & occurrence_fields
    assert retention_fields == {
        "artifact_id",
        "scope",
        "pinned",
        "revision",
        "updated_at",
        "schema_version",
    }
    assert occurrence.metrics["quality"] == 0.9
    assert retention.pinned is True
    assert relation.target_artifact_id == artifact.artifact_id


def test_artifact_retention_requires_boolean_pin_positive_revision_and_utc_time() -> None:
    values = {
        "artifact_id": "artifact-1",
        "scope": StorageScope(project_id="project-1"),
        "pinned": True,
        "revision": 1,
        "updated_at": NOW,
    }
    with pytest.raises(TypeError, match="boolean"):
        ArtifactRetentionRecord(**{**values, "pinned": 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        ArtifactRetentionRecord(**{**values, "revision": 0})
    with pytest.raises(ValueError, match="UTC"):
        ArtifactRetentionRecord(**{**values, "updated_at": datetime(2026, 8, 15)})


def test_artifact_orphan_cleanup_result_is_bounded_and_strictly_typed() -> None:
    result = ArtifactOrphanCleanupResult(
        examined=3,
        deleted_scoped_blobs=2,
        deleted_physical_blobs=1,
        freed_bytes=128,
        has_more=True,
    )

    assert result.deleted_scoped_blobs == 2
    with pytest.raises(ValueError, match="non-negative integer"):
        ArtifactOrphanCleanupResult(
            examined=-1,
            deleted_scoped_blobs=0,
            deleted_physical_blobs=0,
            freed_bytes=0,
            has_more=False,
        )
    with pytest.raises(TypeError, match="boolean"):
        ArtifactOrphanCleanupResult(
            examined=0,
            deleted_scoped_blobs=0,
            deleted_physical_blobs=0,
            freed_bytes=0,
            has_more=1,  # type: ignore[arg-type]
        )


def test_artifact_lineage_rejects_self_edges() -> None:
    with pytest.raises(ValueError, match="self-edge"):
        ArtifactRelation(
            relation_id="relation-1",
            source_artifact_id="artifact-1",
            target_artifact_id="artifact-1",
            kind=ArtifactRelationKind.REFERENCES,
            scope=SCOPE,
            created_at=NOW,
        )


def test_artifact_occurrence_metric_ranking_requires_an_explicit_pair() -> None:
    query = ArtifactOccurrenceQuery(
        owner_scope=StorageScope(project_id="project-1"),
        scope=StorageScope(run_id="run-1"),
        metric="quality",
        metric_order=ArtifactMetricOrder.MAXIMUM,
    )

    assert query.metric_order is ArtifactMetricOrder.MAXIMUM
    with pytest.raises(ValueError, match="supplied together"):
        ArtifactOccurrenceQuery(
            owner_scope=StorageScope(project_id="project-1"),
            scope=StorageScope(run_id="run-1"),
            metric="quality",
        )


def test_search_query_requires_explicit_supported_mode_shape() -> None:
    structural = SearchQuery(
        corpus="memory",
        mode=SearchMode.STRUCTURAL,
        scope=SCOPE,
        top_k=25,
    )
    semantic = SearchQuery(
        corpus="memory",
        mode=SearchMode.SEMANTIC,
        scope=SCOPE,
        query="provider migration",
        occurred_at_min=NOW - timedelta(days=1),
        occurred_at_max=NOW,
        require_indexed_cursor="cursor-10",
    )

    assert structural.query == ""
    assert semantic.mode is SearchMode.SEMANTIC
    with pytest.raises(ValueError, match="require a query"):
        SearchQuery(corpus="memory", mode=SearchMode.HYBRID, scope=SCOPE)


def test_search_document_and_result_freeze_metadata() -> None:
    metadata = {"tags": ["storage"]}
    document = SearchDocument(
        corpus="memory",
        item_id="event-1",
        text="storage migration",
        scope=SCOPE,
        occurred_at=NOW,
        tags=("storage", "canonical"),
        metadata=metadata,
    )
    result = SearchResult(
        corpus="memory",
        item_id="event-1",
        score=0.75,
        mode=SearchMode.HYBRID,
        metadata=metadata,
    )
    metadata["tags"].append("changed")

    assert document.metadata["tags"] == ("storage",)
    assert document.tags == ("canonical", "storage")
    assert result.metadata["tags"] == ("storage",)
    with pytest.raises(ValueError, match="duplicates"):
        SearchQuery(
            corpus="memory",
            mode=SearchMode.STRUCTURAL,
            scope=SCOPE,
            tags=("same", "same"),
        )


def test_cursor_pages_are_bounded_and_immutable() -> None:
    request = PageRequest(limit=50, cursor="opaque-1")
    page = Page(items=("a", "b"), next_cursor="opaque-2")

    assert request.limit == 50
    assert page.items == ("a", "b")
    with pytest.raises(ValueError, match="between"):
        PageRequest(limit=0)
    with pytest.raises(TypeError, match="tuple"):
        Page(items=["a"])  # type: ignore[arg-type]

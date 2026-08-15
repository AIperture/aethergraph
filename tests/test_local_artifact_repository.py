from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import inspect
from pathlib import Path

import pytest
from storage_conformance.suite import check_artifact_repository_conformance

from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactOccurrenceQuery,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    ArtifactRepository,
    ArtifactRetentionRecord,
    EventStore,
    PageRequest,
    StorageConfigurationError,
    StorageConflictError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalArtifactRepository,
    LocalBlobStore,
    LocalDatabaseRole,
    LocalSQLiteDatabase,
)

NOW = datetime(2026, 8, 15, 20, tzinfo=UTC)


def _database(root: Path, mode: StorageOpenMode) -> LocalSQLiteDatabase:
    return LocalSQLiteDatabase.open(
        workspace_root=root,
        role=LocalDatabaseRole.CONTROL,
        mode=mode,
    )


def _artifact(
    artifact_id: str,
    scope: StorageScope,
    *,
    kind: str = "result",
    labels: dict[str, object] | None = None,
) -> ArtifactRecord:
    payload = _artifact_payload(artifact_id)
    digest = hashlib.sha256(payload).hexdigest()
    return ArtifactRecord(
        artifact_id=artifact_id,
        content_hash=digest,
        hash_algorithm="sha256",
        size_bytes=len(payload),
        media_type="application/json",
        kind=kind,
        blob_locator=f"blob:sha256:{digest}",
        owner_scope=scope,
        created_at=NOW,
        provider_version=f"sha256:{digest}",
        labels=labels or {},
    )


def _artifact_payload(artifact_id: str) -> bytes:
    return f"canonical artifact content:{artifact_id}".encode()


async def _publish(
    repository: LocalArtifactRepository,
    blobs: LocalBlobStore,
    record: ArtifactRecord,
) -> ArtifactRecord:
    async def chunks():
        yield _artifact_payload(record.artifact_id)

    blob = await blobs.put(record.owner_scope, chunks())
    assert (
        blob.blob_locator,
        blob.content_hash,
        blob.hash_algorithm,
        blob.size_bytes,
        blob.provider_version,
    ) == (
        record.blob_locator,
        record.content_hash,
        record.hash_algorithm,
        record.size_bytes,
        record.provider_version,
    )
    return await repository.put(record)


async def _publish_blob(blobs: LocalBlobStore, record: ArtifactRecord) -> None:
    async def chunks():
        yield _artifact_payload(record.artifact_id)

    await blobs.put(record.owner_scope, chunks())


def test_artifact_query_contract_is_bounded_immutable_and_owned() -> None:
    owner = StorageScope(tenant_id="tenant-1", project_id="project-1")
    query = ArtifactOccurrenceQuery(
        owner_scope=owner,
        scope=StorageScope(run_id="run-1"),
        tags=("final",),
        labels={"stage": "review"},
    )
    assert query.labels == {"stage": "review"}
    assert "get_many" in ArtifactRepository.__dict__
    assert "get_many" not in EventStore.__dict__

    with pytest.raises(TypeError, match="immutable tuple"):
        ArtifactOccurrenceQuery(
            owner_scope=owner,
            scope=StorageScope(run_id="run-1"),
            tags=["final"],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="must not exceed 500"):
        ArtifactOccurrenceQuery(
            owner_scope=owner,
            scope=StorageScope(run_id="run-1"),
            page=PageRequest(limit=501),
        )
    with pytest.raises(ValueError, match="conflict"):
        ArtifactOccurrenceQuery(
            owner_scope=owner,
            scope=StorageScope(project_id="other", run_id="run-1"),
        )


@pytest.mark.asyncio
async def test_artifact_reference_commit_validates_blob_and_races_delete_safely(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)
    owner = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_owner = StorageScope(tenant_id="tenant-1", project_id="project-2")
    record = _artifact("verified", owner)

    with pytest.raises(StorageNotFoundError, match="blob:sha256"):
        await repository.put(record)
    await _publish_blob(blobs, replace(record, owner_scope=other_owner))
    with pytest.raises(StorageNotFoundError, match="blob:sha256"):
        await repository.put(record)
    await _publish_blob(blobs, record)
    with pytest.raises(StorageIntegrityError, match="conflicts with scoped blob"):
        await repository.put(replace(record, size_bytes=record.size_bytes + 1))
    assert await repository.put(record) == record
    with pytest.raises(StorageConflictError, match="referenced by an artifact"):
        await blobs.delete(owner, record.blob_locator, record.provider_version)
    assert await repository.get(owner, record.artifact_id) == record
    assert await blobs.head(owner, record.blob_locator) is not None

    for index in range(12):
        candidate = _artifact(f"race-{index}", owner)
        await _publish_blob(blobs, candidate)

        async def commit(current: ArtifactRecord = candidate) -> str:
            try:
                await repository.put(current)
            except StorageNotFoundError:
                return "commit_missing"
            return "commit_won"

        async def remove(current: ArtifactRecord = candidate) -> str:
            try:
                deleted = await blobs.delete(
                    owner,
                    current.blob_locator,
                    current.provider_version,
                )
            except StorageConflictError:
                return "delete_conflict"
            return "delete_won" if deleted else "delete_missing"

        operations = (commit(), remove()) if index % 2 == 0 else (remove(), commit())
        outcomes = set(await asyncio.gather(*operations))
        stored = await repository.get(owner, candidate.artifact_id)
        head = await blobs.head(owner, candidate.blob_locator)
        if stored is None:
            assert outcomes == {"commit_missing", "delete_won"}
            assert head is None
        else:
            assert outcomes == {"commit_won", "delete_conflict"}
            assert head is not None
    await database.close()


@pytest.mark.asyncio
async def test_query_occurrences_filters_before_cursor_and_fails_closed(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)
    owner = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_owner = StorageScope(tenant_id="tenant-1", project_id="project-2")
    report = _artifact(
        "report",
        owner,
        kind="report",
        labels={"tags": ("final", "reviewed"), "category": "evidence"},
    )
    draft = _artifact(
        "draft",
        owner,
        kind="report",
        labels={"tags": "draft, reviewed", "category": "evidence"},
    )
    dataset = _artifact(
        "dataset",
        owner,
        kind="dataset",
        labels={"tags": ("final",), "category": "evidence"},
    )
    hidden = _artifact(
        "hidden",
        other_owner,
        kind="report",
        labels={"tags": ("final",), "category": "evidence"},
    )
    for record in (report, draft, dataset, hidden):
        await _publish(repository, blobs, record)
    for record in (report, dataset, hidden):
        await repository.compare_and_set_retention(
            ArtifactRetentionRecord(
                artifact_id=record.artifact_id,
                scope=record.owner_scope,
                pinned=True,
                revision=1,
                updated_at=NOW,
            ),
            0,
        )

    expected: list[ArtifactOccurrence] = []
    for index, (record, execution) in enumerate(
        (
            (report, StorageScope(**owner.as_filter(), run_id="run-1", session_id="s-1")),
            (report, StorageScope(**owner.as_filter(), run_id="run-1", session_id="s-2")),
            (draft, StorageScope(**owner.as_filter(), run_id="run-1", session_id="s-1")),
            (dataset, StorageScope(**owner.as_filter(), run_id="run-1", session_id="s-1")),
            (report, StorageScope(**owner.as_filter(), run_id="run-2", session_id="s-1")),
            (
                hidden,
                StorageScope(**other_owner.as_filter(), run_id="run-1", session_id="s-1"),
            ),
        )
    ):
        occurrence = ArtifactOccurrence(
            occurrence_id=f"occurrence-{index}",
            artifact_id=record.artifact_id,
            scope=execution,
            action=ArtifactAction.PRODUCED,
            occurred_at=NOW + timedelta(microseconds=index),
        )
        await repository.record_occurrence(occurrence)
        if record is report and execution.run_id == "run-1":
            expected.append(occurrence)

    query = ArtifactOccurrenceQuery(
        owner_scope=owner,
        scope=StorageScope(run_id="run-1"),
        page=PageRequest(limit=1),
        kind="report",
        tags=("final", "reviewed"),
        labels={"category": "evidence"},
        pinned=True,
    )
    first = await repository.query_occurrences(query)
    assert first.items == (expected[1],)
    assert first.next_cursor is not None
    second = await repository.query_occurrences(
        replace(query, page=PageRequest(limit=1, cursor=first.next_cursor))
    )
    assert second.items == (expected[0],)
    assert second.next_cursor is None

    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.query_occurrences(
            replace(
                query,
                kind="dataset",
                page=PageRequest(limit=1, cursor=first.next_cursor),
            )
        )
    assert (
        await repository.query_occurrences(
            ArtifactOccurrenceQuery(
                owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-3"),
                scope=StorageScope(run_id="run-1"),
                kind="report",
                tags=("final",),
                pinned=True,
            )
        )
    ).items == ()
    assert (
        await repository.query_occurrences(
            ArtifactOccurrenceQuery(
                owner_scope=owner,
                scope=StorageScope(run_id="run-1"),
                artifact_id="draft",
                pinned=False,
            )
        )
    ).items[0].artifact_id == "draft"
    await database.close()


@pytest.mark.asyncio
async def test_local_artifact_repository_passes_shared_conformance(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)

    await check_artifact_repository_conformance(repository, blobs)

    await database.close()


@pytest.mark.asyncio
async def test_conflicting_idempotency_keys_fail_without_mutation(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    run_scope = replace(scope, run_id="run-1")
    source = _artifact("source", scope, kind="source")
    target = _artifact("target", scope)
    await _publish(repository, blobs, source)
    await _publish(repository, blobs, target)

    with pytest.raises(StorageIntegrityError, match="conflicting metadata"):
        await repository.put(replace(source, labels={"conflict": True}))

    occurrence = ArtifactOccurrence(
        occurrence_id="occurrence-1",
        artifact_id=target.artifact_id,
        scope=run_scope,
        action=ArtifactAction.PRODUCED,
        occurred_at=NOW,
    )
    assert await repository.record_occurrence(occurrence) == occurrence
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await repository.record_occurrence(replace(occurrence, action=ArtifactAction.CONSUMED))

    relation = ArtifactRelation(
        relation_id="relation-1",
        source_artifact_id=source.artifact_id,
        target_artifact_id=target.artifact_id,
        kind=ArtifactRelationKind.DERIVED_FROM,
        scope=scope,
        created_at=NOW,
    )
    assert await repository.add_relation(relation) == relation
    with pytest.raises(StorageIntegrityError, match="conflicts"):
        await repository.add_relation(replace(relation, kind=ArtifactRelationKind.TRANSFORMED_FROM))

    assert await repository.get(scope, source.artifact_id) == source
    assert (await repository.list_occurrences(run_scope, PageRequest())).items == (occurrence,)
    assert (await repository.list_relations(scope, target.artifact_id, PageRequest())).items == (
        relation,
    )
    await database.close()


@pytest.mark.asyncio
async def test_occurrences_and_lineage_fail_closed_across_scope(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)
    owner_scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_scope = StorageScope(tenant_id="tenant-1", project_id="project-2")
    source = _artifact("source", owner_scope, kind="source")
    target = _artifact("target", owner_scope)
    foreign = _artifact("foreign", other_scope)
    for artifact in (source, target, foreign):
        await _publish(repository, blobs, artifact)

    with pytest.raises(StorageNotFoundError, match=target.artifact_id):
        await repository.record_occurrence(
            ArtifactOccurrence(
                occurrence_id="foreign-occurrence",
                artifact_id=target.artifact_id,
                scope=replace(other_scope, run_id="run-1"),
                action=ArtifactAction.CONSUMED,
                occurred_at=NOW,
            )
        )

    with pytest.raises(StorageNotFoundError, match="lineage endpoint"):
        await repository.add_relation(
            ArtifactRelation(
                relation_id="cross-scope-relation",
                source_artifact_id=source.artifact_id,
                target_artifact_id=foreign.artifact_id,
                kind=ArtifactRelationKind.REFERENCES,
                scope=owner_scope,
                created_at=NOW,
            )
        )
    assert (await repository.list_occurrences(other_scope, PageRequest())).items == ()
    assert (
        await repository.list_relations(owner_scope, source.artifact_id, PageRequest())
    ).items == ()
    await database.close()


@pytest.mark.asyncio
async def test_occurrence_and_relation_cursors_are_bound_to_query_context(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)
    scope = StorageScope(project_id="project-1")
    run_scope = replace(scope, run_id="run-1")
    source = _artifact("source", scope, kind="source")
    target = _artifact("target", scope)
    await _publish(repository, blobs, source)
    await _publish(repository, blobs, target)
    for index in range(3):
        await repository.record_occurrence(
            ArtifactOccurrence(
                occurrence_id=f"occurrence-{index}",
                artifact_id=target.artifact_id,
                scope=run_scope,
                action=ArtifactAction.PRODUCED,
                occurred_at=NOW,
            )
        )
    occurrence_page = await repository.list_occurrences(run_scope, PageRequest(limit=1))
    assert occurrence_page.next_cursor is not None
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.list_occurrences(
            run_scope,
            PageRequest(limit=1, cursor=occurrence_page.next_cursor),
            target.artifact_id,
        )

    for index, kind in enumerate(
        (ArtifactRelationKind.DERIVED_FROM, ArtifactRelationKind.REFERENCES)
    ):
        await repository.add_relation(
            ArtifactRelation(
                relation_id=f"relation-{index}",
                source_artifact_id=source.artifact_id,
                target_artifact_id=target.artifact_id,
                kind=kind,
                scope=scope,
                created_at=NOW,
            )
        )
    relation_page = await repository.list_relations(
        scope,
        target.artifact_id,
        PageRequest(limit=1),
    )
    assert relation_page.next_cursor is not None
    with pytest.raises(StorageConfigurationError, match="mismatched"):
        await repository.list_relations(
            scope,
            source.artifact_id,
            PageRequest(limit=1, cursor=relation_page.next_cursor),
        )
    await database.close()


@pytest.mark.asyncio
async def test_schema_normalizes_content_occurrences_and_lineage(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    LocalArtifactRepository(database=database)

    artifact_columns = {
        str(row["name"]) for row in await database.fetch_all("PRAGMA table_info(local_artifacts)")
    }
    occurrence_columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_artifact_occurrences)")
    }
    relation_columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_artifact_relations)")
    }
    retention_columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_artifact_retention)")
    }
    label_columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_artifact_labels)")
    }
    tag_columns = {
        str(row["name"])
        for row in await database.fetch_all("PRAGMA table_info(local_artifact_tags)")
    }
    assert not {"run_id", "session_id", "action"} & artifact_columns
    assert {
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
    } <= occurrence_columns
    assert not {"content_hash", "blob_locator", "media_type"} & occurrence_columns
    assert not {"content_hash", "blob_locator", "media_type"} & relation_columns
    assert retention_columns == {
        "artifact_id",
        "owner_scope_identity",
        "pinned",
        "revision",
        "updated_at",
        "schema_version",
    }
    assert not {"run_id", "session_id", "labels_json", "content_hash"} & retention_columns
    assert label_columns == {"artifact_id", "label_key", "value_json"}
    assert tag_columns == {"artifact_id", "tag"}

    occurrence_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM local_artifact_occurrences
            WHERE scope_identity = ? AND artifact_id = ? AND sequence < ?
            ORDER BY sequence DESC LIMIT ?
            """,
            ('{"project_id":"project-1"}', "artifact-1", 10, 50),
        )
    )
    relation_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM local_artifact_relations
            WHERE scope_identity = ?
              AND (source_artifact_id = ? OR target_artifact_id = ?)
              AND sequence < ?
            ORDER BY sequence DESC LIMIT ?
            """,
            ('{"project_id":"project-1"}', "artifact-1", "artifact-1", 10, 50),
        )
    )
    assert "ix_local_occurrences_artifact" in occurrence_plan
    assert "SCAN local_artifact_occurrences" not in occurrence_plan
    assert "ix_local_relations_source" in relation_plan
    assert "ix_local_relations_target" in relation_plan
    assert "SCAN local_artifact_relations" not in relation_plan
    content_batch_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM local_artifacts
            WHERE owner_scope_identity = ? AND artifact_id IN (?, ?)
            """,
            ('{"project_id":"project-1"}', "artifact-1", "artifact-2"),
        )
    )
    retention_batch_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT * FROM local_artifact_retention
            WHERE owner_scope_identity = ? AND artifact_id IN (?, ?)
            """,
            ('{"project_id":"project-1"}', "artifact-1", "artifact-2"),
        )
    )
    assert "ix_local_artifacts_scope" in content_batch_plan
    assert "SCAN local_artifacts" not in content_batch_plan
    assert "ix_local_artifact_retention_scope" in retention_batch_plan
    assert "SCAN local_artifact_retention" not in retention_batch_plan
    filtered_occurrence_plan = " ".join(
        str(row["detail"])
        for row in await database.fetch_all(
            """
            EXPLAIN QUERY PLAN
            SELECT o.* FROM local_artifact_occurrences o
            JOIN local_artifacts a ON a.artifact_id = o.artifact_id
            LEFT JOIN local_artifact_retention r ON r.artifact_id = o.artifact_id
            WHERE a.owner_scope_identity = ?
              AND o.run_id = ?
              AND a.kind = ?
              AND EXISTS (
                  SELECT 1 FROM local_artifact_tags tag_filter
                  WHERE tag_filter.artifact_id = o.artifact_id AND tag_filter.tag = ?
              )
              AND EXISTS (
                  SELECT 1 FROM local_artifact_labels label_filter
                  WHERE label_filter.artifact_id = o.artifact_id
                    AND label_filter.label_key = ? AND label_filter.value_json = ?
              )
              AND COALESCE(r.pinned, 0) = ?
              AND o.sequence < ?
            ORDER BY o.sequence DESC LIMIT ?
            """,
            (
                '{"project_id":"project-1"}',
                "run-1",
                "report",
                "final",
                "category",
                '"evidence"',
                1,
                10,
                50,
            ),
        )
    )
    assert "ix_local_occurrences_run" in filtered_occurrence_plan
    assert "local_artifact_tags" in filtered_occurrence_plan
    assert "local_artifact_labels" in filtered_occurrence_plan
    assert "SCAN o" not in filtered_occurrence_plan
    assert "SCAN tag_filter" not in filtered_occurrence_plan
    assert "SCAN label_filter" not in filtered_occurrence_plan
    await database.close()


@pytest.mark.asyncio
async def test_read_only_repository_reads_and_rejects_all_mutations(tmp_path: Path) -> None:
    scope = StorageScope(project_id="project-1")
    run_scope = replace(scope, run_id="run-1")
    source = _artifact("source", scope, kind="source")
    target = _artifact("target", scope)
    occurrence = ArtifactOccurrence(
        occurrence_id="occurrence-1",
        artifact_id=target.artifact_id,
        scope=run_scope,
        action=ArtifactAction.PRODUCED,
        occurred_at=NOW,
    )
    relation = ArtifactRelation(
        relation_id="relation-1",
        source_artifact_id=source.artifact_id,
        target_artifact_id=target.artifact_id,
        kind=ArtifactRelationKind.DERIVED_FROM,
        scope=scope,
        created_at=NOW,
    )
    writable_database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    writable_blobs = LocalBlobStore(database=writable_database, workspace_root=tmp_path)
    writable = LocalArtifactRepository(database=writable_database)
    await _publish(writable, writable_blobs, source)
    await _publish(writable, writable_blobs, target)
    retention = ArtifactRetentionRecord(
        artifact_id=target.artifact_id,
        scope=scope,
        pinned=True,
        revision=1,
        updated_at=NOW,
    )
    await writable.compare_and_set_retention(retention, 0)
    await writable.record_occurrence(occurrence)
    await writable.add_relation(relation)
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    LocalBlobStore(database=readonly_database, workspace_root=tmp_path)
    readonly = LocalArtifactRepository(database=readonly_database)
    assert await readonly.get(scope, target.artifact_id) == target
    assert await readonly.get_many(scope, (target.artifact_id, "missing")) == (target, None)
    assert await readonly.get_retention(scope, target.artifact_id) == retention
    assert await readonly.get_retention_many(scope, (target.artifact_id, "missing")) == (
        retention,
        None,
    )
    assert (await readonly.list_occurrences(run_scope, PageRequest())).items == (occurrence,)
    assert (await readonly.list_relations(scope, target.artifact_id, PageRequest())).items == (
        relation,
    )
    with pytest.raises(StorageReadOnlyError):
        await readonly.put(_artifact("new", scope))
    with pytest.raises(StorageReadOnlyError):
        await readonly.compare_and_set_retention(replace(retention, revision=2), 1)
    with pytest.raises(StorageReadOnlyError):
        await readonly.record_occurrence(replace(occurrence, occurrence_id="new"))
    with pytest.raises(StorageReadOnlyError):
        await readonly.add_relation(replace(relation, relation_id="new"))
    await readonly_database.close()


@pytest.mark.asyncio
async def test_artifact_batch_reads_are_bounded_and_validate_every_slot(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalArtifactRepository(database=database)
    scope = StorageScope(project_id="project-1")

    assert await repository.get_many(scope, ()) == ()
    assert await repository.get_retention_many(scope, ()) == ()
    with pytest.raises(StorageConfigurationError, match="at most 500"):
        await repository.get_many(scope, tuple(f"artifact-{index}" for index in range(501)))
    with pytest.raises(StorageConfigurationError, match="at most 500"):
        await repository.get_retention_many(
            scope,
            tuple(f"artifact-{index}" for index in range(501)),
        )
    with pytest.raises(TypeError, match="sequence"):
        await repository.get_many(scope, "artifact-1")
    with pytest.raises(ValueError, match="non-empty"):
        await repository.get_retention_many(scope, ("",))
    await database.close()


@pytest.mark.asyncio
async def test_retention_cas_has_one_concurrent_winner(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    blobs = LocalBlobStore(database=database, workspace_root=tmp_path)
    repository = LocalArtifactRepository(database=database)
    scope = StorageScope(project_id="project-1")
    artifact = _artifact("artifact-1", scope)
    await _publish(repository, blobs, artifact)

    candidate = ArtifactRetentionRecord(
        artifact_id=artifact.artifact_id,
        scope=scope,
        pinned=True,
        revision=1,
        updated_at=NOW,
    )

    async def attempt() -> str:
        try:
            await repository.compare_and_set_retention(candidate, 0)
        except StorageConflictError:
            return "conflict"
        return "winner"

    outcomes = await asyncio.gather(*(attempt() for _index in range(8)))
    assert outcomes.count("winner") == 1
    assert outcomes.count("conflict") == 7
    assert await repository.get_retention(scope, artifact.artifact_id) == candidate
    with pytest.raises(StorageConflictError, match="backward"):
        await repository.compare_and_set_retention(
            replace(
                candidate,
                pinned=False,
                revision=2,
                updated_at=NOW - timedelta(seconds=1),
            ),
            1,
        )
    await database.close()


def test_artifact_batch_and_retention_methods_follow_required_docstring_sections() -> None:
    for name in (
        "get_many",
        "get_retention",
        "get_retention_many",
        "compare_and_set_retention",
    ):
        docstring = inspect.getdoc(getattr(LocalArtifactRepository, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from storage_conformance.suite import check_artifact_repository_conformance

from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactRecord,
    ArtifactRelation,
    ArtifactRelationKind,
    PageRequest,
    StorageConfigurationError,
    StorageIntegrityError,
    StorageNotFoundError,
    StorageOpenMode,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalArtifactRepository,
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
) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=artifact_id,
        content_hash=f"hash-{artifact_id}",
        hash_algorithm="sha256",
        size_bytes=10,
        media_type="application/json",
        kind=kind,
        blob_locator=f"blob:{artifact_id}",
        owner_scope=scope,
        created_at=NOW,
    )


@pytest.mark.asyncio
async def test_local_artifact_repository_passes_shared_conformance(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalArtifactRepository(database=database)

    await check_artifact_repository_conformance(repository)

    await database.close()


@pytest.mark.asyncio
async def test_conflicting_idempotency_keys_fail_without_mutation(tmp_path: Path) -> None:
    database = _database(tmp_path, StorageOpenMode.READ_WRITE)
    repository = LocalArtifactRepository(database=database)
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    run_scope = replace(scope, run_id="run-1")
    source = _artifact("source", scope, kind="source")
    target = _artifact("target", scope)
    await repository.put(source)
    await repository.put(target)

    with pytest.raises(StorageIntegrityError, match="conflicting metadata"):
        await repository.put(replace(source, size_bytes=11))

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
    repository = LocalArtifactRepository(database=database)
    owner_scope = StorageScope(tenant_id="tenant-1", project_id="project-1")
    other_scope = StorageScope(tenant_id="tenant-1", project_id="project-2")
    source = _artifact("source", owner_scope, kind="source")
    target = _artifact("target", owner_scope)
    foreign = _artifact("foreign", other_scope)
    for artifact in (source, target, foreign):
        await repository.put(artifact)

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
    repository = LocalArtifactRepository(database=database)
    scope = StorageScope(project_id="project-1")
    run_scope = replace(scope, run_id="run-1")
    source = _artifact("source", scope, kind="source")
    target = _artifact("target", scope)
    await repository.put(source)
    await repository.put(target)
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
    assert not {"run_id", "session_id", "action"} & artifact_columns
    assert not {"content_hash", "blob_locator", "media_type"} & occurrence_columns
    assert not {"content_hash", "blob_locator", "media_type"} & relation_columns

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
    writable = LocalArtifactRepository(database=writable_database)
    await writable.put(source)
    await writable.put(target)
    await writable.record_occurrence(occurrence)
    await writable.add_relation(relation)
    await writable_database.close()

    readonly_database = _database(tmp_path, StorageOpenMode.READ_ONLY)
    readonly = LocalArtifactRepository(database=readonly_database)
    assert await readonly.get(scope, target.artifact_id) == target
    assert (await readonly.list_occurrences(run_scope, PageRequest())).items == (occurrence,)
    assert (await readonly.list_relations(scope, target.artifact_id, PageRequest())).items == (
        relation,
    )
    with pytest.raises(StorageReadOnlyError):
        await readonly.put(_artifact("new", scope))
    with pytest.raises(StorageReadOnlyError):
        await readonly.record_occurrence(replace(occurrence, occurrence_id="new"))
    with pytest.raises(StorageReadOnlyError):
        await readonly.add_relation(replace(relation, relation_id="new"))
    await readonly_database.close()

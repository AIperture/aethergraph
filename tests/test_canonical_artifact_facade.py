from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
import io
import os
from pathlib import Path
import shutil
import tarfile

import pytest

from aethergraph.services.artifacts import CanonicalArtifactFacade, CanonicalArtifactWriter
from aethergraph.storage.contracts import (
    ArtifactMetricOrder,
    ArtifactRelationKind,
    PageRequest,
    RunRecord,
    RunStatus,
    SearchDocument,
    SearchMode,
    SessionKind,
    SessionRecord,
    StorageCapacityError,
    StorageIntegrityError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)
_SECRET_REF = "secret://tests/artifacts"
_SECRET = b"canonical-artifact-secret-material-32b"


class _Clock:
    def __init__(self) -> None:
        self.value = NOW

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


class _FailingSearch:
    async def upsert(self, document):
        raise RuntimeError("artifact index unavailable")


def _owner_scope() -> StorageScope:
    return StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
    )


def _execution_scope() -> StorageScope:
    return StorageScope(
        **_owner_scope().as_filter(),
        graph_id="graph-1",
        node_id="node-1",
        agent_id="agent-1",
    )


def _open_bundle(root: Path):
    return LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    ).open(
        StorageOpenRequest(
            workspace_id="canonical-artifact-tests",
            workspace_root=root.resolve(),
            owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={"continuation_token_secret_ref": _SECRET_REF},
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=_Clock(),
            secrets=_Secrets(),
        )
    )


def _facade(
    bundle, *, search=None, execution_scope: StorageScope | None = None
) -> CanonicalArtifactFacade:
    return CanonicalArtifactFacade(
        blobs=bundle.blobs,
        artifacts=bundle.artifacts,
        search=search or bundle.search,
        runs=bundle.runs,
        sessions=bundle.sessions,
        owner_scope=_owner_scope(),
        execution_scope=execution_scope or _execution_scope(),
        tool_name="reporter",
        tool_version="1.0",
        clock=_Clock().now,
    )


@pytest.mark.asyncio
async def test_canonical_artifact_write_retry_hydration_retention_and_search(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        first = await facade.save_text(
            "canonical artifact report",
            kind="report",
            original_filename="report.txt",
            content_labels={"category": "evidence", "tags": ("final", "reviewed")},
            occurrence_labels={"stage": "final"},
            metrics={"quality": 0.9},
            pinned=True,
            artifact_id="artifact-1",
            occurrence_id="occurrence-1",
            occurred_at=NOW,
        )
        retried = await facade.save_text(
            "canonical artifact report",
            kind="report",
            original_filename="report.txt",
            content_labels={"category": "evidence", "tags": ("final", "reviewed")},
            occurrence_labels={"stage": "final"},
            metrics={"quality": 0.9},
            pinned=True,
            artifact_id="artifact-1",
            occurrence_id="occurrence-1",
            occurred_at=NOW,
        )

        assert retried == first
        assert first.record.blob_locator.startswith("blob:sha256:")
        assert first.record.owner_scope == _owner_scope()
        assert first.occurrence.scope == _execution_scope()
        assert first.retention is not None and first.retention.pinned is True
        projected_commit = facade.project_commit(
            first,
            deprecated_app_id="legacy-app",
        )
        assert projected_commit.artifact_id == "artifact-1"
        assert projected_commit.occurrence_id == "occurrence-1"
        assert projected_commit.pinned is True
        assert projected_commit.app_id == "legacy-app"
        assert projected_commit.client_id is None
        assert first.record.blob_locator not in projected_commit.uri
        assert await facade.get_public("artifact-1") == facade.project_commit(first)
        assert (
            await facade.get_public(
                "artifact-1",
                scope=StorageScope(run_id="missing-run"),
            )
            is None
        )
        with pytest.raises(TypeError, match="ArtifactCommitReceipt"):
            facade.project_commit(object())  # type: ignore[arg-type]
        assert await facade.load_bytes("artifact-1") == b"canonical artifact report"
        page = await facade.list_occurrences(PageRequest(limit=5))
        assert page.items == (first.occurrence,)
        hits = await facade.search(
            query="canonical",
            mode=SearchMode.LEXICAL,
            tags=("reviewed", "final"),
            require_indexed_cursor=first.indexed_cursor,
        )
        assert [hit.item_id for hit in hits] == ["artifact-1"]
        assert hits[0].metadata["category"] == "evidence"
        assert "app_id" not in hits[0].metadata
        assert "client_id" not in hits[0].metadata
        public_hits = await facade.search_public_artifacts(
            query="canonical",
            mode=SearchMode.LEXICAL,
            tags=("final",),
            metadata={"category": "evidence"},
            require_indexed_cursor=first.indexed_cursor,
        )
        assert [hit.artifact.artifact_id for hit in public_hits] == ["artifact-1"]
        assert public_hits[0].artifact.occurrence_id == "occurrence-1"
        assert public_hits[0].artifact.app_id is None
        assert public_hits[0].score == hits[0].score
        assert public_hits[0].mode is SearchMode.LEXICAL
        assert (
            await facade.search_public_artifacts(
                query="canonical",
                mode=SearchMode.LEXICAL,
                tags=("missing",),
            )
            == ()
        )
        assert await facade.get_occurrences_many(("occurrence-1", "missing")) == (
            first.occurrence,
            None,
        )
        with pytest.raises(ValueError, match="500"):
            await facade.search_public_artifacts(
                query="canonical",
                mode=SearchMode.LEXICAL,
                top_k=501,
            )
        unpinned = await facade.pin("artifact-1", False)
        assert unpinned.pinned is False and unpinned.revision == 2
        assert await facade.get_retention("artifact-1") == unpinned

        second = await facade.save_json(
            {"source": "artifact-1"},
            artifact_id="artifact-2",
            occurrence_id="occurrence-2",
            occurred_at=NOW,
        )
        relation = await facade.add_relation(
            relation_id="relation-1",
            source_artifact_id=first.record.artifact_id,
            target_artifact_id=second.record.artifact_id,
            kind=ArtifactRelationKind.DERIVED_FROM,
            created_at=NOW,
        )
        assert (await facade.list_relations("artifact-2")).items == (relation,)
        assert await facade.get_many(("artifact-1", "missing", "artifact-2")) == (
            first.record,
            None,
            second.record,
        )
        assert await facade.get_retention_many(("artifact-1", "missing")) == (
            unpinned,
            None,
        )
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_occurrence_pages_filter_then_batch_hydrate(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    first_execution = StorageScope(
        **_owner_scope().as_filter(),
        run_id="run-1",
        session_id="session-1",
        graph_id="graph-1",
        node_id="node-1",
    )
    second_execution = StorageScope(
        **_owner_scope().as_filter(),
        run_id="run-1",
        session_id="session-1",
        graph_id="graph-1",
        node_id="node-1",
    )
    first_facade = _facade(bundle, execution_scope=first_execution)
    second_facade = _facade(bundle, execution_scope=second_execution)
    try:
        await bundle.runs.create(
            RunRecord(
                run_id="run-1",
                graph_id="graph-1",
                kind="graph",
                status=RunStatus.RUNNING,
                scope=first_execution,
                revision=1,
                started_at=NOW,
            )
        )
        for session_id in ("session-1",):
            await bundle.sessions.create(
                SessionRecord(
                    session_id=session_id,
                    kind=SessionKind.CHAT,
                    scope=first_execution,
                    revision=1,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
        first = await first_facade.save_text(
            "first report",
            artifact_id="report-1",
            occurrence_id="occurrence-1",
            kind="report",
            content_labels={"tags": ("final", "reviewed"), "category": "evidence"},
            occurrence_labels={"stage": "published"},
            metrics={"quality": 0.8},
            pinned=True,
            occurred_at=NOW,
        )
        await first_facade.save_text(
            "draft report",
            artifact_id="draft-1",
            occurrence_id="occurrence-draft",
            kind="report",
            content_labels={"tags": ("draft",), "category": "evidence"},
            pinned=False,
            occurred_at=NOW,
        )
        second = await second_facade.save_text(
            "second report",
            artifact_id="report-2",
            occurrence_id="occurrence-2",
            kind="report",
            content_labels={"tags": ("final", "reviewed"), "category": "evidence"},
            occurrence_labels={"stage": "published"},
            metrics={"quality": 0.9},
            pinned=True,
            occurred_at=NOW,
        )

        query_args = {
            "scope": StorageScope(run_id="run-1"),
            "kind": "report",
            "tags": ("final", "reviewed"),
            "labels": {"category": "evidence"},
            "pinned": True,
            "metric": "quality",
            "metric_order": ArtifactMetricOrder.MAXIMUM,
            "deprecated_app_id": "legacy-app",
        }
        page_one = await first_facade.query_public_artifacts(
            PageRequest(limit=1),
            **query_args,
        )
        assert page_one.next_cursor is not None
        assert [item.artifact_id for item in page_one.items] == [second.record.artifact_id]
        projected = page_one.items[0]
        assert projected.session_id == "session-1"
        assert projected.occurrence_id == "occurrence-2"
        assert projected.metrics == {"quality": 0.9}
        assert projected.pinned is True
        assert projected.app_id == "legacy-app"
        assert projected.client_id is None
        assert projected.uri == "/api/v1/artifacts/report-2/content"
        assert second.record.blob_locator not in projected.uri

        page_two = await first_facade.query_public_artifacts(
            PageRequest(limit=1, cursor=page_one.next_cursor),
            **query_args,
        )
        assert page_two.next_cursor is None
        assert [item.artifact_id for item in page_two.items] == [first.record.artifact_id]
        assert page_two.items[0].labels["stage"] == "published"

        raw = await first_facade.query_occurrences(
            scope=StorageScope(session_id="session-1"),
            artifact_id="draft-1",
            pinned=False,
        )
        assert [item.occurrence_id for item in raw.items] == ["occurrence-draft"]
        without_compatibility = await first_facade.query_public_artifacts(
            scope=StorageScope(run_id="run-1"),
            artifact_id="report-1",
        )
        assert without_compatibility.items[0].app_id is None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_orphan_reconciliation_is_owner_scoped(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        referenced = await facade.save_text(
            "referenced content",
            artifact_id="referenced",
            occurrence_id="referenced-occurrence",
            occurred_at=NOW,
        )
        orphan = await bundle.blobs.put(_owner_scope(), _bytes(b"orphan content"))

        result = await facade.reconcile_orphans(
            older_than=NOW + timedelta(days=1),
            limit=10,
        )

        assert result.deleted_scoped_blobs == 1
        assert result.deleted_physical_blobs == 1
        assert await bundle.blobs.head(_owner_scope(), orphan.blob_locator) is None
        assert await bundle.blobs.head(_owner_scope(), referenced.record.blob_locator) is not None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_file_source_is_not_persisted_and_cleanup_is_explicit(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path / "workspace")
    facade = _facade(bundle)
    source = tmp_path / "source.bin"
    source.write_bytes(b"artifact-bytes")
    try:
        receipt = await facade.save_file(
            source,
            kind="binary",
            media_type="application/octet-stream",
            cleanup=True,
            artifact_id="artifact-file",
            occurrence_id="occurrence-file",
            occurred_at=NOW,
        )

        assert not source.exists()
        assert receipt.record.original_filename == "source.bin"
        assert str(source) not in str(receipt.record)
        assert await facade.load_bytes("artifact-file") == b"artifact-bytes"
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_hydration_is_bounded(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        await facade.save_text(
            "too large",
            artifact_id="artifact-large",
            occurrence_id="occurrence-large",
            occurred_at=NOW,
        )
        with pytest.raises(StorageCapacityError, match="bound"):
            await facade.load_bytes("artifact-large", max_bytes=3)
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("label_name", ["app_id", "application_id", "client_id"])
async def test_canonical_artifact_write_rejects_deprecated_identity_labels(
    tmp_path: Path,
    label_name: str,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        with pytest.raises(ValueError, match="deprecated identity"):
            await facade.save_text(
                "identity must remain compatibility-only",
                content_labels={label_name: "legacy-value"},
            )
        with pytest.raises(ValueError, match="deprecated identity"):
            await facade.save_text(
                "identity must remain compatibility-only",
                occurrence_labels={label_name: "legacy-value"},
            )
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_artifact_search_rejects_stale_occurrence_projection(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        receipt = await facade.save_text(
            "canonical report",
            artifact_id="artifact-stale",
            occurrence_id="occurrence-valid",
            occurred_at=NOW,
        )
        await bundle.search.upsert(
            SearchDocument(
                corpus="artifact",
                item_id=receipt.record.artifact_id,
                text="canonical report",
                scope=_owner_scope(),
                occurred_at=NOW,
                metadata={"occurrence_id": "occurrence-missing"},
            )
        )

        with pytest.raises(StorageIntegrityError, match="missing authorized"):
            await facade.search_public_artifacts(
                query="canonical",
                mode=SearchMode.LEXICAL,
            )
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_directory_archive_is_deterministic_and_materializes_safely(
    tmp_path: Path,
) -> None:
    first_source = tmp_path / "first"
    second_source = tmp_path / "second"
    (first_source / "nested").mkdir(parents=True)
    (first_source / "empty").mkdir()
    (first_source / "nested" / "data.txt").write_text("canonical\n", encoding="utf-8")
    (first_source / "root.bin").write_bytes(b"root")
    (second_source / "empty").mkdir(parents=True)
    (second_source / "nested").mkdir()
    (second_source / "root.bin").write_bytes(b"root")
    (second_source / "nested" / "data.txt").write_text("canonical\n", encoding="utf-8")
    for path in first_source.rglob("*"):
        os.utime(path, (1_000_000_000, 1_000_000_000))
    for path in second_source.rglob("*"):
        os.utime(path, (1_700_000_000, 1_700_000_000))

    bundle = _open_bundle(tmp_path / "workspace")
    facade = _facade(bundle)
    try:
        first = await facade.save_directory(
            first_source,
            content_labels={"_aethergraph_directory_format": "caller-value"},
            artifact_id="artifact-directory-1",
            occurrence_id="occurrence-directory-1",
            occurred_at=NOW,
        )
        second = await facade.save_directory(
            second_source,
            artifact_id="artifact-directory-2",
            occurrence_id="occurrence-directory-2",
            occurred_at=NOW,
        )

        assert first.record.content_hash == second.record.content_hash
        assert first.record.media_type == "application/vnd.aethergraph.directory+tar"
        assert first.record.labels["_aethergraph_directory_format"] == "tar.v1"
        assert first.record.labels["_aethergraph_directory_entry_count"] == 4
        assert first.record.labels["_aethergraph_directory_file_count"] == 2
        assert first.record.labels["_aethergraph_directory_total_bytes"] == sum(
            path.stat().st_size for path in first_source.rglob("*") if path.is_file()
        )

        materialized = Path(
            await facade.materialize_directory(
                "artifact-directory-1",
                tmp_path / "materialized",
            )
        )
        assert (materialized / "nested" / "data.txt").read_text(encoding="utf-8") == ("canonical\n")
        assert (materialized / "root.bin").read_bytes() == b"root"
        assert (materialized / "empty").is_dir()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_directory_materialization_enforces_resource_bounds(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "first.txt").write_text("first", encoding="utf-8")
    (source / "second.txt").write_text("second", encoding="utf-8")
    bundle = _open_bundle(tmp_path / "workspace")
    facade = _facade(bundle)
    try:
        with pytest.raises(StorageCapacityError, match="source.*entry bound"):
            await facade.save_directory(
                source,
                cleanup=True,
                max_entries=1,
                artifact_id="artifact-source-bound",
                occurrence_id="occurrence-source-bound",
                occurred_at=NOW,
            )
        assert source.is_dir()
        assert await facade.get("artifact-source-bound") is None

        await facade.save_directory(
            source,
            artifact_id="artifact-directory",
            occurrence_id="occurrence-directory",
            occurred_at=NOW,
        )

        entry_target = tmp_path / "too-many-entries"
        with pytest.raises(StorageCapacityError, match="entry bound"):
            await facade.materialize_directory(
                "artifact-directory",
                entry_target,
                max_entries=1,
            )
        assert not entry_target.exists()

        byte_target = tmp_path / "too-many-bytes"
        with pytest.raises(StorageCapacityError, match="materialization bound"):
            await facade.materialize_directory(
                "artifact-directory",
                byte_target,
                max_total_bytes=5,
            )
        assert not byte_target.exists()

        existing_target = tmp_path / "existing"
        existing_target.mkdir()
        marker = existing_target / "preserved.txt"
        marker.write_text("keep", encoding="utf-8")
        with pytest.raises(FileExistsError, match="already exists"):
            await facade.materialize_directory("artifact-directory", existing_target)
        assert marker.read_text(encoding="utf-8") == "keep"
    finally:
        await bundle.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_kind", ["traversal", "link"])
async def test_canonical_directory_materialization_rejects_unsafe_members(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        if unsafe_kind == "traversal":
            content = b"escape"
            member = tarfile.TarInfo("../escape.txt")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        else:
            member = tarfile.TarInfo("linked")
            member.type = tarfile.SYMTYPE
            member.linkname = "../escape.txt"
            archive.addfile(member)

    bundle = _open_bundle(tmp_path / "workspace")
    facade = _facade(bundle)
    try:
        artifact_id = f"artifact-unsafe-{unsafe_kind}"
        await facade.write(
            _bytes(payload.getvalue()),
            kind="directory",
            media_type="application/vnd.aethergraph.directory+tar",
            content_labels={"_aethergraph_directory_format": "tar.v1"},
            artifact_id=artifact_id,
            occurrence_id=f"occurrence-unsafe-{unsafe_kind}",
            occurred_at=NOW,
        )
        target = tmp_path / f"target-{unsafe_kind}"

        with pytest.raises(StorageIntegrityError, match="archive|path|permit"):
            await facade.materialize_directory(artifact_id, target)
        assert not target.exists()
        assert not (tmp_path / "escape.txt").exists()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_writer_commits_once_and_aborts_without_metadata(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        async with facade.writer(
            kind="report",
            media_type="text/plain",
            planned_ext=".txt",
            original_filename="report.txt",
            artifact_id="artifact-writer",
            occurrence_id="occurrence-writer",
            occurred_at=NOW,
        ) as writer:
            writer.add_labels({"category": "evidence"})
            writer.add_occurrence_labels({"stage": "final"})
            writer.add_metrics({"quality": 0.9})
            await writer.write(b"streamed ")
            await writer.write(b"artifact")

        assert writer.receipt is not None
        assert writer.receipt.record.labels["category"] == "evidence"
        assert writer.receipt.occurrence.labels["stage"] == "final"
        assert writer.receipt.occurrence.metrics["quality"] == 0.9
        assert await facade.load_bytes("artifact-writer") == b"streamed artifact"
        with pytest.raises(RuntimeError, match="closed"):
            await writer.write(b"late")

        with pytest.raises(RuntimeError, match="producer failed"):
            async with facade.writer(
                kind="binary",
                artifact_id="artifact-aborted",
                occurrence_id="occurrence-aborted",
                occurred_at=NOW,
            ) as aborted:
                await aborted.write(b"partial")
                raise RuntimeError("producer failed")
        assert await facade.get("artifact-aborted") is None
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_write_validates_metadata_before_blob_commit(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    try:
        with pytest.raises(ValueError, match="finite"):
            await facade.save_text(
                "invalid occurrence",
                metrics={"quality": float("nan")},
                artifact_id="artifact-invalid",
                occurrence_id="occurrence-invalid",
                occurred_at=NOW,
            )

        assert await facade.get("artifact-invalid") is None
        assert (await facade.list_occurrences(artifact_id="artifact-invalid")).items == ()
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_staging_rejects_path_syntax(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle)
    staged_file: Path | None = None
    staged_dir: Path | None = None
    try:
        staged_file = Path(await facade.stage_path(".txt"))
        staged_dir = Path(await facade.stage_dir("_bundle"))
        assert staged_file.is_file()
        assert staged_dir.is_dir()
        with pytest.raises(ValueError, match="path or drive"):
            await facade.stage_path("../escape")
        with pytest.raises(ValueError, match="path or drive"):
            await facade.stage_dir("C:escape")
    finally:
        if staged_file is not None:
            staged_file.unlink(missing_ok=True)
        if staged_dir is not None and staged_dir.exists():
            shutil.rmtree(staged_dir)
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_run_and_session_counters_are_idempotent(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    execution = StorageScope(
        **_owner_scope().as_filter(),
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
    )
    await bundle.sessions.create(
        SessionRecord(
            session_id="session-1",
            kind=SessionKind.CHAT,
            scope=execution,
            revision=1,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    await bundle.runs.create(
        RunRecord(
            run_id="run-1",
            graph_id="graph-1",
            kind="graph",
            status=RunStatus.RUNNING,
            scope=execution,
            revision=1,
            started_at=NOW,
        )
    )
    facade = _facade(bundle, execution_scope=execution)
    try:
        for _index in range(2):
            await facade.save_text(
                "one counted occurrence",
                artifact_id="artifact-counted",
                occurrence_id="occurrence-counted",
                occurred_at=NOW,
            )

        run = await bundle.runs.get(execution, "run-1")
        session = await bundle.sessions.get(execution, "session-1")
        assert run is not None and run.artifact_count == 1
        assert session is not None and session.artifact_count == 1
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_canonical_artifact_search_failure_is_visible_after_durable_records(
    tmp_path: Path,
) -> None:
    bundle = _open_bundle(tmp_path)
    facade = _facade(bundle, search=_FailingSearch())  # type: ignore[arg-type]
    try:
        with pytest.raises(RuntimeError, match="index unavailable"):
            await facade.save_text(
                "durable before projection",
                artifact_id="artifact-1",
                occurrence_id="occurrence-1",
                occurred_at=NOW,
            )

        assert await bundle.artifacts.get(_owner_scope(), "artifact-1") is not None
        page = await bundle.artifacts.list_occurrences(
            _execution_scope(),
            PageRequest(),
            "artifact-1",
        )
        assert [row.occurrence_id for row in page.items] == ["occurrence-1"]
    finally:
        await bundle.close()


def test_canonical_artifact_scope_and_public_docstrings_fail_closed() -> None:
    with pytest.raises(ValueError, match="every exact"):
        CanonicalArtifactFacade(
            blobs=object(),  # type: ignore[arg-type]
            artifacts=object(),  # type: ignore[arg-type]
            search=object(),  # type: ignore[arg-type]
            runs=object(),  # type: ignore[arg-type]
            sessions=object(),  # type: ignore[arg-type]
            owner_scope=StorageScope(project_id="project-1"),
            execution_scope=StorageScope(project_id="project-2"),
        )

    for name in (
        "stage_path",
        "stage_dir",
        "writer",
        "write",
        "save_file",
        "save_directory",
        "materialize_directory",
        "save_text",
        "save_json",
        "get",
        "get_public",
        "get_many",
        "get_occurrences_many",
        "read",
        "load_bytes",
        "pin",
        "get_retention",
        "get_retention_many",
        "add_relation",
        "list_relations",
        "list_occurrences",
        "query_occurrences",
        "query_public_artifacts",
        "reconcile_orphans",
        "project_commit",
        "search",
        "search_public_artifacts",
    ):
        docstring = inspect.getdoc(getattr(CanonicalArtifactFacade, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

    for name in ("receipt", "write", "add_labels", "add_occurrence_labels", "add_metrics"):
        docstring = inspect.getdoc(getattr(CanonicalArtifactWriter, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2


async def _bytes(payload: bytes):
    yield payload

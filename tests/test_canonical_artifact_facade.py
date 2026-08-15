from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.services.artifacts import CanonicalArtifactFacade
from aethergraph.storage.contracts import (
    ArtifactRelationKind,
    PageRequest,
    RunRecord,
    RunStatus,
    SearchMode,
    SessionKind,
    SessionRecord,
    StorageCapacityError,
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
            content_labels={"category": "evidence"},
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
            content_labels={"category": "evidence"},
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
        assert await facade.load_bytes("artifact-1") == b"canonical artifact report"
        page = await facade.list_occurrences(PageRequest(limit=5))
        assert page.items == (first.occurrence,)
        hits = await facade.search(
            query="canonical",
            mode=SearchMode.LEXICAL,
            require_indexed_cursor=first.indexed_cursor,
        )
        assert [hit.item_id for hit in hits] == ["artifact-1"]
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
        "write",
        "save_file",
        "save_text",
        "save_json",
        "get",
        "read",
        "load_bytes",
        "pin",
        "get_retention",
        "add_relation",
        "list_relations",
        "list_occurrences",
        "search",
    ):
        docstring = inspect.getdoc(getattr(CanonicalArtifactFacade, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

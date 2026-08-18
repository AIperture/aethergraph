from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.services.artifacts import (
    CanonicalArtifactFacadeFactory,
    CanonicalPublicArtifactFacade,
)
from aethergraph.storage.contracts import (
    PageRequest,
    SearchMode,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

NOW = datetime(2026, 8, 16, 12, tzinfo=UTC)
_SECRET_REF = "secret://tests/artifact-factory"
_SECRET = b"canonical-artifact-factory-secret-32b"


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


def _owner_scope() -> StorageScope:
    return StorageScope(tenant_id="tenant-1", project_id="project-1")


def _open_bundle(root: Path, clock: _Clock):
    return LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    ).open(
        StorageOpenRequest(
            workspace_id="canonical-artifact-factory-tests",
            workspace_root=root.resolve(),
            owner_scope=_owner_scope(),
            selection=StorageProviderSelection(
                provider="local.sqlite",
                config={"continuation_token_secret_ref": _SECRET_REF},
            ),
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=1,
            clock=clock,
            secrets=_Secrets(),
        )
    )


@pytest.mark.asyncio
async def test_factory_binds_provider_owner_to_partial_execution_scope(tmp_path: Path) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    artifact_ids = iter(("artifact-1",))
    occurrence_ids = iter(("occurrence-1",))
    factory = CanonicalArtifactFacadeFactory(
        bundle=bundle,
        owner_scope=_owner_scope(),
        clock=clock.now,
        artifact_id_factory=lambda: next(artifact_ids),
        occurrence_id_factory=lambda: next(occurrence_ids),
    )
    try:
        facade = factory.for_execution(
            StorageScope(
                graph_id="graph-1",
                node_id="node-1",
            ),
            tool_name="reporter",
            tool_version="1.0",
        )

        assert facade.owner_scope == _owner_scope()
        assert facade.execution_scope == StorageScope(
            **_owner_scope().as_filter(),
            graph_id="graph-1",
            node_id="node-1",
        )
        assert "app_id" not in facade.execution_scope.as_filter()
        assert "client_id" not in facade.execution_scope.as_filter()

        receipt = await facade.save_text("factory-bound artifact")
        projected = facade.project_commit(receipt)
        assert projected.artifact_id == "artifact-1"
        assert projected.occurrence_id == "occurrence-1"
        assert projected.graph_id == "graph-1"
        assert projected.app_id is None
        assert projected.client_id is None

        owner_facade = factory.for_owner()
        assert owner_facade.owner_scope == _owner_scope()
        assert owner_facade.execution_scope == _owner_scope()
        page = await owner_facade.query_public_artifacts(
            scope=StorageScope(graph_id="graph-1"),
        )
        assert [item.artifact_id for item in page.items] == ["artifact-1"]
    finally:
        await bundle.close()


def test_factory_rejects_empty_owner_and_execution_owner_conflicts() -> None:
    with pytest.raises(ValueError, match="owner_scope"):
        CanonicalArtifactFacadeFactory(
            bundle=object(),  # type: ignore[arg-type]
            owner_scope=StorageScope(),
            clock=lambda: NOW,
        )

    factory = CanonicalArtifactFacadeFactory(
        bundle=object(),  # type: ignore[arg-type]
        owner_scope=_owner_scope(),
        clock=lambda: NOW,
    )
    with pytest.raises(ValueError, match="project_id"):
        factory.for_execution(StorageScope(project_id="project-2"))


@pytest.mark.asyncio
async def test_public_factory_projects_runtime_write_read_and_search_core(tmp_path: Path) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    artifact_ids = iter(("artifact/1", "artifact-2", "artifact-3"))
    occurrence_ids = iter(("occurrence-1", "occurrence-2", "occurrence-3"))
    factory = CanonicalArtifactFacadeFactory(
        bundle=bundle,
        owner_scope=_owner_scope(),
        clock=clock.now,
        artifact_id_factory=lambda: next(artifact_ids),
        occurrence_id_factory=lambda: next(occurrence_ids),
    )
    artifacts = factory.for_public_execution(
        StorageScope(graph_id="graph-1", node_id="node-1"),
        tool_name="reporter",
        tool_version="1.0",
        deprecated_app_id="app-legacy",
    )
    try:
        text = await artifacts.save_text(
            "canonical artifact evidence",
            name="report.txt",
            kind="report",
            tags=["verified"],
            metrics={"quality": 0.9},
            pin=True,
        )
        structured = await artifacts.save_json(
            {"status": "ok"},
            suggested_uri="artifact://results/result.json",
        )
        async with artifacts.writer(kind="binary", planned_ext=".bin") as writer:
            writer.add_labels({"category": "stream"})
            await writer.write(b"streamed")
        streamed = artifacts.last_artifact

        assert isinstance(artifacts, CanonicalPublicArtifactFacade)
        assert text.uri == "/api/v1/artifacts/artifact%2F1/content"
        assert text.app_id == "app-legacy"
        assert text.client_id is None
        assert text.tags == ["verified"]
        assert text.labels["filename"] == "report.txt"
        assert text.metrics == {"quality": 0.9}
        assert text.pinned is True
        assert structured.labels["filename"] == "result.json"
        assert streamed is not None
        assert streamed.artifact_id == "artifact-3"
        assert streamed.labels["category"] == "stream"

        assert await artifacts.get_by_id(text.artifact_id) == text
        assert await artifacts.load_text_by_id(text.artifact_id) == "canonical artifact evidence"
        assert await artifacts.load_text(text.uri) == "canonical artifact evidence"
        assert await artifacts.load_text("artifact://artifact%2F1") == (
            "canonical artifact evidence"
        )
        assert await artifacts.load_json_by_id(structured.artifact_id) == {"status": "ok"}

        hits = await artifacts.search_public_artifacts(
            query="evidence",
            mode=SearchMode.LEXICAL,
            tags=["verified"],
        )
        assert [hit.artifact.artifact_id for hit in hits] == [text.artifact_id]
        assert hits[0].artifact.app_id == "app-legacy"
        assert hits[0].mode is SearchMode.LEXICAL

        await artifacts.pin(text.artifact_id, pinned=False)
        assert (await artifacts.get_by_id(text.artifact_id)).pinned is False  # type: ignore[union-attr]
        with pytest.raises(ValueError, match="public Artifact identity"):
            await artifacts.load_text("file:///tmp/provider-path.txt")
        with pytest.raises(ValueError, match="query"):
            await artifacts.load_text("artifact://artifact-1?version=2")
        with pytest.raises(FileNotFoundError, match="missing"):
            await artifacts.load_bytes_by_id("missing")
    finally:
        await bundle.close()


@pytest.mark.asyncio
async def test_public_factory_projects_directory_content_and_structured_helpers(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    bundle = _open_bundle(tmp_path, clock)
    artifact_ids = iter(f"artifact-{index}" for index in range(1, 7))
    occurrence_ids = iter(f"occurrence-{index}" for index in range(1, 7))
    factory = CanonicalArtifactFacadeFactory(
        bundle=bundle,
        owner_scope=_owner_scope(),
        clock=clock.now,
        artifact_id_factory=lambda: next(artifact_ids),
        occurrence_id_factory=lambda: next(occurrence_ids),
    )
    artifacts = factory.for_public_execution(
        StorageScope(graph_id="graph-1", node_id="node-1"),
        deprecated_app_id="app-legacy",
    )
    source_dir = tmp_path / "source-directory"
    source_dir.mkdir()
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "value.txt").write_text("directory value", encoding="utf-8")
    binary_path = tmp_path / "payload.bin"
    binary_path.write_bytes(b"\x00\x01binary")
    staged_path = tmp_path / "staged.txt"
    staged_path.write_text("consumed staging", encoding="utf-8")
    try:
        directory = await artifacts.save_directory(
            str(source_dir),
            name="bundle.tar",
            tags=["final"],
        )
        structured = await artifacts.save_json({"status": "ok"}, name="result.json")
        high = await artifacts.save_text(
            "high quality",
            kind="report",
            tags=["ranked"],
            metrics={"quality": 0.9},
        )
        low = await artifacts.save_text(
            "low quality",
            kind="report",
            tags=["ranked"],
            metrics={"quality": 0.2},
        )
        binary = await artifacts.save_file(
            str(binary_path),
            kind="binary",
            name="payload.bin",
            cleanup=False,
        )
        with pytest.warns(DeprecationWarning, match="save_file"):
            ingested = await artifacts.ingest_file(staged_path.as_posix(), kind="note")

        assert source_dir.is_dir()
        assert not staged_path.exists()
        assert directory.labels["filename"] == "bundle.tar"
        assert directory.app_id == "app-legacy"
        assert ingested.kind == "note"

        destination = tmp_path / "materialized"
        assert await artifacts.materialize_directory(
            directory.artifact_id,
            str(destination),
        ) == str(destination.resolve())
        assert (destination / "nested" / "value.txt").read_text(encoding="utf-8") == (
            "directory value"
        )
        transient_dir = Path(await artifacts.as_local_dir_by_id(directory.artifact_id))
        assert (transient_dir / "nested" / "value.txt").read_text(encoding="utf-8") == (
            "directory value"
        )

        local_binary = Path(await artifacts.as_local_file_by_id(binary.artifact_id))
        assert local_binary.read_bytes() == b"\x00\x01binary"
        assert (await artifacts.load_content(structured.artifact_id)).json == {"status": "ok"}
        assert (await artifacts.load_content(high.artifact_id)).text == "high quality"
        binary_content = await artifacts.load_content(binary.artifact_id)
        assert binary_content.mode == "bytes"
        assert binary_content.data == b"\x00\x01binary"

        page = await artifacts.query_public_artifacts(
            PageRequest(limit=2),
            kind="report",
            tags=["ranked"],
        )
        assert len(page.items) == 2
        assert all(item.app_id == "app-legacy" for item in page.items)
        assert {item.artifact_id for item in await artifacts.list(kind="report")} == {
            high.artifact_id,
            low.artifact_id,
        }
        assert (
            await artifacts.best(
                kind="report",
                metric="quality",
                metric_mode="max",
                tags=["ranked"],
            )
            == high
        )
        assert (
            await artifacts.best(
                kind="report",
                metric="quality",
                metric_mode="min",
                tags=["ranked"],
            )
            == low
        )
        with pytest.raises(ValueError, match="500"):
            await artifacts.query_public_artifacts(PageRequest(limit=501))
    finally:
        await bundle.close()


def test_factory_public_docstrings_follow_service_contract() -> None:
    for name in ("__init__", "for_execution", "for_owner", "for_public_execution"):
        docstring = inspect.getdoc(getattr(CanonicalArtifactFacadeFactory, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

    for member in (
        CanonicalPublicArtifactFacade.__init__,
        CanonicalPublicArtifactFacade.stage_path,
        CanonicalPublicArtifactFacade.stage_dir,
        CanonicalPublicArtifactFacade.save_directory,
        CanonicalPublicArtifactFacade.ingest_file,
        CanonicalPublicArtifactFacade.ingest_dir,
        CanonicalPublicArtifactFacade.writer,
        CanonicalPublicArtifactFacade.save_file,
        CanonicalPublicArtifactFacade.save_text,
        CanonicalPublicArtifactFacade.save_json,
        CanonicalPublicArtifactFacade.get_by_id,
        CanonicalPublicArtifactFacade.load_bytes_by_id,
        CanonicalPublicArtifactFacade.load_text_by_id,
        CanonicalPublicArtifactFacade.load_json_by_id,
        CanonicalPublicArtifactFacade.load_content,
        CanonicalPublicArtifactFacade.as_local_file_by_id,
        CanonicalPublicArtifactFacade.materialize_directory,
        CanonicalPublicArtifactFacade.as_local_dir_by_id,
        CanonicalPublicArtifactFacade.load_bytes,
        CanonicalPublicArtifactFacade.load_text,
        CanonicalPublicArtifactFacade.load_json,
        CanonicalPublicArtifactFacade.query_public_artifacts,
        CanonicalPublicArtifactFacade.list,
        CanonicalPublicArtifactFacade.best,
        CanonicalPublicArtifactFacade.search_public_artifacts,
        CanonicalPublicArtifactFacade.pin,
    ):
        docstring = inspect.getdoc(member) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

    source = inspect.getsource(CanonicalPublicArtifactFacade)
    assert "except Exception" not in source
    assert "ScopedIndices" not in source
    assert "suggested_uri" in source
    assert "deprecated_app_id" in inspect.signature(CanonicalPublicArtifactFacade).parameters
    assert "app_id" not in inspect.signature(CanonicalPublicArtifactFacade).parameters


def test_public_facade_retires_locator_and_ambiguous_legacy_helpers() -> None:
    for name in (
        "as_local_file",
        "as_local_dir",
        "to_local_path",
        "to_local_file",
        "to_local_dir",
        "load_artifact",
        "load_artifact_bytes",
        "load_artifact_dir",
        "search",
        "stage",
        "ingest",
        "save",
        "tmp_path",
    ):
        assert not hasattr(CanonicalPublicArtifactFacade, name)

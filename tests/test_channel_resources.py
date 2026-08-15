from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.channel.resources import (
    ArtifactIngressScope,
    InputResource,
    InputResourceNormalizer,
    ResourceSet,
    ResourceStager,
)
from aethergraph.services.scope.scope_factory import ScopeFactory


def test_normalizer_extracts_urls_and_trusted_local_paths(tmp_path: Path) -> None:
    local_file = tmp_path / "input.txt"
    local_file.write_text("hello", encoding="utf-8")

    resources = InputResourceNormalizer().from_text(
        f"read https://example.com/data.csv and {local_file}",
        source="webui",
    )
    payloads = resources.to_dicts()

    assert any(
        item["kind"] == "url" and item["url"] == "https://example.com/data.csv" for item in payloads
    )
    path_resource = next(item for item in payloads if item.get("path") == str(local_file))
    assert path_resource["kind"] == "file_path"
    assert path_resource["meta"]["materialization_allowed"] is True


def test_untrusted_local_paths_are_inert_candidates(tmp_path: Path) -> None:
    local_file = tmp_path / "secret.txt"
    local_file.write_text("secret", encoding="utf-8")

    resources = InputResourceNormalizer().from_text(str(local_file), source="slack")
    payload = resources.to_dicts()[0]

    assert payload["kind"] == "file_path"
    assert payload["status"] == "candidate"
    assert payload["meta"]["materialization_allowed"] is False


def test_artifact_content_url_normalizes_to_artifact_resource() -> None:
    resource = InputResource.from_dict(
        {
            "url": "/api/v1/artifacts/art-1/content",
            "mimetype": "text/plain",
            "name": "notes.txt",
        },
        source="webui",
    )
    payload = resource.to_dict()

    assert payload["kind"] == "artifact"
    assert payload["artifact_id"] == "art-1"
    assert payload["mime"] == "text/plain"
    assert payload["mimetype"] == "text/plain"


def test_public_artifact_projects_through_one_typed_resource_path() -> None:
    artifact = Artifact(
        artifact_id="artifact-typed",
        kind="report",
        bytes=0,
        mime="text/markdown",
        labels={"filename": "report.md", "category": "evidence"},
        uri="/api/v1/artifacts/artifact-typed/content",
    )

    resource = InputResourceNormalizer().from_artifact(artifact, source="agent")

    assert resource.artifact_id == "artifact-typed"
    assert resource.name == "report.md"
    assert resource.mime == "text/markdown"
    assert resource.size == 0
    assert resource.uri == artifact.uri
    assert resource.url == artifact.uri
    assert resource.labels == artifact.labels
    with pytest.raises(TypeError, match="public Artifact DTO"):
        InputResourceNormalizer().from_artifact(object())  # type: ignore[arg-type]


def test_text_normalizer_extracts_artifact_refs_before_paths() -> None:
    resources = InputResourceNormalizer().from_text(
        "use /api/v1/artifacts/art-2/content and artifact://art-3.py",
        source="webui",
    )
    payloads = resources.to_dicts()

    assert {item["kind"] for item in payloads} == {"artifact", "artifact_uri"}
    assert any(item.get("artifact_id") == "art-2" for item in payloads)
    assert any(item.get("uri") == "artifact://art-3.py" for item in payloads)
    assert all(item.get("path") is None for item in payloads)


def test_resource_set_dedupe_prefers_materialized_artifact() -> None:
    resources = ResourceSet(
        [
            InputResource(
                kind="url",
                source="webui",
                status="candidate",
                url="/api/v1/artifacts/art-1/content",
            ),
            InputResource(
                kind="artifact",
                source="webui",
                status="materialized",
                artifact_id="art-1",
                url="/api/v1/artifacts/art-1/content",
                uri="artifact://art-1",
            ),
        ]
    )

    payloads = resources.to_dicts()

    assert len(payloads) == 1
    assert payloads[0]["kind"] == "artifact"
    assert payloads[0]["artifact_id"] == "art-1"
    assert payloads[0]["uri"] == "artifact://art-1"


class _FakeArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.saved: dict[str, Any] | None = None

    async def plan_staging_path(self, planned_ext: str = "") -> str:
        path = self.root / f"stage{planned_ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    async def save_file(self, **kwargs: Any) -> Artifact:
        self.saved = dict(kwargs)
        path = Path(kwargs["path"])
        return Artifact(
            artifact_id="artifact-1",
            uri="artifact://artifact-1",
            kind=kwargs["kind"],
            bytes=path.stat().st_size,
            mime=kwargs.get("mime"),
            run_id=kwargs.get("run_id"),
            graph_id=kwargs.get("graph_id"),
            node_id=kwargs.get("node_id"),
            tool_name=kwargs.get("tool_name"),
            tool_version=kwargs.get("tool_version"),
            labels=dict(kwargs.get("labels") or {}),
        )


class _FakeArtifactIndex:
    def __init__(self) -> None:
        self.upserts: list[Artifact] = []
        self.occurrences: list[Artifact] = []

    async def upsert(self, artifact: Artifact) -> None:
        self.upserts.append(artifact)

    async def record_occurrence(self, artifact: Artifact) -> None:
        self.occurrences.append(artifact)

    async def get(self, artifact_id: str) -> Artifact | None:
        for artifact in self.upserts:
            if artifact.artifact_id == artifact_id:
                return artifact
        return None


class _FakeContainer:
    def __init__(self, root: Path) -> None:
        self.artifacts = _FakeArtifactStore(root)
        self.artifact_index = _FakeArtifactIndex()
        self.scope_factory = ScopeFactory()
        self.scoped_indices = None


@pytest.mark.asyncio
async def test_resource_stager_uses_facade_and_preserves_channel_scope(tmp_path: Path) -> None:
    container = _FakeContainer(tmp_path)
    stager = ResourceStager(container=container)

    resource = await stager.stage_bytes(
        b"hello",
        name="hello.txt",
        mime="text/plain",
        file_id="platform-file-1",
        scope=ArtifactIngressScope(
            source="slack",
            channel_key="slack:team/T:chan/C",
            conversation_id="slack:thread",
        ),
    )

    assert resource.artifact_id == "artifact-1"
    assert resource.id == "platform-file-1"
    assert resource.uri == "artifact://artifact-1"
    assert resource.labels["scope_id"] == "channel:slack:team/T:chan/C"
    assert resource.labels["channel_key"] == "slack:team/T:chan/C"
    assert "run_id" not in resource.labels
    assert container.artifacts.saved is not None
    assert container.artifacts.saved["run_id"] == ""
    assert len(container.artifact_index.upserts) == 1

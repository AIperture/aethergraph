from __future__ import annotations

from datetime import UTC, datetime
import inspect

import pytest

from aethergraph.services.artifacts import project_public_artifact
from aethergraph.storage.contracts import (
    ArtifactAction,
    ArtifactOccurrence,
    ArtifactRecord,
    ArtifactRetentionRecord,
    StorageIntegrityError,
    StorageScope,
)

NOW = datetime(2026, 8, 16, 16, tzinfo=UTC)


def _owner() -> StorageScope:
    return StorageScope(
        tenant_id="tenant-1",
        project_id="project-1",
        org_id="org-1",
        user_id="user-1",
    )


def _execution() -> StorageScope:
    return StorageScope(
        **_owner().as_filter(),
        session_id="session-1",
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
        agent_id="agent-1",
    )


def _record(*, hash_algorithm: str = "sha256") -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id="artifact/1",
        content_hash="a" * 64,
        hash_algorithm=hash_algorithm,
        size_bytes=12,
        media_type="text/plain",
        kind="report",
        blob_locator="blob:opaque-provider-value",
        owner_scope=_owner(),
        created_at=NOW,
        original_filename="report.txt",
        provider_version="provider-version",
        labels={"tags": ["final", "report"], "nested": {"reviewed": True}},
    )


def _occurrence() -> ArtifactOccurrence:
    return ArtifactOccurrence(
        occurrence_id="occurrence-1",
        artifact_id="artifact/1",
        scope=_execution(),
        action=ArtifactAction.PRODUCED,
        occurred_at=NOW,
        tool_name="reporter",
        tool_version="1.0",
        labels={"stage": "final"},
        metrics={"quality": 0.9},
    )


def test_public_artifact_projection_preserves_frozen_shape_without_provider_locator() -> None:
    retention = ArtifactRetentionRecord(
        artifact_id="artifact/1",
        scope=_owner(),
        pinned=True,
        revision=1,
        updated_at=NOW,
    )

    artifact = project_public_artifact(
        _record(),
        occurrence=_occurrence(),
        retention=retention,
        deprecated_app_id="app-compatibility-only",
    )

    assert artifact.uri == "/api/v1/artifacts/artifact%2F1/content"
    assert "opaque-provider-value" not in str(artifact.to_dict())
    assert artifact.run_id == "run-1"
    assert artifact.session_id == "session-1"
    assert artifact.graph_id == "graph-1"
    assert artifact.node_id == "node-1"
    assert artifact.tool_name == "reporter"
    assert artifact.sha256 == "a" * 64
    assert artifact.mime == "text/plain"
    assert artifact.mimetype == "text/plain"
    assert artifact.bytes == 12
    assert artifact.tags == ["final", "report"]
    assert artifact.labels == {
        "tags": ["final", "report"],
        "nested": {"reviewed": True},
        "stage": "final",
        "filename": "report.txt",
    }
    assert artifact.metrics == {"quality": 0.9}
    assert artifact.pinned is True
    assert artifact.org_id == "org-1"
    assert artifact.user_id == "user-1"
    assert artifact.client_id is None
    assert artifact.app_id == "app-compatibility-only"
    assert artifact.occurrence_id == "occurrence-1"
    assert artifact.to_dict()["mime"] == artifact.to_dict()["mimetype"]


def test_public_artifact_projection_does_not_relabel_hash_or_infer_app_identity() -> None:
    artifact = project_public_artifact(_record(hash_algorithm="sha512"))

    assert artifact.sha256 is None
    assert artifact.app_id is None
    assert artifact.client_id is None
    assert artifact.run_id is None
    assert artifact.occurrence_id is None
    assert artifact.created_at == NOW.isoformat()


def test_public_artifact_projection_rejects_cross_owner_components() -> None:
    occurrence = ArtifactOccurrence(
        occurrence_id="occurrence-cross-owner",
        artifact_id="artifact/1",
        scope=StorageScope(tenant_id="tenant-2", project_id="project-1"),
        action=ArtifactAction.PRODUCED,
        occurred_at=NOW,
    )
    retention = ArtifactRetentionRecord(
        artifact_id="artifact/1",
        scope=StorageScope(tenant_id="tenant-2", project_id="project-1"),
        pinned=True,
        revision=1,
        updated_at=NOW,
    )

    with pytest.raises(StorageIntegrityError, match="occurrence crosses"):
        project_public_artifact(_record(), occurrence=occurrence)
    with pytest.raises(StorageIntegrityError, match="retention crosses"):
        project_public_artifact(_record(), retention=retention)


def test_public_artifact_projection_marks_app_id_as_explicit_optional_metadata() -> None:
    with pytest.raises(ValueError, match="deprecated_app_id"):
        project_public_artifact(_record(), deprecated_app_id="")

    docstring = inspect.getdoc(project_public_artifact) or ""
    assert docstring.index("Examples:") < docstring.index("Args:")
    assert docstring.index("Args:") < docstring.index("Returns:")
    assert docstring.index("Returns:") < docstring.index("Notes:")
    assert docstring.count("```python") >= 2
    assert "deprecated" in docstring.lower()

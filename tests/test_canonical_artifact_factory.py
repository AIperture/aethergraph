from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.services.artifacts import CanonicalArtifactFacadeFactory
from aethergraph.storage.contracts import (
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


def test_factory_public_docstrings_follow_service_contract() -> None:
    for name in ("__init__", "for_execution", "for_owner"):
        docstring = inspect.getdoc(getattr(CanonicalArtifactFacadeFactory, name)) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path

import pytest

from aethergraph.services.memory import CanonicalMemoryFacade, CanonicalMemoryFacadeFactory
from aethergraph.storage.contracts import (
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = "secret://tests/memory-factory"
_SECRET = b"canonical-memory-factory-secret-32"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 10, tzinfo=UTC)

    def now(self) -> datetime:
        value = self.value
        self.value += timedelta(microseconds=1)
        return value


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider must not resolve {reference!r}")


def _open_bundle(root: Path):
    return LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    ).open(
        StorageOpenRequest(
            workspace_id="memory-factory-tests",
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


@pytest.mark.asyncio
async def test_memory_factory_binds_exact_bundle_and_merged_scope(tmp_path: Path) -> None:
    bundle = _open_bundle(tmp_path)
    factory = CanonicalMemoryFacadeFactory(
        bundle=bundle,
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        hot_max_events=3,
        hot_ttl_seconds=10,
    )
    try:
        memory = factory.for_execution(
            StorageScope(session_id="session-1", run_id="run-1", agent_id="writer")
        )

        assert isinstance(memory, CanonicalMemoryFacade)
        assert memory.scope == StorageScope(
            tenant_id="tenant-1",
            project_id="project-1",
            session_id="session-1",
            run_id="run-1",
            agent_id="writer",
        )
        committed = await memory.commit_state(
            key="draft",
            value={"value": 1},
            expected_revision=0,
        )
        assert (
            await bundle.state.get(
                memory.scope,
                "memory.state.state.snapshot",
                "draft",
            )
            == committed
        )
    finally:
        await bundle.close()


def test_memory_factory_rejects_owner_conflicts_and_performs_no_lifecycle_probe() -> None:
    class _Bundle:
        def __getattribute__(self, name: str):
            if name in {"open", "health", "close"}:
                raise AssertionError(f"factory must not access lifecycle member {name}")
            return object.__getattribute__(self, name)

    bundle = _Bundle()
    bundle.memory_events = object()
    bundle.state = object()
    bundle.search = object()
    factory = CanonicalMemoryFacadeFactory(
        bundle=bundle,  # type: ignore[arg-type]
        owner_scope=StorageScope(project_id="project-1"),
    )

    with pytest.raises(ValueError, match="project_id"):
        factory.for_execution(StorageScope(project_id="project-2", run_id="run-1"))
    with pytest.raises(ValueError, match="owner_scope"):
        CanonicalMemoryFacadeFactory(
            bundle=bundle,  # type: ignore[arg-type]
            owner_scope=StorageScope(project_id="project-1", run_id="run-1"),
        )
    with pytest.raises(ValueError, match="10000"):
        CanonicalMemoryFacadeFactory(
            bundle=bundle,  # type: ignore[arg-type]
            owner_scope=StorageScope(project_id="project-1"),
            hot_max_events=10_001,
        )

    source = inspect.getsource(CanonicalMemoryFacadeFactory)
    assert "build_" not in source
    assert "StorageProvider" not in source
    assert "app_id" not in inspect.signature(factory.for_execution).parameters
    for member in (
        CanonicalMemoryFacadeFactory.__init__,
        CanonicalMemoryFacadeFactory.for_execution,
    ):
        docstring = inspect.getdoc(member) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2

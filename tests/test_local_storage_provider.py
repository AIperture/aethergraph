from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
import inspect
from pathlib import Path
import sqlite3
from typing import get_type_hints

import pytest

from aethergraph.storage.composition import StorageComposition
from aethergraph.storage.contracts import (
    RuntimeOutputFrame,
    RuntimeOutputStream,
    StorageBundle,
    StorageCapability,
    StorageConfigurationError,
    StorageHealthError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageReadOnlyError,
    StorageScope,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry
from aethergraph.storage.providers.local_sqlite import (
    LocalDatabaseRole,
    LocalStorageBundle,
    LocalStorageProvider,
    provider as provider_module,
    read_local_workspace_manifest,
)

SECRET_REF = "secret://storage/continuations"
SECRET = b"local-provider-continuation-secret-32-bytes"


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 16, 5, tzinfo=UTC)

    def now(self) -> datetime:
        current = self.value
        self.value += timedelta(microseconds=1)
        return current


class _Secrets:
    async def resolve(self, reference: str) -> str | bytes:
        raise AssertionError(f"provider open must not resolve {reference!r}")


class _Embedder:
    async def embed_result(self, texts, **kwargs):  # pragma: no cover - not invoked
        raise AssertionError("not invoked")

    async def embed(self, texts, **kwargs):  # pragma: no cover - not invoked
        return [[1.0, 0.0] for _text in texts]

    async def embed_one(self, text, **kwargs):  # pragma: no cover - not invoked
        return [1.0, 0.0]


def _provider(*, embedder=None) -> LocalStorageProvider:
    return LocalStorageProvider(
        continuation_token_secret_ref=SECRET_REF,
        continuation_token_secret=SECRET,
        embedder=embedder,
    )


def _request(
    root: Path,
    *,
    mode: StorageOpenMode = StorageOpenMode.READ_WRITE,
    config: dict[str, object] | None = None,
    clock: _Clock | None = None,
) -> StorageOpenRequest:
    options = {"continuation_token_secret_ref": SECRET_REF}
    if config is not None:
        options.update(config)
    return StorageOpenRequest(
        workspace_id="workspace-1",
        workspace_root=root.resolve(),
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        selection=StorageProviderSelection(provider="local.sqlite", config=options),
        mode=mode,
        expected_format_version=1,
        clock=clock or _Clock(),
        secrets=_Secrets(),
    )


def _frame() -> RuntimeOutputFrame:
    return RuntimeOutputFrame(
        output_id="output-1",
        execution_id="execution-1",
        scope=StorageScope(
            tenant_id="tenant-1",
            project_id="project-1",
            session_id="session-1",
            run_id="run-1",
            node_id="node-1",
        ),
        stream=RuntimeOutputStream.STDOUT,
        sequence=1,
        text="hello",
        source="python",
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"fallback": "legacy"}, "Unknown local storage options"),
        ({"durability": "unsafe"}, "durability"),
        ({"busy_timeout_ms": 0}, "busy_timeout_ms"),
        ({"runtime_output_max_pending_frames": True}, "runtime_output"),
        ({"search_max_candidates": 100}, "search_max_candidates"),
        ({"continuation_token_secret_ref": " other "}, "secret_ref"),
    ],
)
def test_local_provider_config_is_exact_and_typed(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(StorageConfigurationError, match=message):
        _provider().validate_config(_request(tmp_path, config=changes).selection)

    mismatched = LocalStorageProvider(
        continuation_token_secret_ref="secret://different",
        continuation_token_secret=SECRET,
    )
    with pytest.raises(StorageConfigurationError, match="does not match"):
        mismatched.validate_config(_request(tmp_path).selection)


@pytest.mark.asyncio
async def test_local_provider_opens_one_coherent_bundle_and_closes_cleanly(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    bundle = _provider().open(request)

    assert isinstance(bundle, LocalStorageBundle)
    assert bundle.provider_name == "local.sqlite"
    assert bundle.format_version == 1
    assert bundle.mode is StorageOpenMode.READ_WRITE
    assert (await bundle.health()).ready is True
    for field_name in get_type_hints(StorageBundle):
        assert hasattr(bundle, field_name), field_name
    assert bundle.auth_grants is bundle.kv
    assert bundle.auth_invites is bundle.kv
    assert bundle.registry_manifests is bundle.documents
    assert {database.role for database in bundle._databases} == set(LocalDatabaseRole)
    with pytest.raises(FrozenInstanceError):
        bundle.mode = StorageOpenMode.READ_ONLY  # type: ignore[misc]

    manifest_text = (tmp_path / "workspace.json").read_text(encoding="utf-8")
    assert SECRET_REF not in manifest_text
    assert SECRET.decode() not in manifest_text
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is False

    maintenance = await bundle.checkpoint()
    assert set(maintenance) == set(LocalDatabaseRole)
    assert read_local_workspace_manifest(tmp_path).last_maintenance_at is not None

    await bundle.close()
    await bundle.close()

    manifest = read_local_workspace_manifest(tmp_path)
    assert manifest.clean_shutdown is True
    assert manifest.last_maintenance_at is not None
    assert (await bundle.health()).ready is False


@pytest.mark.asyncio
async def test_registry_composition_opens_and_owns_the_exact_local_bundle(tmp_path: Path) -> None:
    registry = StorageProviderRegistry({"local.sqlite": _provider})
    composition = StorageComposition(
        registry,
        frozenset(
            {
                StorageCapability.DURABLE,
                StorageCapability.TRANSACTIONS,
                StorageCapability.ATOMIC_COMPARE_AND_SET,
                StorageCapability.ORDERED_APPEND,
                StorageCapability.MONOTONIC_CURSORS,
                StorageCapability.BLOB_STREAMING,
                StorageCapability.SEARCH_LEXICAL,
            }
        ),
    )

    bundle = await composition.open(_request(tmp_path))

    assert isinstance(bundle, LocalStorageBundle)
    assert (await composition.health()).ready is True
    await composition.close()
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is True


@pytest.mark.asyncio
async def test_bundle_close_flushes_runtime_output_before_clean_manifest(tmp_path: Path) -> None:
    bundle = _provider().open(_request(tmp_path))
    bundle.runtime_output.emit(_frame())

    await bundle.close()

    connection = sqlite3.connect(tmp_path / "local" / "events.sqlite3")
    try:
        count = connection.execute("SELECT COUNT(*) FROM local_runtime_output").fetchone()[0]
    finally:
        connection.close()
    assert count == 1
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is True


@pytest.mark.asyncio
async def test_failed_output_barrier_keeps_local_close_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _provider().open(_request(tmp_path))
    original_flush = bundle.runtime_output._flush_all
    attempts = 0

    async def flaky_flush() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise StorageHealthError("injected output flush failure")
        await original_flush()

    monkeypatch.setattr(bundle.runtime_output, "_flush_all", flaky_flush)

    with pytest.raises(StorageHealthError, match="injected output flush failure"):
        await bundle.close()

    assert (await bundle.health()).ready is True
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is False

    await bundle.close()

    assert attempts == 2
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is True


@pytest.mark.asyncio
async def test_failed_manifest_commit_retries_without_reopening_databases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _provider().open(_request(tmp_path))
    original_update = provider_module.update_local_workspace_lifecycle
    clean_attempts = 0

    def flaky_update(root: Path, *, clean_shutdown: bool, last_maintenance_at):
        nonlocal clean_attempts
        if clean_shutdown:
            clean_attempts += 1
            if clean_attempts == 1:
                raise StorageHealthError("injected manifest commit failure")
        return original_update(
            root,
            clean_shutdown=clean_shutdown,
            last_maintenance_at=last_maintenance_at,
        )

    monkeypatch.setattr(provider_module, "update_local_workspace_lifecycle", flaky_update)

    with pytest.raises(StorageHealthError, match="injected manifest commit failure"):
        await bundle.close()

    assert read_local_workspace_manifest(tmp_path).clean_shutdown is False
    assert (await bundle.health()).detail == "closed"

    await bundle.close()

    assert clean_attempts == 2
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is True


@pytest.mark.asyncio
async def test_read_only_historical_open_preserves_manifest_and_rejects_maintenance(
    tmp_path: Path,
) -> None:
    writable = _provider().open(_request(tmp_path))
    await writable.close()
    before = (tmp_path / "workspace.json").read_bytes()

    readonly = _provider().open(_request(tmp_path, mode=StorageOpenMode.READ_ONLY))

    assert (await readonly.health()).ready is True
    with pytest.raises(StorageReadOnlyError):
        readonly.runtime_output.emit(_frame())
    with pytest.raises(StorageReadOnlyError):
        await readonly.checkpoint()
    await readonly.close()

    assert (tmp_path / "workspace.json").read_bytes() == before


@pytest.mark.asyncio
async def test_unclean_workspace_reopens_without_legacy_recovery(tmp_path: Path) -> None:
    interrupted = _provider().open(_request(tmp_path))
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is False
    for database in reversed(interrupted._databases):
        database._close_during_open_failure()

    reopened = _provider().open(_request(tmp_path))

    assert (await reopened.health()).ready is True
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is False
    await reopened.close()
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is True


def test_partial_bundle_construction_closes_every_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingSearch:
        def __init__(self, **kwargs) -> None:
            raise StorageHealthError("injected search construction failure")

    monkeypatch.setattr(provider_module, "LocalSearchBackend", _FailingSearch)

    with pytest.raises(StorageHealthError, match="search construction"):
        _provider().open(_request(tmp_path))

    database_files = tuple((tmp_path / "local").glob("*.sqlite3"))
    assert {path.name for path in database_files} == {
        "control.sqlite3",
        "events.sqlite3",
        "search.sqlite3",
    }
    for path in database_files:
        path.unlink()
    assert read_local_workspace_manifest(tmp_path).clean_shutdown is False


def test_semantic_capabilities_require_an_injected_embedder(tmp_path: Path) -> None:
    lexical = _provider().open(_request(tmp_path / "lexical"))
    semantic = _provider(embedder=_Embedder()).open(_request(tmp_path / "semantic"))

    assert lexical.capabilities.supports(StorageCapability.SEARCH_LEXICAL)
    assert not lexical.capabilities.supports(StorageCapability.SEARCH_SEMANTIC)
    assert not lexical.capabilities.supports(StorageCapability.SEARCH_HYBRID)
    assert semantic.capabilities.supports(StorageCapability.SEARCH_SEMANTIC)
    assert semantic.capabilities.supports(StorageCapability.SEARCH_HYBRID)

    for bundle in (lexical, semantic):
        for database in reversed(bundle._databases):
            database._close_during_open_failure()


def test_local_provider_public_docstrings_follow_required_format() -> None:
    methods = (
        LocalStorageProvider.__init__,
        LocalStorageProvider.validate_config,
        LocalStorageProvider.open,
        LocalStorageBundle.health,
        LocalStorageBundle.checkpoint,
        LocalStorageBundle.close,
    )
    for method in methods:
        docstring = inspect.getdoc(method) or ""
        assert docstring.index("Examples:") < docstring.index("Args:")
        assert docstring.index("Args:") < docstring.index("Returns:")
        assert docstring.index("Returns:") < docstring.index("Notes:")
        assert docstring.count("```python") >= 2


def test_local_provider_source_has_no_compatibility_routing_or_host_dependency() -> None:
    source = inspect.getsource(provider_module)
    forbidden = (
        "app_id",
        "application_id",
        "client_id",
        "aethergraph_engine",
        "ag_studio",
        "services.integration",
        "open_legacy",
        "try_open",
        "dual_read",
        "dual_write",
    )
    assert all(term not in source for term in forbidden)

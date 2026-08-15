from __future__ import annotations

from datetime import UTC, datetime
import inspect
from pathlib import Path

import pytest

from aethergraph.storage.composition import StorageComposition
from aethergraph.storage.contracts import (
    StorageBundle,
    StorageCapabilities,
    StorageCapability,
    StorageCapabilityError,
    StorageConfigurationError,
    StorageConflictError,
    StorageFormatError,
    StorageHealth,
    StorageHealthError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProvider,
    StorageProviderSelection,
    StorageScope,
    UnknownStorageProviderError,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 15, 12, tzinfo=UTC)


class _Secrets:
    async def resolve(self, reference: str) -> str:
        return f"resolved:{reference}"


class _Bundle:
    def __init__(
        self,
        *,
        provider_name: str = "company.external",
        mode: StorageOpenMode = StorageOpenMode.READ_WRITE,
        format_version: int = 1,
        capabilities: StorageCapabilities | None = None,
        ready: bool = True,
    ) -> None:
        self.provider_name = provider_name
        self.mode = mode
        self.format_version = format_version
        self.capabilities = capabilities or StorageCapabilities.of(
            StorageCapability.DURABLE,
            StorageCapability.HEALTH,
        )
        self.ready = ready
        self.close_calls = 0

    async def health(self) -> StorageHealth:
        return StorageHealth(ready=self.ready, detail="unavailable" if not self.ready else "ready")

    async def close(self) -> None:
        self.close_calls += 1


class _Provider:
    name = "company.external"

    def __init__(self, bundle: _Bundle) -> None:
        self.bundle = bundle
        self.open_calls = 0

    def validate_config(self, selection: StorageProviderSelection) -> None:
        if selection.provider != self.name or set(selection.config) != {"endpoint"}:
            raise StorageConfigurationError("invalid external selection")

    def open(self, request: StorageOpenRequest) -> StorageBundle:
        self.open_calls += 1
        return self.bundle  # type: ignore[return-value]


class _RetryableCloseBundle(_Bundle):
    def __init__(self) -> None:
        super().__init__()
        self.remaining_failures = 1

    async def close(self) -> None:
        self.close_calls += 1
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise StorageHealthError("durable flush failed")


def _request(tmp_path: Path) -> StorageOpenRequest:
    return StorageOpenRequest(
        workspace_id="workspace-1",
        workspace_root=tmp_path.resolve(),
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        selection=StorageProviderSelection(
            provider="company.external",
            config={"endpoint": "postgresql://storage"},
        ),
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )


def _composition(
    bundle: _Bundle,
    *,
    required: frozenset[StorageCapability] = frozenset({StorageCapability.DURABLE}),
) -> tuple[StorageComposition, _Provider]:
    provider = _Provider(bundle)
    registry = StorageProviderRegistry({provider.name: lambda: provider})
    return StorageComposition(registry, required), provider


def test_provider_construction_is_synchronous_while_bundle_lifecycle_is_async() -> None:
    assert inspect.iscoroutinefunction(StorageProvider.open) is False
    assert inspect.iscoroutinefunction(StorageBundle.health) is True
    assert inspect.iscoroutinefunction(StorageBundle.close) is True


def test_composition_rejects_non_capability_requirements() -> None:
    with pytest.raises(TypeError, match="StorageCapability"):
        StorageComposition(
            StorageProviderRegistry(),
            frozenset({"durable"}),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_composition_opens_validates_and_closes_one_exact_bundle(tmp_path: Path) -> None:
    composition, provider = _composition(_Bundle())

    bundle = await composition.open(_request(tmp_path))

    assert provider.open_calls == 1
    assert bundle is provider.bundle
    assert (await composition.health()).ready is True

    await composition.close()
    await composition.close()

    assert provider.bundle.close_calls == 1
    with pytest.raises(StorageHealthError, match="no active bundle"):
        await composition.health()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bundle", "error"),
    [
        (_Bundle(provider_name="wrong"), StorageConfigurationError),
        (_Bundle(mode=StorageOpenMode.READ_ONLY), StorageConfigurationError),
        (_Bundle(format_version=2), StorageFormatError),
        (
            _Bundle(capabilities=StorageCapabilities.of(StorageCapability.HEALTH)),
            StorageCapabilityError,
        ),
        (_Bundle(ready=False), StorageHealthError),
    ],
)
async def test_composition_closes_partial_bundle_on_every_post_open_failure(
    tmp_path: Path,
    bundle: _Bundle,
    error: type[Exception],
) -> None:
    composition, provider = _composition(bundle)

    with pytest.raises(error):
        await composition.open(_request(tmp_path))

    assert provider.open_calls == 1
    assert bundle.close_calls == 1
    with pytest.raises(StorageHealthError, match="already closed"):
        await composition.open(_request(tmp_path))
    assert provider.open_calls == 1


@pytest.mark.asyncio
async def test_composition_is_single_open_and_has_no_reselection(tmp_path: Path) -> None:
    composition, provider = _composition(_Bundle())
    request = _request(tmp_path)

    await composition.open(request)
    with pytest.raises(StorageConflictError, match="already owns"):
        await composition.open(request)

    assert provider.open_calls == 1
    await composition.close()


@pytest.mark.asyncio
async def test_failed_bundle_close_remains_retryable(tmp_path: Path) -> None:
    bundle = _RetryableCloseBundle()
    composition, _provider = _composition(bundle)
    await composition.open(_request(tmp_path))

    with pytest.raises(StorageHealthError, match="durable flush failed"):
        await composition.close()

    assert bundle.close_calls == 1
    assert (await composition.health()).ready is True

    await composition.close()

    assert bundle.close_calls == 2
    with pytest.raises(StorageHealthError, match="no active bundle"):
        await composition.health()


@pytest.mark.asyncio
async def test_unknown_external_selection_never_constructs_registered_local_provider(
    tmp_path: Path,
) -> None:
    local_factory_calls = 0

    def local_factory() -> _Provider:
        nonlocal local_factory_calls
        local_factory_calls += 1
        return _Provider(_Bundle())

    composition = StorageComposition(StorageProviderRegistry({"local.sqlite": local_factory}))

    with pytest.raises(UnknownStorageProviderError, match="company.external"):
        await composition.open(_request(tmp_path))

    assert local_factory_calls == 0
    with pytest.raises(StorageHealthError, match="already closed"):
        await composition.open(_request(tmp_path))

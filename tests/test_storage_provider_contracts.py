from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    DuplicateStorageProviderError,
    StorageCapabilities,
    StorageCapability,
    StorageCapabilityError,
    StorageConfigurationError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
    StorageScopeError,
    UnknownStorageProviderError,
)
from aethergraph.storage.provider_registry import StorageProviderRegistry


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class _Secrets:
    async def resolve(self, reference: str) -> str:
        return f"resolved:{reference}"


class _Provider:
    name = "local.sqlite"

    def validate_config(self, selection: StorageProviderSelection) -> None:
        return None

    async def open(self, request: StorageOpenRequest):
        raise NotImplementedError


def test_storage_scope_is_immutable_and_excludes_deprecated_identity_aliases() -> None:
    scope = StorageScope(tenant_id="tenant-1", project_id="project-1", scope_key="thread:1")

    assert scope.require("tenant_id", "project_id") is scope
    assert scope.as_filter() == {
        "tenant_id": "tenant-1",
        "project_id": "project-1",
        "scope_key": "thread:1",
    }
    assert "app_id" not in StorageScope.__dataclass_fields__
    assert "application_id" not in StorageScope.__dataclass_fields__
    assert "client_id" not in StorageScope.__dataclass_fields__
    with pytest.raises(FrozenInstanceError):
        scope.tenant_id = "other"  # type: ignore[misc]


def test_storage_scope_fails_closed_for_missing_unknown_or_blank_dimensions() -> None:
    with pytest.raises(StorageScopeError, match="tenant_id"):
        StorageScope().require("tenant_id")
    with pytest.raises(StorageScopeError, match="Unknown"):
        StorageScope().require("app_id")
    with pytest.raises(StorageScopeError, match="non-empty"):
        StorageScope(run_id="  ")


def test_capability_require_reports_all_missing_without_fallback() -> None:
    capabilities = StorageCapabilities.of(
        StorageCapability.DURABLE,
        StorageCapability.TRANSACTIONS,
    )

    assert capabilities.supports(StorageCapability.DURABLE)
    with pytest.raises(StorageCapabilityError) as failure:
        capabilities.require(
            "local.sqlite",
            frozenset(
                {
                    StorageCapability.ATOMIC_COMPARE_AND_SET,
                    StorageCapability.SEARCH_HYBRID,
                }
            ),
        )
    assert failure.value.missing == ("atomic_compare_and_set", "search_hybrid")


def test_provider_selection_copies_config_and_open_request_requires_absolute_root(
    tmp_path: Path,
) -> None:
    mutable = {"busy_timeout_ms": 5000}
    selection = StorageProviderSelection(provider="local.sqlite", config=mutable)
    mutable["busy_timeout_ms"] = 1

    assert selection.config["busy_timeout_ms"] == 5000
    with pytest.raises(TypeError):
        selection.config["new"] = True  # type: ignore[index]
    request = StorageOpenRequest(
        workspace_id="workspace-1",
        workspace_root=tmp_path.resolve(),
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        selection=selection,
        mode=StorageOpenMode.READ_WRITE,
        expected_format_version=1,
        clock=_Clock(),
        secrets=_Secrets(),
    )
    assert request.selection is selection
    with pytest.raises(ValueError, match="absolute"):
        StorageOpenRequest(
            workspace_id="workspace-1",
            workspace_root=Path("relative"),
            owner_scope=StorageScope(),
            selection=selection,
            mode=StorageOpenMode.READ_ONLY,
            expected_format_version=1,
            clock=_Clock(),
            secrets=_Secrets(),
        )


def test_registry_is_exact_rejects_duplicates_and_has_no_default_fallback() -> None:
    registry = StorageProviderRegistry({"local.sqlite": _Provider})

    assert registry.names() == ("local.sqlite",)
    assert registry.create("local.sqlite").name == "local.sqlite"
    with pytest.raises(DuplicateStorageProviderError):
        registry.register("local.sqlite", _Provider)
    with pytest.raises(UnknownStorageProviderError):
        registry.resolve("company.external")
    with pytest.raises(StorageConfigurationError):
        registry.register("Local SQLite", _Provider)


def test_registry_rejects_factory_name_mismatch() -> None:
    class WrongProvider(_Provider):
        name = "wrong"

    registry = StorageProviderRegistry({"local.sqlite": WrongProvider})

    with pytest.raises(StorageConfigurationError, match="returned provider"):
        registry.create("local.sqlite")

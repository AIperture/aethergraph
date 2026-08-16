from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from aethergraph.storage.contracts import (
    StorageFormatError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LOCAL_STORAGE_FORMAT_VERSION,
    open_local_workspace_manifest,
    read_local_workspace_manifest,
)

NOW = datetime(2026, 8, 15, 14, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW


class _Secrets:
    async def resolve(self, reference: str) -> str:
        return f"resolved:{reference}"


def _request(root: Path, *, mode: StorageOpenMode = StorageOpenMode.READ_WRITE):
    return StorageOpenRequest(
        workspace_id="workspace-1",
        workspace_root=root.resolve(),
        owner_scope=StorageScope(tenant_id="tenant-1", project_id="project-1"),
        selection=StorageProviderSelection(
            provider="local.sqlite",
            config={"durability": "normal", "credential_ref": "secret://local"},
        ),
        mode=mode,
        expected_format_version=LOCAL_STORAGE_FORMAT_VERSION,
        clock=_Clock(),
        secrets=_Secrets(),
    )


def test_manifest_initializes_only_empty_workspace_without_persisting_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"

    manifest = open_local_workspace_manifest(_request(root))
    payload_text = (root / "workspace.json").read_text(encoding="utf-8")

    assert manifest.workspace_id == "workspace-1"
    assert manifest.owner_scope.project_id == "project-1"
    assert manifest.created_at == NOW
    assert manifest.clean_shutdown is False
    assert "secret://local" not in payload_text
    assert "durability" not in payload_text
    assert not list(root.glob(".workspace.json.*.tmp"))
    assert read_local_workspace_manifest(root) == manifest


def test_manifest_reopen_requires_identity_and_writable_config(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    request = _request(root)
    created = open_local_workspace_manifest(request)

    assert open_local_workspace_manifest(request) == created
    assert (
        open_local_workspace_manifest(replace(request, mode=StorageOpenMode.READ_ONLY)) == created
    )
    with pytest.raises(StorageFormatError, match="workspace_id"):
        open_local_workspace_manifest(replace(request, workspace_id="other"))
    with pytest.raises(StorageFormatError, match="owner scope"):
        open_local_workspace_manifest(replace(request, owner_scope=StorageScope(tenant_id="other")))

    with pytest.raises(StorageFormatError, match="requested workspace format"):
        open_local_workspace_manifest(replace(request, expected_format_version=2))
    with pytest.raises(StorageFormatError, match="fingerprint"):
        open_local_workspace_manifest(
            replace(
                request,
                selection=StorageProviderSelection(
                    provider="local.sqlite",
                    config={"durability": "full"},
                ),
            )
        )

    historical = replace(
        request,
        mode=StorageOpenMode.READ_ONLY,
        selection=StorageProviderSelection(
            provider="local.sqlite",
            config={"durability": "full"},
        ),
    )
    assert open_local_workspace_manifest(historical) == created


def test_unmanifested_read_only_or_nonempty_workspace_fails_without_mutation(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(StorageFormatError, match="requires an existing manifest"):
        open_local_workspace_manifest(_request(empty, mode=StorageOpenMode.READ_ONLY))
    assert list(empty.iterdir()) == []

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    legacy_database = legacy / "events.db"
    legacy_database.write_bytes(b"legacy")
    with pytest.raises(StorageFormatError, match="non-empty unmanifested"):
        open_local_workspace_manifest(_request(legacy))
    assert legacy_database.read_bytes() == b"legacy"
    assert not (legacy / "workspace.json").exists()

    invalid_root = tmp_path / "workspace-file"
    invalid_root.write_text("not a workspace", encoding="utf-8")
    with pytest.raises(StorageFormatError, match="must be a directory"):
        open_local_workspace_manifest(_request(invalid_root))
    assert invalid_root.read_text(encoding="utf-8") == "not a workspace"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 99),
        ("provider", "company.external"),
        ("created_at", "2026-08-15T14:00:00"),
    ],
)
def test_manifest_rejects_unsupported_or_malformed_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = tmp_path / field
    open_local_workspace_manifest(_request(root))
    path = root / "workspace.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StorageFormatError):
        read_local_workspace_manifest(root)


def test_manifest_rejects_unknown_fields_and_noncanonical_config(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    open_local_workspace_manifest(_request(root))
    path = root / "workspace.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["legacy_database"] = "events.db"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StorageFormatError, match="fields"):
        read_local_workspace_manifest(root)

    invalid = _request(tmp_path / "invalid")
    invalid = replace(
        invalid,
        selection=StorageProviderSelection(
            provider="local.sqlite",
            config={"unsupported": object()},
        ),
    )
    with pytest.raises(StorageFormatError, match="canonical JSON"):
        open_local_workspace_manifest(invalid)

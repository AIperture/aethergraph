"""Exact versioned workspace manifest owned by the local SQLite provider."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from ...contracts import (
    StorageFormatError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageScope,
)

LOCAL_STORAGE_FORMAT_VERSION = 1
LOCAL_PROVIDER_NAME = "local.sqlite"
WORKSPACE_MANIFEST_NAME = "workspace.json"
_MANIFEST_FIELDS = frozenset(
    {
        "format_version",
        "workspace_id",
        "owner_scope",
        "provider",
        "config_fingerprint",
        "created_at",
        "runtime_compatibility",
        "lifecycle",
    }
)


@dataclass(frozen=True, slots=True)
class LocalWorkspaceManifest:
    """Validated provider-private identity and compatibility for one workspace."""

    format_version: int
    workspace_id: str
    owner_scope: StorageScope
    provider: str
    config_fingerprint: str
    created_at: datetime
    runtime_compatibility: Mapping[str, str | int]
    clean_shutdown: bool
    last_maintenance_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.format_version != LOCAL_STORAGE_FORMAT_VERSION:
            raise StorageFormatError(f"Unsupported local workspace format: {self.format_version!r}")
        if not isinstance(self.workspace_id, str) or not self.workspace_id.strip():
            raise StorageFormatError("Local workspace manifest has a blank workspace_id")
        if self.provider != LOCAL_PROVIDER_NAME:
            raise StorageFormatError(
                f"Local workspace manifest selects unexpected provider {self.provider!r}"
            )
        if not self.config_fingerprint.startswith("sha256:"):
            raise StorageFormatError("Local workspace manifest has an invalid config fingerprint")
        _require_aware(self.created_at, "created_at")
        if self.last_maintenance_at is not None:
            _require_aware(self.last_maintenance_at, "last_maintenance_at")
        object.__setattr__(
            self,
            "runtime_compatibility",
            MappingProxyType(dict(self.runtime_compatibility)),
        )


def open_local_workspace_manifest(request: StorageOpenRequest) -> LocalWorkspaceManifest:
    """Initialize an empty local workspace or validate its exact existing manifest.

    A writable request may initialize only an absent or empty authorized root. An
    existing manifest is validated against the request without inferring, importing,
    or renaming any legacy provider-private layout.

    Examples:
        Initialize a fresh writable workspace:
            ```python
            manifest = open_local_workspace_manifest(request)
            ```

        Reject an unmanifested historical workspace:
            ```python
            with pytest.raises(StorageFormatError):
                open_local_workspace_manifest(read_only_request)
            ```

    Args:
        request: Trusted local-provider open request with an authorized absolute root.

    Returns:
        LocalWorkspaceManifest: Newly written or exactly validated workspace manifest.

    Notes:
        The manifest stores only a SHA-256 configuration fingerprint. Raw options,
        secret references, and resolved credentials are never persisted in it.
    """
    if request.selection.provider != LOCAL_PROVIDER_NAME:
        raise StorageFormatError(
            f"Local manifest opener cannot select provider {request.selection.provider!r}"
        )
    if request.expected_format_version != LOCAL_STORAGE_FORMAT_VERSION:
        raise StorageFormatError(
            "Local provider does not support requested workspace format "
            f"{request.expected_format_version!r}"
        )
    root = request.workspace_root
    if root.exists() and not root.is_dir():
        raise StorageFormatError("Local workspace root must be a directory")
    manifest_path = root / WORKSPACE_MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        manifest = read_local_workspace_manifest(root)
        _validate_request(manifest, request)
        return manifest

    if request.mode is StorageOpenMode.READ_ONLY:
        raise StorageFormatError("Read-only local workspace requires an existing manifest")
    if root.exists() and any(root.iterdir()):
        raise StorageFormatError("Refusing to initialize a non-empty unmanifested local workspace")

    root.mkdir(parents=True, exist_ok=True)
    created_at = request.clock.now()
    _require_aware(created_at, "clock.now()")
    manifest = LocalWorkspaceManifest(
        format_version=LOCAL_STORAGE_FORMAT_VERSION,
        workspace_id=request.workspace_id,
        owner_scope=request.owner_scope,
        provider=LOCAL_PROVIDER_NAME,
        config_fingerprint=_config_fingerprint(request.selection.config),
        created_at=created_at,
        runtime_compatibility={"storage_contract_version": 1},
        clean_shutdown=False,
    )
    _write_manifest(manifest_path, manifest)
    return manifest


def read_local_workspace_manifest(root: Path) -> LocalWorkspaceManifest:
    """Read and strictly validate one local-provider workspace manifest.

    The reader accepts only the current exact field set and converts canonical scope
    and timestamps into typed values. It never probes legacy database filenames.

    Examples:
        Read an initialized workspace:
            ```python
            manifest = read_local_workspace_manifest(workspace_root)
            ```

        Reject a malformed manifest:
            ```python
            with pytest.raises(StorageFormatError):
                read_local_workspace_manifest(workspace_root)
            ```

    Args:
        root: Authorized local workspace root containing `workspace.json`.

    Returns:
        LocalWorkspaceManifest: Strictly parsed current-format manifest.

    Notes:
        Unknown or missing fields, symlinks, malformed JSON, and unsupported formats
        raise `StorageFormatError` without attempting recovery or fallback.
    """
    manifest_path = root / WORKSPACE_MANIFEST_NAME
    if manifest_path.is_symlink():
        raise StorageFormatError("Local workspace manifest must not be a symlink")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageFormatError("Local workspace manifest is missing or malformed") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise StorageFormatError("Local workspace manifest fields do not match this format")
    try:
        owner_payload = payload["owner_scope"]
        lifecycle = payload["lifecycle"]
        compatibility = payload["runtime_compatibility"]
        if not isinstance(owner_payload, dict):
            raise TypeError("owner_scope")
        if not isinstance(lifecycle, dict) or set(lifecycle) != {
            "clean_shutdown",
            "last_maintenance_at",
        }:
            raise TypeError("lifecycle")
        if not isinstance(compatibility, dict):
            raise TypeError("runtime_compatibility")
        if any(not isinstance(key, str) for key in compatibility):
            raise TypeError("runtime_compatibility keys")
        if any(
            isinstance(value, bool) or not isinstance(value, (str, int))
            for value in compatibility.values()
        ):
            raise TypeError("runtime_compatibility values")
        clean_shutdown = lifecycle["clean_shutdown"]
        if not isinstance(clean_shutdown, bool):
            raise TypeError("clean_shutdown")
        last_maintenance = lifecycle["last_maintenance_at"]
        return LocalWorkspaceManifest(
            format_version=_strict_int(payload["format_version"]),
            workspace_id=_strict_str(payload["workspace_id"]),
            owner_scope=StorageScope(**owner_payload),
            provider=_strict_str(payload["provider"]),
            config_fingerprint=_strict_str(payload["config_fingerprint"]),
            created_at=_parse_datetime(payload["created_at"]),
            runtime_compatibility=compatibility,
            clean_shutdown=clean_shutdown,
            last_maintenance_at=(
                None if last_maintenance is None else _parse_datetime(last_maintenance)
            ),
        )
    except (TypeError, ValueError, KeyError, StorageFormatError) as exc:
        if isinstance(exc, StorageFormatError):
            raise
        raise StorageFormatError("Local workspace manifest values are malformed") from exc


def update_local_workspace_lifecycle(
    root: Path,
    *,
    clean_shutdown: bool,
    last_maintenance_at: datetime | None,
) -> LocalWorkspaceManifest:
    """Atomically update only the provider-owned workspace lifecycle fields.

    The current exact manifest is reread and validated before replacement. Identity,
    provider selection, compatibility, and configuration fingerprint remain unchanged.

    Examples:
        Mark a workspace active:
            ```python
            manifest = update_local_workspace_lifecycle(
                root,
                clean_shutdown=False,
                last_maintenance_at=previous_maintenance,
            )
            ```

        Record a clean maintained shutdown:
            ```python
            manifest = update_local_workspace_lifecycle(
                root,
                clean_shutdown=True,
                last_maintenance_at=clock.now(),
            )
            ```

    Args:
        root: Authorized manifested local workspace root.
        clean_shutdown: Whether all provider durability barriers and closes completed.
        last_maintenance_at: Latest completed checkpoint timestamp, when available.

    Returns:
        LocalWorkspaceManifest: Strict updated manifest committed by atomic replacement.

    Notes:
        This operation never changes or persists raw provider configuration, secret
        references, or resolved credentials.
    """
    if not isinstance(clean_shutdown, bool):
        raise TypeError("clean_shutdown must be a bool")
    if last_maintenance_at is not None:
        _require_aware(last_maintenance_at, "last_maintenance_at")
    current = read_local_workspace_manifest(root)
    updated = replace(
        current,
        clean_shutdown=clean_shutdown,
        last_maintenance_at=last_maintenance_at,
    )
    _write_manifest(root / WORKSPACE_MANIFEST_NAME, updated)
    return updated


def _validate_request(
    manifest: LocalWorkspaceManifest,
    request: StorageOpenRequest,
) -> None:
    if manifest.workspace_id != request.workspace_id:
        raise StorageFormatError("Local workspace_id does not match the open request")
    if manifest.owner_scope != request.owner_scope:
        raise StorageFormatError("Local workspace owner scope does not match the open request")
    if manifest.config_fingerprint != _config_fingerprint(request.selection.config):
        raise StorageFormatError("Local workspace configuration fingerprint does not match")


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StorageFormatError("Local provider config is not canonical JSON") from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _write_manifest(path: Path, manifest: LocalWorkspaceManifest) -> None:
    payload = {
        "format_version": manifest.format_version,
        "workspace_id": manifest.workspace_id,
        "owner_scope": manifest.owner_scope.as_filter(),
        "provider": manifest.provider,
        "config_fingerprint": manifest.config_fingerprint,
        "created_at": manifest.created_at.isoformat(),
        "runtime_compatibility": dict(manifest.runtime_compatibility),
        "lifecycle": {
            "clean_shutdown": manifest.clean_shutdown,
            "last_maintenance_at": (
                None
                if manifest.last_maintenance_at is None
                else manifest.last_maintenance_at.isoformat()
            ),
        },
    }
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise StorageFormatError("Failed to commit local workspace manifest") from exc


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("timestamp")
    parsed = datetime.fromisoformat(value)
    _require_aware(parsed, "timestamp")
    return parsed


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise StorageFormatError(f"Local workspace {field_name} must be timezone-aware")


def _strict_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("integer")
    return value


def _strict_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("string")
    return value

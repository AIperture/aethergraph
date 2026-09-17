"""Explicit maintenance of fenced, current-format local runtime history."""

from pathlib import Path
import secrets

from aethergraph.config.storage_provider import StorageProviderSettings
from aethergraph.services.canonical_storage_scope import validate_storage_owner_scope
from aethergraph.services.clock.clock import SystemClock
from aethergraph.storage.contracts import (
    StorageConfigurationError,
    StorageOpenMode,
    StorageOpenRequest,
    StorageProviderSelection,
    StorageScope,
)
from aethergraph.storage.providers.local_sqlite import (
    LocalStorageProvider,
    read_local_workspace_manifest,
)
from aethergraph.storage.providers.local_sqlite.session_purge import LocalSessionPurge


class _NoExecutionSecrets:
    async def resolve(self, reference: str) -> bytes:
        raise StorageConfigurationError("Historical deletion cannot resolve execution secrets")


async def purge_local_sessions(
    workspace_root: Path,
    *,
    owner_scope: StorageScope,
    session_ids: tuple[str, ...],
    selection: StorageProviderSelection | None = None,
    stopped_session_ids: tuple[str, ...] = (),
) -> None:
    """Permanently remove exact quiescent sessions without removing their workspace.

    Examples:
        Delete a previously fenced session:
            ```python
            await purge_local_sessions(root, owner_scope=owner, session_ids=("session-1",))
            ```

    Args:
        workspace_root: Authorized canonical AG workspace, never a source directory.
        owner_scope: Exact storage owner captured from the authorized workspace.
        session_ids: Exact session identities owned by the caller's deletion scope.
        selection: Original provider configuration; defaults to built-in local settings.
        stopped_session_ids: Selected sessions whose execution process is confirmed gone.
            The caller must fence their admission before providing this evidence.

    Returns:
        None: All selected evidence and exclusively referenced blob bytes are removed.

    Notes:
        The caller must persist an execution fence first. Interrupted maintenance is
        resumable with identical arguments. Other sessions and shared blobs remain.
        Missing workspaces, provider mismatches, or inaccessible storage fail visibly;
        the owning application determines whether absence is already-cleaned evidence.
    """
    root = Path(workspace_root).absolute()
    if root.resolve(strict=True) != root:
        raise StorageConfigurationError("Runtime maintenance cannot follow substituted paths")
    manifest = read_local_workspace_manifest(root)
    if manifest.owner_scope != owner_scope:
        raise StorageConfigurationError("Runtime maintenance owner does not match the workspace")
    validate_storage_owner_scope(owner_scope)
    selected = selection or StorageProviderSettings(provider="local.sqlite").to_selection()
    reference = selected.config["continuation_token_secret_ref"]
    bundle = LocalStorageProvider(
        continuation_token_secret_ref=reference,
        continuation_token_secret=secrets.token_bytes(32),
    ).open(
        StorageOpenRequest(
            workspace_id=manifest.workspace_id,
            workspace_root=root,
            owner_scope=manifest.owner_scope,
            selection=selected,
            mode=StorageOpenMode.READ_WRITE,
            expected_format_version=manifest.format_version,
            clock=SystemClock(),
            secrets=_NoExecutionSecrets(),
        )
    )
    try:
        await LocalSessionPurge(bundle).purge(
            owner_scope, session_ids, stopped_session_ids=stopped_session_ids
        )
    finally:
        await bundle.close()


def local_cleanup_owner(workspace_root: Path) -> StorageScope:
    """Read the storage owner of an already-authorized canonical runtime workspace.

    The application must authorize the path first and retain this identity in its
    cleanup receipt. Storage ownership is independent of application project IDs.
    This function neither opens databases nor starts a runtime.
    """
    root = Path(workspace_root).absolute()
    if root.resolve(strict=True) != root:
        raise StorageConfigurationError("Runtime maintenance cannot follow substituted paths")
    owner = read_local_workspace_manifest(root).owner_scope
    validate_storage_owner_scope(owner)
    return owner


__all__ = ["StorageScope", "local_cleanup_owner", "purge_local_sessions"]

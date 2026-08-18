"""Synchronous dependency composition for the built-in local storage provider."""

from __future__ import annotations

import hashlib
import hmac

from aethergraph.contracts.services.llm import EmbeddingClientProtocol

from .contracts import StorageConfigurationError, StorageProviderSelection
from .provider_markers import BUILTIN_LOCAL_CONTINUATION_SECRET_REF
from .provider_registry import StorageProviderRegistry
from .providers.local_sqlite import LocalStorageProvider
from .providers.local_sqlite.manifest import LOCAL_PROVIDER_NAME

_CONTINUATION_SECRET_DOMAIN = b"aethergraph.storage.continuation.v1\x00"


def build_builtin_local_storage_registry(
    *,
    selection: StorageProviderSelection,
    workspace_id: str,
    auth_signing_secret: str | bytes,
    embedder: EmbeddingClientProtocol | None = None,
) -> StorageProviderRegistry:
    """Build the exact built-in local registry for one synchronous runtime open.

    Intro:
        Derives workspace-bound continuation-token HMAC material from the already
        resolved authentication signing secret, then captures only that material and
        the built-in derivation reference in a zero-argument local-provider factory.
        The function performs no provider open, secret lookup, or filesystem access.

    Examples:
        Build the registry for a local workspace:
            ```python
            registry = build_builtin_local_storage_registry(
                selection=settings.to_selection(),
                workspace_id="workspace-1",
                auth_signing_secret="resolved-auth-secret",
            )
            ```

        Capture an optional embedding client:
            ```python
            registry = build_builtin_local_storage_registry(
                selection=selection,
                workspace_id="workspace-2",
                auth_signing_secret=auth_secret,
                embedder=embedding_client,
            )
            ```

    Args:
        selection: Exact built-in local selection carrying the fixed derivation reference.
        workspace_id: Stable exact canonical identity of the workspace being opened.
        auth_signing_secret: Already resolved AG authentication signing material.
        embedder: Optional embedding client captured for semantic local search.

    Returns:
        StorageProviderRegistry: Registry containing only the exact built-in local factory.

    Notes:
        The derived bytes stay process-local in the factory closure and are never added
        to provider selection, manifests, settings, diagnostics, or logs. External
        providers must be explicitly assembled and registered by their own composition
        boundary; this helper never selects or falls back to local for them. Rotating
        the authentication signing secret invalidates outstanding continuation tokens.
    """
    if selection.provider != LOCAL_PROVIDER_NAME:
        raise StorageConfigurationError(
            "Built-in local composition requires the exact local.sqlite selection"
        )
    reference = selection.config.get("continuation_token_secret_ref")
    if reference != BUILTIN_LOCAL_CONTINUATION_SECRET_REF:
        raise StorageConfigurationError(
            "Built-in local composition requires its fixed continuation derivation reference"
        )
    continuation_secret = _derive_workspace_continuation_secret(
        workspace_id=workspace_id,
        auth_signing_secret=auth_signing_secret,
    )

    def create_local_provider() -> LocalStorageProvider:
        return LocalStorageProvider(
            continuation_token_secret_ref=BUILTIN_LOCAL_CONTINUATION_SECRET_REF,
            continuation_token_secret=continuation_secret,
            embedder=embedder,
        )

    return StorageProviderRegistry({LOCAL_PROVIDER_NAME: create_local_provider})


def _derive_workspace_continuation_secret(
    *,
    workspace_id: str,
    auth_signing_secret: str | bytes,
) -> bytes:
    if (
        not isinstance(workspace_id, str)
        or not workspace_id.strip()
        or workspace_id != workspace_id.strip()
    ):
        raise StorageConfigurationError("workspace_id must be an exact non-empty string")
    if isinstance(auth_signing_secret, str):
        key = auth_signing_secret.encode("utf-8")
    elif isinstance(auth_signing_secret, bytes):
        key = bytes(auth_signing_secret)
    else:
        raise StorageConfigurationError("auth_signing_secret must be str or bytes")
    if not key or not key.strip():
        raise StorageConfigurationError("auth_signing_secret must be non-empty")
    return hmac.digest(
        key,
        _CONTINUATION_SECRET_DOMAIN + workspace_id.encode("utf-8"),
        hashlib.sha256,
    )

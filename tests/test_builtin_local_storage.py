from __future__ import annotations

import inspect

import pytest

from aethergraph.config.storage_provider import StorageProviderSettings
from aethergraph.storage.builtin_local import (
    _derive_workspace_continuation_secret,
    build_builtin_local_storage_registry,
)
from aethergraph.storage.contracts import StorageConfigurationError, StorageProviderSelection
from aethergraph.storage.provider_markers import BUILTIN_LOCAL_CONTINUATION_SECRET_REF
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_AUTH_SECRET = "already-resolved-auth-signing-secret"


def _selection() -> StorageProviderSelection:
    return StorageProviderSettings(provider="local.sqlite").to_selection()


def test_builtin_local_registry_captures_stable_workspace_bound_material() -> None:
    first_registry = build_builtin_local_storage_registry(
        selection=_selection(),
        workspace_id="workspace-1",
        auth_signing_secret=_AUTH_SECRET,
    )
    second_registry = build_builtin_local_storage_registry(
        selection=_selection(),
        workspace_id="workspace-1",
        auth_signing_secret=_AUTH_SECRET.encode(),
    )

    assert first_registry.names() == ("local.sqlite",)
    first = first_registry.create("local.sqlite")
    second = second_registry.create("local.sqlite")
    assert isinstance(first, LocalStorageProvider)
    assert isinstance(second, LocalStorageProvider)
    assert first is not second
    assert first._continuation_token_secret_ref == BUILTIN_LOCAL_CONTINUATION_SECRET_REF
    assert first._continuation_token_secret == second._continuation_token_secret
    assert len(first._continuation_token_secret) == 32
    assert first._continuation_token_secret != _AUTH_SECRET.encode()
    first.validate_config(_selection())


def test_builtin_local_derivation_is_domain_and_workspace_separated() -> None:
    baseline = _derive_workspace_continuation_secret(
        workspace_id="workspace-1",
        auth_signing_secret=_AUTH_SECRET,
    )

    assert baseline != _derive_workspace_continuation_secret(
        workspace_id="workspace-2",
        auth_signing_secret=_AUTH_SECRET,
    )
    assert baseline != _derive_workspace_continuation_secret(
        workspace_id="workspace-1",
        auth_signing_secret="different-auth-signing-secret",
    )


@pytest.mark.parametrize(
    ("workspace_id", "auth_secret"),
    [
        ("", _AUTH_SECRET),
        (" workspace-1", _AUTH_SECRET),
        ("workspace-1 ", _AUTH_SECRET),
        ("workspace-1", ""),
        ("workspace-1", "   "),
        ("workspace-1", b""),
        ("workspace-1", object()),
    ],
)
def test_builtin_local_derivation_rejects_ambiguous_or_missing_identity(
    workspace_id: str,
    auth_secret: object,
) -> None:
    with pytest.raises(StorageConfigurationError):
        _derive_workspace_continuation_secret(
            workspace_id=workspace_id,
            auth_signing_secret=auth_secret,  # type: ignore[arg-type]
        )


def test_builtin_local_registry_rejects_external_or_inexact_selection_without_fallback() -> None:
    external = StorageProviderSelection(provider="company.external", config={})
    wrong_reference = StorageProviderSelection(
        provider="local.sqlite",
        config={"continuation_token_secret_ref": "secret://other"},
    )

    with pytest.raises(StorageConfigurationError, match="exact local.sqlite"):
        build_builtin_local_storage_registry(
            selection=external,
            workspace_id="workspace-1",
            auth_signing_secret=_AUTH_SECRET,
        )
    with pytest.raises(StorageConfigurationError, match="fixed continuation"):
        build_builtin_local_storage_registry(
            selection=wrong_reference,
            workspace_id="workspace-1",
            auth_signing_secret=_AUTH_SECRET,
        )


def test_builtin_local_registry_public_docstring_has_required_sections() -> None:
    docstring = inspect.getdoc(build_builtin_local_storage_registry)

    assert docstring is not None
    required = ("Intro:", "Examples:", "Args:", "Returns:", "Notes:")
    positions = tuple(docstring.index(section) for section in required)
    assert positions == tuple(sorted(positions))
    assert docstring.count("```python") >= 2

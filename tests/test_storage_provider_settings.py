from __future__ import annotations

from dataclasses import fields

from pydantic import ValidationError
import pytest

from aethergraph.config.storage_provider import (
    LocalSQLiteProviderOptions,
    StorageProviderSettings,
)
from aethergraph.storage.provider_markers import BUILTIN_LOCAL_CONTINUATION_SECRET_REF
from aethergraph.storage.providers.local_sqlite import LocalStorageProvider

_SECRET_REF = BUILTIN_LOCAL_CONTINUATION_SECRET_REF
_SECRET = b"provider-settings-test-secret-32bytes"


def test_local_provider_settings_are_typed_exact_and_provider_validated() -> None:
    settings = StorageProviderSettings(
        provider="local.sqlite",
        profile="default",
        options={
            "continuation_token_secret_ref": _SECRET_REF,
            "durability": "full",
            "search_max_candidates": 20_000,
        },
    )

    assert isinstance(settings.options, LocalSQLiteProviderOptions)
    selection = settings.to_selection()
    assert selection.provider == "local.sqlite"
    assert selection.config == {
        "busy_timeout_ms": 5_000,
        "continuation_token_secret_ref": _SECRET_REF,
        "durability": "full",
        "runtime_output_max_pending_frames": 10_000,
        "search_max_candidates": 20_000,
    }
    LocalStorageProvider(
        continuation_token_secret_ref=_SECRET_REF,
        continuation_token_secret=_SECRET,
    ).validate_config(selection)


@pytest.mark.parametrize(
    "payload",
    [
        {"provider": " local.sqlite", "options": {"continuation_token_secret_ref": _SECRET_REF}},
        {
            "provider": "local.sqlite",
            "profile": " ",
            "options": {"continuation_token_secret_ref": _SECRET_REF},
        },
        {
            "provider": "local.sqlite",
            "options": {"continuation_token_secret_ref": f" {_SECRET_REF}"},
        },
        {
            "provider": "local.sqlite",
            "options": {"continuation_token_secret_ref": _SECRET_REF, "fallback": True},
        },
        {
            "provider": "local.sqlite",
            "options": {"continuation_token_secret_ref": _SECRET_REF},
            "path": "hidden.db",
        },
        {"provider": "company.external", "options": {"app_id": "deprecated"}},
        {"provider": "company.external", "options": {"application_id": "alias"}},
        {"provider": "company.external", "options": {"client_id": "compatibility"}},
    ],
)
def test_provider_settings_reject_inexact_or_unknown_builtin_configuration(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        StorageProviderSettings.model_validate(payload)


def test_external_provider_options_are_copied_without_local_interpretation() -> None:
    original = {"cluster": "primary", "routing": {"region": "west"}}
    settings = StorageProviderSettings(
        provider="company.external",
        profile="production",
        options=original,
    )
    original["cluster"] = "changed"
    original["routing"]["region"] = "changed"  # type: ignore[index]

    selection = settings.to_selection()

    assert selection.provider == "company.external"
    assert selection.config == {"cluster": "primary", "routing": {"region": "west"}}
    assert settings.profile == "production"


def test_builtin_local_settings_supply_only_the_fixed_derivation_reference() -> None:
    settings = StorageProviderSettings(provider="local.sqlite")

    assert isinstance(settings.options, LocalSQLiteProviderOptions)
    assert settings.options.continuation_token_secret_ref == _SECRET_REF
    assert settings.to_selection().config["continuation_token_secret_ref"] == _SECRET_REF

    with pytest.raises(ValidationError, match="workspace-bound auth-signing derivation"):
        StorageProviderSettings(
            provider="local.sqlite",
            options={"continuation_token_secret_ref": "secret://legacy/continuations"},
        )


def test_provider_settings_are_frozen_and_exclude_deprecated_identity_and_paths() -> None:
    settings = StorageProviderSettings(
        provider="company.external",
        options={"endpoint": "managed"},
    )

    with pytest.raises(ValidationError):
        settings.provider = "other"  # type: ignore[misc]
    assert {field.name for field in fields(settings.to_selection())} == {"provider", "config"}
    assert not {"app_id", "application_id", "client_id", "workspace_root"}.intersection(
        StorageProviderSettings.model_fields
    )

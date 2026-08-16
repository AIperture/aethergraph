"""Strict provider-selection settings prepared for the S9 composition cut."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aethergraph.storage.contracts import StorageProviderSelection
from aethergraph.storage.provider_markers import BUILTIN_LOCAL_CONTINUATION_SECRET_REF

_DEPRECATED_IDENTITY_OPTION_KEYS = frozenset({"app_id", "application_id", "client_id"})


class LocalSQLiteProviderOptions(BaseModel):
    """Typed current-format options accepted by the built-in local provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    busy_timeout_ms: int = Field(default=5_000, ge=1, le=120_000)
    continuation_token_secret_ref: str = Field(
        default=BUILTIN_LOCAL_CONTINUATION_SECRET_REF,
        min_length=1,
    )
    durability: Literal["normal", "full"] = "normal"
    runtime_output_max_pending_frames: int = Field(default=10_000, ge=1, le=1_000_000)
    search_max_candidates: int = Field(default=10_000, ge=1_000, le=100_000)

    @field_validator("continuation_token_secret_ref")
    @classmethod
    def _validate_secret_reference(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("continuation_token_secret_ref must be exact without whitespace")
        if value != BUILTIN_LOCAL_CONTINUATION_SECRET_REF:
            raise ValueError(
                "built-in local continuation_token_secret_ref must select the "
                "workspace-bound auth-signing derivation"
            )
        return value


class StorageProviderSettings(BaseModel):
    """Select one exact storage provider without activating runtime composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    profile: str = "default"
    options: LocalSQLiteProviderOptions | dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _type_builtin_options(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        data = deepcopy(dict(value))
        raw_options = data.get("options", {})
        if isinstance(raw_options, Mapping):
            deprecated_keys = _DEPRECATED_IDENTITY_OPTION_KEYS.intersection(raw_options)
            if deprecated_keys:
                joined = ", ".join(sorted(deprecated_keys))
                raise ValueError(
                    f"deprecated compatibility identity is not provider configuration: {joined}"
                )
        if data.get("provider") == "local.sqlite":
            data["options"] = LocalSQLiteProviderOptions.model_validate(raw_options)
        return data

    @model_validator(mode="after")
    def _validate_exact_selection(self) -> StorageProviderSettings:
        for name in ("provider", "profile"):
            value = getattr(self, name)
            if not value or value != value.strip():
                raise ValueError(f"{name} must be an exact non-empty string")
        if self.provider == "local.sqlite" and not isinstance(
            self.options,
            LocalSQLiteProviderOptions,
        ):
            raise ValueError("local.sqlite options must use LocalSQLiteProviderOptions")
        return self

    def to_selection(self) -> StorageProviderSelection:
        """Create the immutable canonical provider selection.

        Built-in local options are emitted with validated defaults, while external
        provider options are copied without interpreting provider-owned keys. The
        profile label remains a configuration-resolution concern and is not sent to
        provider validation or persisted in its option map.

        Examples:
            Convert a local provider selection:
                ```python
                settings = StorageProviderSettings(
                    provider="local.sqlite",
                )
                selection = settings.to_selection()
                ```

            Preserve external provider options:
                ```python
                settings = StorageProviderSettings(
                    provider="company.external",
                    profile="production",
                    options={"cluster": "primary"},
                )
                selection = settings.to_selection()
                ```

        Args:
            None.

        Returns:
            StorageProviderSelection: Exact provider name and copied option mapping.

        Notes:
            This method performs no provider lookup, secret resolution, filesystem
            access, fallback selection, or runtime mutation. Deprecated App/client
            metadata is not accepted as a canonical setting.
        """
        if isinstance(self.options, LocalSQLiteProviderOptions):
            options = self.options.model_dump(mode="python")
        else:
            options = deepcopy(self.options)
        return StorageProviderSelection(provider=self.provider, config=options)

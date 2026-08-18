"""Canonical environment encoding for AetherGraph LLM profiles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any

from pydantic import BaseModel, SecretStr

from .llm import EmbeddingProfile, ImageGenerationProfileSettings, LLMProfile

_PROFILE_FIELDS = frozenset(LLMProfile.model_fields)
_EMBEDDING_PROFILE_FIELDS = frozenset(EmbeddingProfile.model_fields)
_IMAGE_GENERATION_PROFILE_FIELDS = frozenset(ImageGenerationProfileSettings.model_fields)


def aethergraph_env_key(*parts: str) -> str:
    """
    Build one canonical nested AetherGraph environment key.

    Each component is normalized to uppercase and joined with the Pydantic
    settings nested delimiter used by `AppSettings`.

    Examples:
        Build a default-profile model key:
        ```python
        assert aethergraph_env_key("llm", "default", "model") == (
            "AETHERGRAPH_LLM__DEFAULT__MODEL"
        )
        ```

        Build a named-profile provider key:
        ```python
        assert aethergraph_env_key("llm", "profiles", "fast", "provider") == (
            "AETHERGRAPH_LLM__PROFILES__FAST__PROVIDER"
        )
        ```

    Args:
        parts: Nonempty key path components.

    Returns:
        str: The normalized `AETHERGRAPH_` environment key.

    Notes:
        Empty components are rejected so callers cannot create ambiguous nested
        paths.
    """

    if not parts or any(not str(part).strip() for part in parts):
        raise ValueError("AetherGraph environment key parts must be nonempty.")
    return "AETHERGRAPH_" + "__".join(str(part).upper() for part in parts)


def encode_llm_profile_env(
    name: str,
    profile: LLMProfile | BaseModel | Mapping[str, Any],
) -> dict[str, str]:
    """
    Encode one complete or partial LLM profile as environment rows.

    The function accepts canonical profiles and validated API payload models.
    Only fields owned by `LLMProfile` are emitted, and values set to `None` are
    omitted.

    Examples:
        Encode the Agent Engine profile:
        ```python
        rows = encode_llm_profile_env(
            "default",
            LLMProfile(provider="openai", model="gpt-5-mini"),
        )
        assert rows["AETHERGRAPH_LLM__DEFAULT__MODEL"] == "gpt-5-mini"
        ```

        Encode a partial named-profile payload:
        ```python
        rows = encode_llm_profile_env(
            "summarizer",
            {"provider": "anthropic", "vision_enabled": False},
        )
        assert rows["AETHERGRAPH_LLM__PROFILES__SUMMARIZER__VISION_ENABLED"] == "false"
        ```

    Args:
        name: Profile name; `default` selects the default-profile path.
        profile: Canonical profile, validated model, or profile-field mapping.

    Returns:
        dict[str, str]: Environment keys and serialized values for the profile.

    Notes:
        Unknown fields are rejected. Secret values are unwrapped only for
        persistence and are never logged by this module.
    """

    return _encode_operation_profile_env(
        name,
        profile,
        section="llm",
        fields=_PROFILE_FIELDS,
        label="LLM",
    )


def encode_llm_profiles_env(
    profiles: Mapping[str, LLMProfile | BaseModel | Mapping[str, Any]],
) -> dict[str, str]:
    """
    Encode a profile mapping into one deterministic environment row set.

    Profiles are encoded in normalized name order so exact-file persistence is
    stable across equivalent mappings.

    Examples:
        Encode only the required default profile:
        ```python
        rows = encode_llm_profiles_env({"default": LLMProfile()})
        assert "AETHERGRAPH_LLM__DEFAULT__PROVIDER" in rows
        ```

        Encode default and named profiles:
        ```python
        rows = encode_llm_profiles_env({
            "default": LLMProfile(model="gpt-5-mini"),
            "fast": LLMProfile(provider="openrouter", model="fast-model"),
        })
        assert rows["AETHERGRAPH_LLM__PROFILES__FAST__MODEL"] == "fast-model"
        ```

    Args:
        profiles: Profile names mapped to canonical or validated profile data.

    Returns:
        dict[str, str]: Combined deterministic environment rows.

    Notes:
        Duplicate normalized names are rejected rather than merged.
    """

    return _encode_operation_profiles_env(
        profiles,
        encoder=encode_llm_profile_env,
        label="LLM",
    )


def encode_embedding_profile_env(
    name: str,
    profile: EmbeddingProfile | BaseModel | Mapping[str, Any],
) -> dict[str, str]:
    """Encode one complete or partial Embedding profile as environment rows.

    Intro:
        Uses AG's canonical `embed` settings path and closed public profile field
        set for deterministic application persistence.

    Examples:
        Encode the default embedding model:
            ```python
            rows = encode_embedding_profile_env(
                "default",
                EmbeddingProfile(model="text-embedding-3-large"),
            )
            assert rows["AETHERGRAPH_EMBED__DEFAULT__MODEL"] == "text-embedding-3-large"
            ```

        Encode a named local profile:
            ```python
            rows = encode_embedding_profile_env(
                "local",
                {"provider": "ollama", "model": "nomic-embed-text"},
            )
            assert rows["AETHERGRAPH_EMBED__PROFILES__LOCAL__PROVIDER"] == "ollama"
            ```

    Args:
        name: Profile name; `default` selects the default-profile path.
        profile: Canonical profile, validated model, or profile-field mapping.

    Returns:
        dict[str, str]: Environment keys and serialized values for the profile.

    Notes:
        Unknown Chat or Image Generation fields are rejected at this boundary.
    """

    return _encode_operation_profile_env(
        name,
        profile,
        section="embed",
        fields=_EMBEDDING_PROFILE_FIELDS,
        label="Embedding",
    )


def encode_embedding_profiles_env(
    profiles: Mapping[
        str,
        EmbeddingProfile | BaseModel | Mapping[str, Any],
    ],
) -> dict[str, str]:
    """Encode a deterministic collection of Embedding profiles.

    Intro:
        Normalizes and sorts default plus named profiles before delegating each
        profile to the closed Embedding writer.

    Examples:
        Encode a default profile:
            ```python
            rows = encode_embedding_profiles_env({"default": EmbeddingProfile()})
            assert "AETHERGRAPH_EMBED__DEFAULT__MODEL" in rows
            ```

        Encode a named profile:
            ```python
            rows = encode_embedding_profiles_env({
                "default": EmbeddingProfile(),
                "search": EmbeddingProfile(model="text-embedding-3-large"),
            })
            assert "AETHERGRAPH_EMBED__PROFILES__SEARCH__MODEL" in rows
            ```

    Args:
        profiles: Profile names mapped to canonical or validated profile data.

    Returns:
        dict[str, str]: Combined deterministic environment rows.

    Notes:
        Duplicate normalized names are rejected rather than merged.
    """

    return _encode_operation_profiles_env(
        profiles,
        encoder=encode_embedding_profile_env,
        label="Embedding",
    )


def encode_image_generation_profile_env(
    name: str,
    profile: ImageGenerationProfileSettings | BaseModel | Mapping[str, Any],
) -> dict[str, str]:
    """Encode one Image Generation profile as environment rows.

    Intro:
        Uses AG's canonical `image_generation` settings path without passing
        Image Generation fields through a Chat profile.

    Examples:
        Encode the default image model:
            ```python
            rows = encode_image_generation_profile_env(
                "default",
                ImageGenerationProfileSettings(model="gpt-image-1"),
            )
            assert rows["AETHERGRAPH_IMAGE_GENERATION__DEFAULT__MODEL"] == "gpt-image-1"
            ```

        Encode a named profile:
            ```python
            rows = encode_image_generation_profile_env(
                "design",
                {"provider": "google", "model": "imagen-4.0-generate-001"},
            )
            assert rows["AETHERGRAPH_IMAGE_GENERATION__PROFILES__DESIGN__PROVIDER"] == "google"
            ```

    Args:
        name: Profile name; `default` selects the default-profile path.
        profile: Canonical profile, validated model, or profile-field mapping.

    Returns:
        dict[str, str]: Environment keys and serialized values for the profile.

    Notes:
        Credentials are unwrapped only for persistence and never logged.
    """

    return _encode_operation_profile_env(
        name,
        profile,
        section="image_generation",
        fields=_IMAGE_GENERATION_PROFILE_FIELDS,
        label="Image Generation",
    )


def encode_image_generation_profiles_env(
    profiles: Mapping[
        str,
        ImageGenerationProfileSettings | BaseModel | Mapping[str, Any],
    ],
) -> dict[str, str]:
    """Encode a deterministic collection of Image Generation profiles.

    Intro:
        Normalizes and sorts default plus named image profiles through one AG
        persistence implementation.

    Examples:
        Encode a default profile:
            ```python
            rows = encode_image_generation_profiles_env({
                "default": ImageGenerationProfileSettings()
            })
            assert "AETHERGRAPH_IMAGE_GENERATION__DEFAULT__MODEL" in rows
            ```

        Encode a named profile:
            ```python
            rows = encode_image_generation_profiles_env({
                "default": ImageGenerationProfileSettings(),
                "design": ImageGenerationProfileSettings(model="image-model"),
            })
            assert "AETHERGRAPH_IMAGE_GENERATION__PROFILES__DESIGN__MODEL" in rows
            ```

    Args:
        profiles: Profile names mapped to canonical or validated profile data.

    Returns:
        dict[str, str]: Combined deterministic environment rows.

    Notes:
        Duplicate normalized names are rejected rather than merged.
    """

    return _encode_operation_profiles_env(
        profiles,
        encoder=encode_image_generation_profile_env,
        label="Image Generation",
    )


def _encode_operation_profile_env(
    name: str,
    profile: BaseModel | Mapping[str, Any],
    *,
    section: str,
    fields: frozenset[str],
    label: str,
) -> dict[str, str]:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError(f"{label} profile name must be nonempty.")
    values = _profile_values(profile)
    unknown = set(values) - fields
    if unknown:
        raise ValueError(f"Unknown {label} profile fields: {', '.join(sorted(unknown))}")
    prefix = (
        (section, "default")
        if normalized_name == "default"
        else (section, "profiles", normalized_name)
    )
    return {
        aethergraph_env_key(*prefix, field): _serialize_env_value(value)
        for field, value in values.items()
        if value is not None
    }


def _encode_operation_profiles_env(
    profiles: Mapping[str, Any],
    *,
    encoder: Callable[[str, Any], dict[str, str]],
    label: str,
) -> dict[str, str]:
    normalized: dict[str, Any] = {}
    for name, profile in profiles.items():
        key = name.strip().lower()
        if key in normalized:
            raise ValueError(f"Duplicate normalized {label} profile name: {key}")
        normalized[key] = profile
    rows: dict[str, str] = {}
    for name in sorted(normalized, key=lambda item: (item != "default", item)):
        rows.update(encoder(name, normalized[name]))
    return rows


def _profile_values(
    profile: BaseModel | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(profile, BaseModel):
        return profile.model_dump(exclude_none=True)
    return {str(key): value for key, value in profile.items() if value is not None}


def _serialize_env_value(value: Any) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


__all__ = [
    "aethergraph_env_key",
    "encode_embedding_profile_env",
    "encode_embedding_profiles_env",
    "encode_image_generation_profile_env",
    "encode_image_generation_profiles_env",
    "encode_llm_profile_env",
    "encode_llm_profiles_env",
]

"""Request / response schemas for the Settings API (local mode only)."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_secret(value: str | None) -> str | None:
    """Return a masked version of a secret string, or None if empty."""
    if not value:
        return None
    if len(value) <= 8:
        return "****"
    return value[:4] + "****" + value[-4:]


MASKED_SENTINEL = "****"


def _is_masked(value: str | None) -> bool:
    """Return True if the value looks like a masked secret (should not be persisted)."""
    if not value:
        return True
    return MASKED_SENTINEL in value


# ---------------------------------------------------------------------------
# Status (lightweight check)
# ---------------------------------------------------------------------------


class SettingsStatusResponse(BaseModel):
    workspace: str
    deploy_mode: str
    llm_configured: bool = False
    embedding_configured: bool = False
    slack_configured: bool = False
    telegram_configured: bool = False


# ---------------------------------------------------------------------------
# LLM profile views & payloads
# ---------------------------------------------------------------------------


class LLMProfileView(BaseModel):
    """Read-only view of an LLM profile (secrets masked)."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    base_url: str | None = None
    timeout: float = 60.0
    api_key: str | None = None  # masked
    thinking_budget: int | None = None
    reasoning_summary: str | None = None


class LLMProfilePayload(BaseModel):
    """Write payload for an LLM profile."""

    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None


# ---------------------------------------------------------------------------
# Embedding profile views & payloads
# ---------------------------------------------------------------------------


class EmbeddingProfileView(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    timeout: float = 60.0
    api_key: str | None = None  # masked


class EmbeddingProfilePayload(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None


# ---------------------------------------------------------------------------
# Channel views & payloads
# ---------------------------------------------------------------------------


class SlackView(BaseModel):
    enabled: bool = False
    bot_token: str | None = None  # masked
    signing_secret: str | None = None  # masked


class SlackPayload(BaseModel):
    enabled: bool | None = None
    bot_token: str | None = None
    signing_secret: str | None = None


class TelegramView(BaseModel):
    enabled: bool = False
    bot_token: str | None = None  # masked


class TelegramPayload(BaseModel):
    enabled: bool | None = None
    bot_token: str | None = None


# ---------------------------------------------------------------------------
# Full settings GET / PUT
# ---------------------------------------------------------------------------


class SettingsGetResponse(BaseModel):
    workspace: str
    deploy_mode: str
    llm: dict[str, LLMProfileView] = Field(default_factory=dict)
    embedding: dict[str, EmbeddingProfileView] = Field(default_factory=dict)
    slack: SlackView = SlackView()
    telegram: TelegramView = TelegramView()


class SettingsUpdateRequest(BaseModel):
    llm: dict[str, LLMProfilePayload] | None = None
    embedding: dict[str, EmbeddingProfilePayload] | None = None
    slack: SlackPayload | None = None
    telegram: TelegramPayload | None = None

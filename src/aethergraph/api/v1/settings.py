"""Settings API — local-mode only configuration endpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException  # type: ignore

from aethergraph.config.dotenv_writer import write_dotenv
from aethergraph.core.runtime.runtime_services import current_services

from .deps import RequestIdentity, get_identity
from .schemas.settings import (
    EmbeddingProfilePayload,
    EmbeddingProfileView,
    LLMProfilePayload,
    LLMProfileView,
    SettingsGetResponse,
    SettingsStatusResponse,
    SettingsUpdateRequest,
    SlackPayload,
    SlackView,
    TelegramPayload,
    TelegramView,
    _is_masked,
    _mask_secret,
)

logger = logging.getLogger("aethergraph.api.settings")

router = APIRouter(prefix="/settings", tags=["settings"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_local(identity: RequestIdentity) -> None:
    if not identity.is_local:
        raise HTTPException(status_code=403, detail="Settings API is only available in local mode.")


def _get_settings():
    container = current_services()
    return container.settings


def _secret_str_value(secret) -> str | None:
    """Extract plain string from a SecretStr or return None."""
    if secret is None:
        return None
    return secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)


def _has_secret(secret) -> bool:
    """Return True if a SecretStr has a non-empty value."""
    val = _secret_str_value(secret)
    return bool(val)


# ---------------------------------------------------------------------------
# GET /settings/status
# ---------------------------------------------------------------------------


@router.get("/status", response_model=SettingsStatusResponse)
async def settings_status(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SettingsStatusResponse:
    _require_local(identity)
    cfg = _get_settings()

    llm_configured = _has_secret(cfg.llm.default.api_key)
    embedding_configured = _has_secret(cfg.embed.default.api_key)
    slack_configured = cfg.slack.enabled and _has_secret(cfg.slack.bot_token)
    telegram_configured = cfg.telegram.enabled and _has_secret(cfg.telegram.bot_token)

    return SettingsStatusResponse(
        workspace=str(Path(cfg.workspace).resolve()),
        deploy_mode=cfg.deploy_mode,
        llm_configured=llm_configured,
        embedding_configured=embedding_configured,
        slack_configured=slack_configured,
        telegram_configured=telegram_configured,
    )


# ---------------------------------------------------------------------------
# GET /settings
# ---------------------------------------------------------------------------


def _llm_profile_view(profile) -> LLMProfileView:
    return LLMProfileView(
        provider=profile.provider,
        model=profile.model,
        base_url=profile.base_url,
        timeout=profile.timeout,
        api_key=_mask_secret(_secret_str_value(profile.api_key)),
        thinking_budget=profile.thinking_budget,
        reasoning_summary=profile.reasoning_summary,
    )


def _embed_profile_view(profile) -> EmbeddingProfileView:
    return EmbeddingProfileView(
        provider=profile.provider,
        model=profile.model,
        base_url=profile.base_url,
        timeout=profile.timeout,
        api_key=_mask_secret(_secret_str_value(profile.api_key)),
    )


@router.get("", response_model=SettingsGetResponse)
async def get_settings(
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SettingsGetResponse:
    _require_local(identity)
    cfg = _get_settings()

    # LLM profiles
    llm_profiles: dict[str, LLMProfileView] = {"default": _llm_profile_view(cfg.llm.default)}
    for name, profile in cfg.llm.profiles.items():
        llm_profiles[name] = _llm_profile_view(profile)

    # Embedding profiles
    embed_profiles: dict[str, EmbeddingProfileView] = {
        "default": _embed_profile_view(cfg.embed.default)
    }
    for name, profile in cfg.embed.profiles.items():
        embed_profiles[name] = _embed_profile_view(profile)

    return SettingsGetResponse(
        workspace=str(Path(cfg.workspace).resolve()),
        deploy_mode=cfg.deploy_mode,
        llm=llm_profiles,
        embedding=embed_profiles,
        slack=SlackView(
            enabled=cfg.slack.enabled,
            bot_token=_mask_secret(_secret_str_value(cfg.slack.bot_token)),
            signing_secret=_mask_secret(_secret_str_value(cfg.slack.signing_secret)),
        ),
        telegram=TelegramView(
            enabled=cfg.telegram.enabled,
            bot_token=_mask_secret(_secret_str_value(cfg.telegram.bot_token)),
        ),
    )


# ---------------------------------------------------------------------------
# PUT /settings
# ---------------------------------------------------------------------------


def _env_key(*parts: str) -> str:
    """Build an AETHERGRAPH env var name from dot-path parts.

    Example: _env_key("LLM", "DEFAULT", "API_KEY") -> "AETHERGRAPH_LLM__DEFAULT__API_KEY"
    """
    return "AETHERGRAPH_" + "__".join(p.upper() for p in parts)


def _collect_llm_env(
    profiles: dict[str, LLMProfilePayload],
) -> dict[str, str]:
    """Convert LLM profile payloads to env var updates."""
    env: dict[str, str] = {}
    for name, payload in profiles.items():
        prefix = ("LLM", name)
        if payload.provider is not None:
            env[_env_key(*prefix, "PROVIDER")] = payload.provider
        if payload.model is not None:
            env[_env_key(*prefix, "MODEL")] = payload.model
        if payload.api_key is not None and not _is_masked(payload.api_key):
            env[_env_key(*prefix, "API_KEY")] = payload.api_key
        if payload.base_url is not None:
            env[_env_key(*prefix, "BASE_URL")] = payload.base_url
        if payload.timeout is not None:
            env[_env_key(*prefix, "TIMEOUT")] = str(payload.timeout)
    return env


def _collect_embed_env(
    profiles: dict[str, EmbeddingProfilePayload],
) -> dict[str, str]:
    env: dict[str, str] = {}
    for name, payload in profiles.items():
        prefix = ("EMBED", name)
        if payload.provider is not None:
            env[_env_key(*prefix, "PROVIDER")] = payload.provider
        if payload.model is not None:
            env[_env_key(*prefix, "MODEL")] = payload.model
        if payload.api_key is not None and not _is_masked(payload.api_key):
            env[_env_key(*prefix, "API_KEY")] = payload.api_key
        if payload.base_url is not None:
            env[_env_key(*prefix, "BASE_URL")] = payload.base_url
        if payload.timeout is not None:
            env[_env_key(*prefix, "TIMEOUT")] = str(payload.timeout)
    return env


def _collect_slack_env(payload: SlackPayload) -> dict[str, str]:
    env: dict[str, str] = {}
    if payload.enabled is not None:
        env[_env_key("SLACK", "ENABLED")] = str(payload.enabled).lower()
    if payload.bot_token is not None and not _is_masked(payload.bot_token):
        env[_env_key("SLACK", "BOT_TOKEN")] = payload.bot_token
    if payload.signing_secret is not None and not _is_masked(payload.signing_secret):
        env[_env_key("SLACK", "SIGNING_SECRET")] = payload.signing_secret
    return env


def _collect_telegram_env(payload: TelegramPayload) -> dict[str, str]:
    env: dict[str, str] = {}
    if payload.enabled is not None:
        env[_env_key("TELEGRAM", "ENABLED")] = str(payload.enabled).lower()
    if payload.bot_token is not None and not _is_masked(payload.bot_token):
        env[_env_key("TELEGRAM", "BOT_TOKEN")] = payload.bot_token
    return env


def _hot_reload_llm(profiles: dict[str, LLMProfilePayload]) -> None:
    """Apply LLM profile changes in-memory via LLMService.configure_profile."""
    container = current_services()
    llm_service = getattr(container, "llm", None)
    if llm_service is None:
        return

    for name, payload in profiles.items():
        kwargs: dict = {}
        if payload.provider is not None:
            kwargs["provider"] = payload.provider
        if payload.model is not None:
            kwargs["model"] = payload.model
        if payload.api_key is not None and not _is_masked(payload.api_key):
            kwargs["api_key"] = payload.api_key
        if payload.base_url is not None:
            kwargs["base_url"] = payload.base_url
        if payload.timeout is not None:
            kwargs["timeout"] = payload.timeout
        if kwargs:
            llm_service.configure_profile(profile=name, **kwargs)
            logger.info("Hot-reloaded LLM profile %r", name)


def _hot_reload_embedding(profiles: dict[str, EmbeddingProfilePayload]) -> None:
    """Apply embedding profile changes in-memory."""
    container = current_services()
    embed_service = getattr(container, "embed_service", None)
    if embed_service is None:
        return

    for name, payload in profiles.items():
        kwargs: dict = {}
        if payload.provider is not None:
            kwargs["provider"] = payload.provider
        if payload.model is not None:
            kwargs["model"] = payload.model
        if payload.api_key is not None and not _is_masked(payload.api_key):
            kwargs["api_key"] = payload.api_key
        if payload.base_url is not None:
            kwargs["base_url"] = payload.base_url
        if payload.timeout is not None:
            kwargs["timeout"] = payload.timeout
        if kwargs:
            embed_service.configure_profile(name=name, **kwargs)
            logger.info("Hot-reloaded embedding profile %r", name)


def _hot_reload_slack(payload: SlackPayload) -> None:
    """Update Slack settings in-memory (adapter reconnect requires restart)."""
    cfg = _get_settings()
    if payload.enabled is not None:
        cfg.slack.enabled = payload.enabled
    if payload.bot_token is not None and not _is_masked(payload.bot_token):
        from pydantic import SecretStr

        cfg.slack.bot_token = SecretStr(payload.bot_token)
    if payload.signing_secret is not None and not _is_masked(payload.signing_secret):
        from pydantic import SecretStr

        cfg.slack.signing_secret = SecretStr(payload.signing_secret)


def _hot_reload_telegram(payload: TelegramPayload) -> None:
    """Update Telegram settings in-memory (adapter reconnect requires restart)."""
    cfg = _get_settings()
    if payload.enabled is not None:
        cfg.telegram.enabled = payload.enabled
    if payload.bot_token is not None and not _is_masked(payload.bot_token):
        from pydantic import SecretStr

        cfg.telegram.bot_token = SecretStr(payload.bot_token)


@router.put("", response_model=SettingsGetResponse)
async def update_settings(
    body: SettingsUpdateRequest,
    identity: RequestIdentity = Depends(get_identity),  # noqa: B008
) -> SettingsGetResponse:
    _require_local(identity)
    cfg = _get_settings()

    # 1) Collect env var updates
    env_updates: dict[str, str] = {}
    if body.llm:
        env_updates.update(_collect_llm_env(body.llm))
    if body.embedding:
        env_updates.update(_collect_embed_env(body.embedding))
    if body.slack:
        env_updates.update(_collect_slack_env(body.slack))
    if body.telegram:
        env_updates.update(_collect_telegram_env(body.telegram))

    # 2) Persist to workspace .env
    if env_updates:
        workspace_env = Path(cfg.workspace).resolve() / ".env"
        write_dotenv(workspace_env, env_updates)
        logger.info("Wrote %d env vars to %s", len(env_updates), workspace_env)

    # 3) Hot-reload in-memory services
    if body.llm:
        _hot_reload_llm(body.llm)
    if body.embedding:
        _hot_reload_embedding(body.embedding)
    if body.slack:
        _hot_reload_slack(body.slack)
    if body.telegram:
        _hot_reload_telegram(body.telegram)

    # 4) Return updated settings view
    return await get_settings(identity=identity)

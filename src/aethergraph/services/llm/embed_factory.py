# aethergraph/services/llm/embedding_factory.py
from __future__ import annotations

import logging
import os

from pydantic import SecretStr

from aethergraph.config.llm import EmbeddingProfile, EmbeddingSettings
from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient
from aethergraph.services.llm.provider_transport import ProviderRateGate
from aethergraph.services.metering.eventlog_metering import MeteringService

from ..secrets.base import Secrets
from .compat import embedding_profile_from_legacy
from .credentials import resolve_provider_credential
from .profiles import EmbeddingProfileSpec
from .registry import provider_default_base_url


def _apply_env_overrides_to_embed_profile(
    name: str,
    p: EmbeddingProfile,
    *,
    is_default: bool,
    secrets: Secrets,
) -> EmbeddingProfile:
    """
    Mutate + return profile with env-based overrides.

    - For the default embedding profile, allow generic EMBED_* env vars
      (and fall back to LLM_* for smoother migration).
    - For all profiles, fill missing base_url / api_key from provider-specific env.
    """
    # 1) Generic overrides for DEFAULT embedding profile
    if is_default:
        provider_env = os.getenv("EMBED_PROVIDER") or os.getenv("LLM_PROVIDER")
        # MIGRATION: allow old LLM_EMBED_MODEL env to still work
        model_env = os.getenv("EMBED_MODEL") or os.getenv("LLM_EMBED_MODEL")
        base_env = os.getenv("EMBED_BASE_URL") or os.getenv("LLM_BASE_URL")
        timeout_env = os.getenv("EMBED_TIMEOUT")

        if provider_env:
            p.provider = provider_env.lower()  # type: ignore[assignment]
        if model_env:
            p.model = model_env
        if base_env:
            p.base_url = base_env
        if timeout_env:
            try:
                p.timeout = float(timeout_env)
            except ValueError:
                logger = logging.getLogger("aethergraph.services.llm")
                logger.warning(f"Invalid EMBED_TIMEOUT value: {timeout_env}")

    # 2) Provider-specific base_url fallback
    if not p.base_url:
        p.base_url = provider_default_base_url(p.provider)

    # 3) API key resolution:
    #    - prefer explicit api_key on profile
    #    - else api_key_ref + Secrets
    #    - else provider-specific env name
    credential = resolve_provider_credential(
        provider_id=p.provider,
        direct=p.api_key,
        secret_ref=p.api_key_ref,
        secrets=secrets,
    )
    api_key = credential.value
    if api_key and not p.api_key_ref and credential.source_ref:
        p.api_key_ref = credential.source_ref

    if api_key:
        p.api_key = SecretStr(api_key)

    return p


def embed_client_from_profile(
    p: EmbeddingProfileSpec,
    secrets: Secrets,
    *,
    metering: MeteringService | None = None,
    rate_gate: ProviderRateGate | None = None,
) -> GenericEmbeddingClient:
    """Build one embedding client from a canonical embedding profile.

    Intro:
        Embedding construction consumes its own operation contract and never
        inherits Chat model or capability configuration.

    Examples:
        Build a default embedding client:
            ```python
            client = embed_client_from_profile(canonical_profile, secrets)
            ```

        Attach shared controls:
            ```python
            client = embed_client_from_profile(
                canonical_profile, secrets, metering=metering, rate_gate=gate
            )
            ```

    Args:
        p: Canonical immutable embedding profile.
        secrets: Secret store used for an exact configured reference.
        metering: Optional shared embedding metering service.
        rate_gate: Optional shared provider quota gate.

    Returns:
        GenericEmbeddingClient: Configured provider-neutral embedding client.

    Notes:
        Endpoint selection has already occurred in the compatibility codec.
    """

    api_key = resolve_provider_credential(
        provider_id=p.connection.provider_id,
        direct=p.credentials.inline_secret,
        secret_ref=p.credentials.secret_ref,
        secrets=secrets,
    ).value

    return GenericEmbeddingClient(
        provider=p.connection.provider_id,
        model=p.model.model_id,
        base_url=p.connection.base_url,
        api_key=api_key,
        azure_deployment=p.connection.deployment,
        timeout=p.transport.timeout_s,
        retry_settings=p.transport.retry,
        rate_limit_group=p.transport.rate_limit_group,
        rate_gate=rate_gate,
        metering=metering,
    )


def build_embedding_clients(
    cfg: EmbeddingSettings,
    secrets: Secrets,
    *,
    metering: MeteringService | None = None,
    rate_gate: ProviderRateGate | None = None,
) -> dict[str, GenericEmbeddingClient]:
    """Build all enabled embedding clients through their canonical boundary.

    Intro:
        Public embedding settings retain environment compatibility, then each
        profile is projected once into the separate embedding contract.

    Examples:
        Build enabled embedding clients:
            ```python
            clients = build_embedding_clients(settings, secrets)
            assert "default" in clients
            ```

        Respect disabled settings:
            ```python
            clients = build_embedding_clients(disabled_settings, secrets)
            assert clients == {}
            ```

    Args:
        cfg: Public legacy-compatible embedding settings.
        secrets: Secret store for configured credential references.
        metering: Optional shared embedding metering service.
        rate_gate: Optional shared provider quota gate.

    Returns:
        dict[str, GenericEmbeddingClient]: Clients keyed by profile name.

    Notes:
        One rate gate is shared across the clients produced by this call.
    """

    if not cfg.enabled:
        return {}

    shared_rate_gate = rate_gate or ProviderRateGate()

    # Default profile
    default_profile = _apply_env_overrides_to_embed_profile(
        name="default",
        p=cfg.default,
        is_default=True,
        secrets=secrets,
    )
    clients: dict[str, GenericEmbeddingClient] = {
        "default": embed_client_from_profile(
            embedding_profile_from_legacy(default_profile),
            secrets,
            metering=metering,
            rate_gate=shared_rate_gate,
        )
    }

    # Extra profiles
    for name, prof in (cfg.profiles or {}).items():
        prof = _apply_env_overrides_to_embed_profile(
            name=name,
            p=prof,
            is_default=False,
            secrets=secrets,
        )
        clients[name] = embed_client_from_profile(
            embedding_profile_from_legacy(prof),
            secrets,
            metering=metering,
            rate_gate=shared_rate_gate,
        )

    return clients

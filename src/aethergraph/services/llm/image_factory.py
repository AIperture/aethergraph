"""Canonical image-generation client construction."""

from __future__ import annotations

from aethergraph.config.config import ImageGenerationUsageQuotaSettings
from aethergraph.config.llm import ImageGenerationSettings
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.server.security.credentials import SecretStore
from aethergraph.services.llm.compat import image_generation_profile_from_settings
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.generic_image_client import GenericImageGenerationClient
from aethergraph.services.llm.profiles import ImageGenerationProfile
from aethergraph.services.llm.provider_transport import ProviderRateGate


def image_client_from_profile(
    profile: ImageGenerationProfile,
    secrets: SecretStore,
    *,
    metering: MeteringService | None = None,
    rate_gate: ProviderRateGate | None = None,
    profile_name: str | None = None,
    operation_quota_cfg: ImageGenerationUsageQuotaSettings | None = None,
) -> GenericImageGenerationClient:
    """Build one image client from a canonical operation profile.

    Intro:
        Consumes only image-generation connection, model, credential, transport,
        and request-default state after exact endpoint selection.

    Examples:
        Build a default client:
            ```python
            client = image_client_from_profile(canonical_profile, secrets)
            ```

        Attach shared controls and identity:
            ```python
            client = image_client_from_profile(
                canonical_profile,
                secrets,
                metering=metering,
                rate_gate=gate,
                profile_name="design",
            )
            ```

    Args:
        profile: Canonical immutable image-generation profile.
        secrets: Secret store used for an exact configured reference.
        metering: Optional shared model metering service.
        rate_gate: Optional container-shared provider rate gate.
        profile_name: Optional configured profile identity.
        operation_quota_cfg: Optional infrastructure-owned per-run image quota
            policy.

    Returns:
        GenericImageGenerationClient: Exact-bound independent image client.

    Notes:
        No Chat profile or client participates in construction.
    """

    api_key = resolve_provider_credential(
        provider_id=profile.connection.provider_id,
        direct=profile.credentials.inline_secret,
        secret_ref=profile.credentials.secret_ref,
        secrets=secrets,
    ).value
    return GenericImageGenerationClient(
        provider=profile.connection.provider_id,
        model=profile.model.model_id,
        endpoint_id=profile.connection.endpoint_id,
        base_url=profile.connection.base_url,
        api_key=api_key,
        azure_deployment=profile.connection.deployment,
        timeout=profile.transport.timeout_s,
        retry_settings=profile.transport.retry,
        rate_limit_group=profile.transport.rate_limit_group,
        rate_gate=rate_gate,
        metering=metering,
        operation_quota_cfg=operation_quota_cfg,
        default_count=profile.defaults.count,
        default_size=profile.defaults.size,
        default_quality=profile.defaults.quality,
        default_output_format=profile.defaults.output_format,
        default_response_format=profile.defaults.response_format,
        default_background=profile.defaults.background,
        profile_name=profile_name,
    )


def build_image_generation_clients(
    settings: ImageGenerationSettings,
    secrets: SecretStore,
    *,
    metering: MeteringService | None = None,
    rate_gate: ProviderRateGate | None = None,
    operation_quota_cfg: ImageGenerationUsageQuotaSettings | None = None,
) -> dict[str, GenericImageGenerationClient]:
    """Build every enabled image client through the canonical profile boundary.

    Intro:
        Projects public startup settings once, shares one provider rate gate, and
        returns no clients when the operation is disabled.

    Examples:
        Build enabled image clients:
            ```python
            clients = build_image_generation_clients(settings, secrets)
            assert "default" in clients
            ```

        Respect disabled settings:
            ```python
            clients = build_image_generation_clients(disabled_settings, secrets)
            assert clients == {}
            ```

    Args:
        settings: Public image-generation settings.
        secrets: Secret store for configured credential references.
        metering: Optional shared model metering service.
        rate_gate: Optional container-shared provider rate gate.
        operation_quota_cfg: Optional infrastructure-owned per-run image quota
            policy shared by every profile.

    Returns:
        dict[str, GenericImageGenerationClient]: Clients keyed by profile name.

    Notes:
        Endpoint validation fails closed during construction before graph work.
    """

    if not settings.enabled:
        return {}
    shared_rate_gate = rate_gate or ProviderRateGate()
    profiles = {"default": settings.default, **dict(settings.profiles or {})}
    return {
        name: image_client_from_profile(
            image_generation_profile_from_settings(profile),
            secrets,
            metering=metering,
            rate_gate=shared_rate_gate,
            profile_name=name,
            operation_quota_cfg=operation_quota_cfg,
        )
        for name, profile in profiles.items()
    }

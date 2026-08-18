"""Legacy Chat-client image assignment boundary."""

from __future__ import annotations

from typing import Any

from aethergraph.services.llm.generic_image_client import GenericImageGenerationClient
from aethergraph.services.llm.registry import resolve_endpoint_adapter
from aethergraph.services.llm.types import LLMUnsupportedFeatureError


def image_client_from_legacy_chat(chat_client: Any) -> GenericImageGenerationClient:
    """Project one legacy Chat connection into an exact image client.

    Intro:
        Preserves direct `GenericLLMClient.generate_image()` construction by
        translating the Chat connection once into an independent image client.

    Examples:
        Project an OpenAI client:
            ```python
            image_client = image_client_from_legacy_chat(chat_client)
            ```

        Reject a provider without image generation:
            ```python
            try:
                image_client_from_legacy_chat(anthropic_chat_client)
            except LLMUnsupportedFeatureError:
                pass
            ```

    Args:
        chat_client: Legacy Chat client supplying provider connection and shared
            infrastructure policy.

    Returns:
        GenericImageGenerationClient: Independently owned exact image client.

    Notes:
        This is a compatibility codec, not runtime fallback selection. Container-
        managed clients receive explicit image assignments and do not use it.
    """

    provider = str(chat_client.provider or "").strip().lower()
    try:
        endpoint = resolve_endpoint_adapter(provider, "image_generation")
    except ValueError as exc:
        detail = (
            "Anthropic does not support image generation via Claude API (vision is input-only)."
            if provider == "anthropic"
            else f"provider '{provider}' does not support generate_image() in this client."
        )
        raise LLMUnsupportedFeatureError(
            provider,
            getattr(chat_client, "model", None),
            "image generation",
            detail,
        ) from exc

    retry = getattr(chat_client, "_provider_retry", None)
    return GenericImageGenerationClient(
        provider=provider,
        model=str(chat_client.model),
        endpoint_id=endpoint.adapter_id,
        base_url=chat_client.base_url,
        api_key=chat_client.api_key,
        azure_deployment=chat_client.azure_deployment,
        timeout=float(chat_client._timeout),
        retry_settings=getattr(retry, "settings", None),
        rate_limit_group=chat_client.rate_limit_group,
        rate_gate=getattr(retry, "rate_gate", None),
        metering=chat_client.metering,
        profile_name=(
            f"legacy-chat:{chat_client.profile_name}" if chat_client.profile_name else "legacy-chat"
        ),
    )

"""Legacy flat-profile projection into canonical operation profiles."""

from __future__ import annotations

from aethergraph.config.llm import (
    EmbeddingProfile,
    ImageGenerationProfileSettings,
    LLMProfile,
)

from ..profiles import (
    ChatCapabilityOverrides,
    ChatDefaults,
    ChatProfile,
    CredentialSelection,
    EmbeddingProfileSpec,
    ImageGenerationDefaults,
    ImageGenerationProfile,
    ModelSelection,
    MultimodalInputPolicy,
    ProviderConnection,
    TransportPolicy,
)
from ..registry import resolve_endpoint_adapter


def chat_profile_from_legacy(
    profile: LLMProfile,
    *,
    endpoint_id: str | None = None,
) -> ChatProfile:
    """Project one public flat `LLMProfile` into canonical Chat configuration.

    Intro:
        The boundary codec preserves existing connection, request defaults,
        transport, and image policy while separating model capability assertions.
        It does not mutate the public input profile.

    Examples:
        Convert a default OpenAI profile:
            ```python
            canonical = chat_profile_from_legacy(LLMProfile())
            assert canonical.operation == "chat"
            ```

        Pin an explicit endpoint:
            ```python
            canonical = chat_profile_from_legacy(
                LLMProfile(provider="azure", model="deployment"),
                endpoint_id="azure_responses",
            )
            ```

    Args:
        profile: Existing public flat Chat profile.
        endpoint_id: Optional explicit endpoint adapter selection.

    Returns:
        ChatProfile: Immutable canonical operation-specific profile.

    Notes:
        Legacy `embed_model` is deliberately not copied into the Chat profile.
        It remains available only to the separate embedding migration codec.
    """

    selected_endpoint_id = endpoint_id or profile.endpoint_id
    adapter = resolve_endpoint_adapter(
        profile.provider,
        "chat",
        endpoint_id=selected_endpoint_id,
    )
    credentials = (
        CredentialSelection(inline_secret=profile.api_key)
        if profile.api_key is not None
        else CredentialSelection(secret_ref=profile.api_key_ref)
    )
    return ChatProfile(
        connection=ProviderConnection(
            provider_id=profile.provider,
            endpoint_id=adapter.adapter_id,
            base_url=profile.base_url,
            deployment=profile.azure_deployment,
        ),
        model=ModelSelection(model_id=profile.model),
        credentials=credentials,
        transport=TransportPolicy(
            timeout_s=profile.timeout,
            retry=profile.retry,
            rate_limit_group=profile.rate_limit_group,
        ),
        defaults=ChatDefaults(
            reasoning_effort=profile.reasoning_effort,
            thinking_mode=profile.thinking_mode,
            thinking_budget=profile.thinking_budget,
            reasoning_summary=profile.reasoning_summary,
            compatibility_policy=profile.compatibility_policy,
            structured_output_policy=profile.structured_output_policy,
            prompt_cache_policy=profile.prompt_cache_policy,
            context_window_tokens=profile.context_window_tokens,
        ),
        input_policy=MultimodalInputPolicy(
            image_input_enabled=profile.vision_enabled,
            allow_remote_urls=profile.vision_enabled,
            max_images=profile.vision_max_images,
            max_image_bytes=profile.vision_max_image_bytes,
            accepted_mime_prefixes=tuple(profile.vision_accepted_mime_prefixes),
            accepted_mime_types=tuple(profile.vision_accepted_mime_types),
            resize_enabled=profile.vision_resize_enabled,
            resize_max_dimension=profile.vision_resize_max_dimension,
            resize_max_pixels=profile.vision_resize_max_pixels,
            jpeg_quality=profile.vision_resize_jpeg_quality,
            min_jpeg_quality=profile.vision_resize_min_jpeg_quality,
        ),
        capability_overrides=ChatCapabilityOverrides(
            image_input="supported" if profile.vision_enabled else "unknown"
        ),
    )


def embedding_profile_from_legacy(
    profile: EmbeddingProfile,
    *,
    endpoint_id: str | None = None,
) -> EmbeddingProfileSpec:
    """Project one public embedding profile into its canonical contract.

    Intro:
        The boundary codec selects one endpoint before invocation and preserves
        transport and credential fields without involving Chat configuration.

    Examples:
        Convert the default embedding profile:
            ```python
            canonical = embedding_profile_from_legacy(EmbeddingProfile())
            assert canonical.operation == "embeddings"
            ```

        Convert an Azure deployment:
            ```python
            canonical = embedding_profile_from_legacy(
                EmbeddingProfile(provider="azure", azure_deployment="embed-prod")
            )
            ```

    Args:
        profile: Existing public flat embedding profile.
        endpoint_id: Optional explicit endpoint adapter selection.

    Returns:
        EmbeddingProfileSpec: Immutable canonical embedding profile.

    Notes:
        This codec performs no secret-store or environment resolution.
    """

    adapter = resolve_endpoint_adapter(
        profile.provider,
        "embeddings",
        endpoint_id=endpoint_id or profile.endpoint_id,
    )
    credentials = (
        CredentialSelection(inline_secret=profile.api_key)
        if profile.api_key is not None
        else CredentialSelection(secret_ref=profile.api_key_ref)
    )
    return EmbeddingProfileSpec(
        connection=ProviderConnection(
            provider_id=profile.provider,
            endpoint_id=adapter.adapter_id,
            base_url=profile.base_url,
            deployment=profile.azure_deployment,
        ),
        model=ModelSelection(model_id=profile.model),
        credentials=credentials,
        transport=TransportPolicy(
            timeout_s=profile.timeout,
            retry=profile.retry,
            rate_limit_group=profile.rate_limit_group,
        ),
    )


def image_generation_profile_from_settings(
    profile: ImageGenerationProfileSettings,
    *,
    endpoint_id: str | None = None,
) -> ImageGenerationProfile:
    """Project one public image profile into its canonical contract.

    Intro:
        Selects one exact image endpoint and separates image defaults from Chat
        configuration before runtime client construction.

    Examples:
        Convert the default image profile:
            ```python
            canonical = image_generation_profile_from_settings(
                ImageGenerationProfileSettings()
            )
            assert canonical.operation == "image_generation"
            ```

        Convert an Azure deployment:
            ```python
            canonical = image_generation_profile_from_settings(
                ImageGenerationProfileSettings(
                    provider="azure",
                    model="image-deployment",
                    azure_deployment="image-deployment",
                )
            )
            ```

    Args:
        profile: Public image-generation profile settings.
        endpoint_id: Optional explicit endpoint adapter overriding the profile.

    Returns:
        ImageGenerationProfile: Immutable canonical image-generation profile.

    Notes:
        This codec performs no secret-store or environment resolution and never
        reads a Chat profile.
    """

    adapter = resolve_endpoint_adapter(
        profile.provider,
        "image_generation",
        endpoint_id=endpoint_id or profile.endpoint_id,
    )
    credentials = (
        CredentialSelection(inline_secret=profile.api_key)
        if profile.api_key is not None
        else CredentialSelection(secret_ref=profile.api_key_ref)
    )
    return ImageGenerationProfile(
        connection=ProviderConnection(
            provider_id=profile.provider,
            endpoint_id=adapter.adapter_id,
            base_url=profile.base_url,
            deployment=profile.azure_deployment,
        ),
        model=ModelSelection(model_id=profile.model),
        credentials=credentials,
        transport=TransportPolicy(
            timeout_s=profile.timeout,
            retry=profile.retry,
            rate_limit_group=profile.rate_limit_group,
        ),
        defaults=ImageGenerationDefaults(
            count=profile.count,
            size=profile.size,
            quality=profile.quality,
            output_format=profile.output_format,
            response_format=profile.response_format,
            background=profile.background,
        ),
    )


__all__ = [
    "chat_profile_from_legacy",
    "embedding_profile_from_legacy",
    "image_generation_profile_from_settings",
]

"""Immutable operation-specific model profile contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from .provider_transport import ProviderRetrySettings

CapabilityState = Literal["supported", "unsupported", "unknown"]
PromptCachePolicy = Literal["disabled", "auto", "required"]


class ProfileContract(BaseModel):
    """Base class for closed immutable canonical profile records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConnection(ProfileContract):
    """Provider and endpoint identity selected before model invocation."""

    provider_id: str = Field(min_length=1, max_length=128)
    endpoint_id: str = Field(min_length=1, max_length=128)
    base_url: str | None = None
    deployment: str | None = None


class ModelSelection(ProfileContract):
    """Provider model identity with an optional pinned catalog entry."""

    model_id: str = Field(min_length=1, max_length=512)
    catalog_key: str | None = Field(default=None, min_length=1, max_length=512)


class CredentialSelection(ProfileContract):
    """One secret reference or inline compatibility-boundary credential."""

    secret_ref: str | None = Field(default=None, min_length=1, max_length=512)
    inline_secret: SecretStr | None = None

    @model_validator(mode="after")
    def _require_one_selection(self) -> CredentialSelection:
        """Reject ambiguous credential precedence inside canonical profiles.

        Intro:
            Canonical profiles select at most one credential source so secret
            precedence is handled once by the legacy boundary codec.

        Examples:
            Select a secret reference:
                ```python
                selection = CredentialSelection(secret_ref="OPENAI_API_KEY")
                ```

            Reject two sources:
                ```python
                try:
                    CredentialSelection(
                        secret_ref="OPENAI_API_KEY",
                        inline_secret=SecretStr("value"),
                    )
                except ValueError:
                    pass
                ```

        Args:
            self: Fully parsed credential selection.

        Returns:
            CredentialSelection: Unchanged unambiguous selection.

        Notes:
            Empty credentials are valid for local unauthenticated providers.
        """

        if self.secret_ref is not None and self.inline_secret is not None:
            raise ValueError("canonical credentials require secret_ref or inline_secret, not both")
        return self


class TransportPolicy(ProfileContract):
    """Shared timeout, retry, and rate-gate selection for one profile."""

    timeout_s: float = Field(default=60.0, gt=0, le=604_800)
    retry: ProviderRetrySettings = Field(default_factory=ProviderRetrySettings)
    rate_limit_group: str | None = Field(default=None, min_length=1, max_length=256)


class ChatDefaults(ProfileContract):
    """Request defaults that do not assert model capability truth."""

    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    thinking_mode: Literal["auto", "on", "off"] | None = None
    thinking_budget: int | None = Field(default=None, ge=0)
    reasoning_summary: str | None = None
    structured_output_policy: Literal["best_available", "native_required"] = "best_available"
    prompt_cache_policy: PromptCachePolicy = "auto"


class MultimodalInputPolicy(ProfileContract):
    """Application-owned image admission and normalization limits."""

    image_input_enabled: bool = False
    max_images: int | None = Field(default=None, ge=1)
    max_image_bytes: int | None = Field(default=None, ge=1)
    accepted_mime_prefixes: tuple[str, ...] = ("image/",)
    accepted_mime_types: tuple[str, ...] = ()
    resize_enabled: bool = True
    resize_max_dimension: int = Field(default=1280, ge=1)
    resize_max_pixels: int = Field(default=1_500_000, ge=1)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    min_jpeg_quality: int = Field(default=70, ge=1, le=100)
    allow_remote_urls: bool = False

    @model_validator(mode="after")
    def _validate_quality_range(self) -> MultimodalInputPolicy:
        """Require the minimum JPEG quality not to exceed the initial quality.

        Intro:
            Image normalization can reduce quality only within the configured
            closed interval.

        Examples:
            Validate the defaults:
                ```python
                policy = MultimodalInputPolicy()
                assert policy.min_jpeg_quality <= policy.jpeg_quality
                ```

            Reject an inverted range:
                ```python
                try:
                    MultimodalInputPolicy(jpeg_quality=70, min_jpeg_quality=80)
                except ValueError:
                    pass
                ```

        Args:
            self: Fully parsed multimodal input policy.

        Returns:
            MultimodalInputPolicy: Unchanged policy with a valid quality range.

        Notes:
            Remote URL fetching remains disabled by default.
        """

        if self.min_jpeg_quality > self.jpeg_quality:
            raise ValueError("min_jpeg_quality must not exceed jpeg_quality")
        return self


class ChatCapabilityOverrides(ProfileContract):
    """User assertions about model facts, expressed as tri-state overrides."""

    image_input: CapabilityState = "unknown"
    streaming: CapabilityState = "unknown"
    native_tool_calling: CapabilityState = "unknown"
    tool_result_continuation: CapabilityState = "unknown"
    parallel_tool_calls: CapabilityState = "unknown"
    structured_output: CapabilityState = "unknown"
    prompt_cache: CapabilityState = "unknown"
    native_tool_search_hosted: CapabilityState = "unknown"
    native_tool_search_client: CapabilityState = "unknown"


class ChatProfile(ProfileContract):
    """Canonical Chat profile with separated connection, policy, and facts."""

    operation: Literal["chat"] = "chat"
    connection: ProviderConnection
    model: ModelSelection
    credentials: CredentialSelection = Field(default_factory=CredentialSelection)
    transport: TransportPolicy = Field(default_factory=TransportPolicy)
    defaults: ChatDefaults = Field(default_factory=ChatDefaults)
    input_policy: MultimodalInputPolicy = Field(default_factory=MultimodalInputPolicy)
    capability_overrides: ChatCapabilityOverrides = Field(default_factory=ChatCapabilityOverrides)


class EmbeddingDefaults(ProfileContract):
    """Embedding request defaults independent from model capabilities."""

    dimensions: int | None = Field(default=None, ge=1)
    batch_size: int | None = Field(default=None, ge=1)


class EmbeddingCapabilityOverrides(ProfileContract):
    """User assertions about embedding model capability facts."""

    text_embeddings: CapabilityState = "unknown"
    dimensions: CapabilityState = "unknown"


class EmbeddingProfileSpec(ProfileContract):
    """Canonical embedding profile separated from Chat configuration."""

    operation: Literal["embeddings"] = "embeddings"
    connection: ProviderConnection
    model: ModelSelection
    credentials: CredentialSelection = Field(default_factory=CredentialSelection)
    transport: TransportPolicy = Field(default_factory=TransportPolicy)
    defaults: EmbeddingDefaults = Field(default_factory=EmbeddingDefaults)
    capability_overrides: EmbeddingCapabilityOverrides = Field(
        default_factory=EmbeddingCapabilityOverrides
    )


class ImageGenerationDefaults(ProfileContract):
    """Image-generation request defaults independent from model capabilities."""

    count: int = Field(default=1, ge=1)
    size: str | None = None
    quality: str | None = None
    output_format: str | None = None
    background: str | None = None


class ImageGenerationCapabilityOverrides(ProfileContract):
    """User assertions about image-generation capability facts."""

    text_to_image: CapabilityState = "unknown"
    image_editing: CapabilityState = "unknown"
    multiple_outputs: CapabilityState = "unknown"


class ImageGenerationProfile(ProfileContract):
    """Canonical image-generation profile independent from Chat clients."""

    operation: Literal["image_generation"] = "image_generation"
    connection: ProviderConnection
    model: ModelSelection
    credentials: CredentialSelection = Field(default_factory=CredentialSelection)
    transport: TransportPolicy = Field(default_factory=TransportPolicy)
    defaults: ImageGenerationDefaults = Field(default_factory=ImageGenerationDefaults)
    capability_overrides: ImageGenerationCapabilityOverrides = Field(
        default_factory=ImageGenerationCapabilityOverrides
    )


__all__ = [
    "CapabilityState",
    "ChatCapabilityOverrides",
    "ChatDefaults",
    "ChatProfile",
    "CredentialSelection",
    "EmbeddingCapabilityOverrides",
    "EmbeddingDefaults",
    "EmbeddingProfileSpec",
    "ImageGenerationCapabilityOverrides",
    "ImageGenerationDefaults",
    "ImageGenerationProfile",
    "ModelSelection",
    "MultimodalInputPolicy",
    "ProfileContract",
    "PromptCachePolicy",
    "ProviderConnection",
    "TransportPolicy",
]

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from aethergraph.services.llm.provider_transport import ProviderRetrySettings
from aethergraph.services.llm.providers import Provider


class LLMProfile(BaseModel):
    provider: Provider = "openai"
    model: str = "gpt-4o-mini"
    endpoint_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Optional explicit registered Chat endpoint adapter. Omission keeps "
            "the legacy 0.1.x provider routing compatibility boundary."
        ),
    )
    embed_model: str | None = None  # separate embedding model
    base_url: str | None = None
    timeout: float = 60.0
    retry: ProviderRetrySettings = Field(default_factory=ProviderRetrySettings)
    rate_limit_group: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Optional shared provider quota bucket used by the container rate gate.",
    )
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    thinking_mode: Literal["auto", "on", "off"] | None = None
    compatibility_policy: Literal["compat", "strict"] = Field(
        default="compat",
        description="How to handle requested capabilities that are not natively supported by the provider.",
    )
    structured_output_policy: Literal["best_available", "native_required"] = Field(
        default="best_available",
        description=(
            "Select the strongest safe structured-output mode or require native schema enforcement."
        ),
    )
    prompt_cache_policy: Literal["disabled", "auto", "required"] = Field(
        default="auto",
        description=(
            "Disable cache directives, use them when requested and supported, "
            "or require an explicit supported stable-prefix cache request."
        ),
    )
    context_window_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Explicit model context-window capacity. When omitted, AG records "
            "request estimates but leaves context admission to the provider."
        ),
    )

    # provider-specific
    azure_deployment: str | None = None

    # secrets (either direct value or ref name)
    api_key: SecretStr | None = None
    api_key_ref: str | None = Field(
        default=None, description="Name in secret store, e.g. 'OPENAI_API_KEY'"
    )

    # thinking / reasoning
    thinking_budget: int | None = Field(
        default=4096,
        description="Anthropic extended thinking budget_tokens. Enables thinking when set.",
    )
    reasoning_summary: str | None = Field(
        default="auto",
        description="OpenAI reasoning summary mode: 'auto' or 'concise'. Enables reasoning summaries when set.",
    )

    # explicit multimodal capability metadata
    vision_enabled: bool = Field(
        default=False,
        description="Whether this profile's loaded model is allowed to receive image inputs.",
    )
    vision_max_images: int | None = Field(
        default=None,
        description="Maximum image attachments a vision tool may send for this profile.",
    )
    vision_max_image_bytes: int | None = Field(
        default=None,
        description="Maximum bytes per hydrated image for this profile.",
    )
    vision_resize_enabled: bool = Field(
        default=True,
        description="Whether vision tools should downsample image inputs before LLM calls.",
    )
    vision_resize_max_dimension: int = Field(
        default=1280,
        description="Maximum width or height, in pixels, for resized vision images.",
    )
    vision_resize_max_pixels: int = Field(
        default=1_500_000,
        description="Maximum total pixel count for resized vision images.",
    )
    vision_resize_jpeg_quality: int = Field(
        default=85,
        description="Initial JPEG quality used when encoding resized vision images.",
    )
    vision_resize_min_jpeg_quality: int = Field(
        default=70,
        description="Lowest JPEG quality used while fitting resized vision images.",
    )
    vision_accepted_mime_prefixes: list[str] = Field(
        default_factory=lambda: ["image/"],
        description="Accepted MIME prefixes for image inputs.",
    )
    vision_accepted_mime_types: list[str] = Field(
        default_factory=list,
        description="Accepted exact MIME types for image inputs.",
    )


class LLMObservabilitySettings(BaseModel):
    capture_mode: Literal["off", "metadata", "manifest", "full"] = "manifest"


class LLMSettings(BaseModel):
    enabled: bool = True
    default: LLMProfile = LLMProfile()
    profiles: dict[str, LLMProfile] = Field(default_factory=dict)
    observability: LLMObservabilitySettings = LLMObservabilitySettings()


class EmbeddingProfile(BaseModel):
    provider: Provider = "openai"
    model: str = "text-embedding-3-small"
    endpoint_id: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: str | None = None
    timeout: float = 60.0
    retry: ProviderRetrySettings = Field(default_factory=ProviderRetrySettings)
    rate_limit_group: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Optional shared provider quota bucket used by the container rate gate.",
    )

    # provider-specific
    azure_deployment: str | None = None

    # secrets (either direct value or ref name)
    api_key: SecretStr | None = None
    api_key_ref: str | None = Field(
        default=None, description="Name in secret store, e.g. 'OPENAI_API_KEY'"
    )


class EmbeddingSettings(BaseModel):
    enabled: bool = True
    default: EmbeddingProfile = EmbeddingProfile()
    profiles: dict[str, EmbeddingProfile] = Field(default_factory=dict)


class ImageGenerationProfileSettings(BaseModel):
    provider: Provider = "openai"
    model: str = "gpt-image-1"
    endpoint_id: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: str | None = None
    timeout: float = 60.0
    retry: ProviderRetrySettings = Field(default_factory=ProviderRetrySettings)
    rate_limit_group: str | None = Field(default=None, min_length=1, max_length=256)
    azure_deployment: str | None = None
    api_key: SecretStr | None = None
    api_key_ref: str | None = Field(
        default=None, description="Name in secret store, e.g. 'OPENAI_API_KEY'"
    )
    count: int = Field(default=1, ge=1)
    size: str | None = None
    quality: str | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    response_format: Literal["b64_json", "url"] | None = None
    background: str | None = None


class ImageGenerationSettings(BaseModel):
    enabled: bool = True
    default: ImageGenerationProfileSettings = ImageGenerationProfileSettings()
    profiles: dict[str, ImageGenerationProfileSettings] = Field(default_factory=dict)

from typing import Literal

from pydantic import BaseModel, Field, SecretStr

from aethergraph.services.llm.providers import Provider


class LLMProfile(BaseModel):
    provider: Provider = "openai"
    model: str = "gpt-4o-mini"
    embed_model: str | None = None  # separate embedding model
    base_url: str | None = None
    timeout: float = 60.0

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


class LLMObservabilitySettings(BaseModel):
    enabled: bool = True
    sink: Literal["console", "file"] = "file"
    path: str = "events/llm/llm_calls.jsonl"
    capture_mode: Literal["metadata", "full"] = "full"
    prompt_view: Literal["off", "compact", "truncated", "full"] = "compact"


class LLMSettings(BaseModel):
    enabled: bool = True
    default: LLMProfile = LLMProfile()
    profiles: dict[str, LLMProfile] = Field(default_factory=dict)
    observability: LLMObservabilitySettings = LLMObservabilitySettings()


class EmbeddingProfile(BaseModel):
    provider: Provider = "openai"
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    timeout: float = 60.0

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

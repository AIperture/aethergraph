from typing import Optional, Literal, Dict
from pydantic import BaseModel, Field, SecretStr
from aethergraph.services.llm.providers import Provider

class LLMProfile(BaseModel):
    provider: Provider = "openai"
    model: str = "gpt-4o-mini"
    embed_model: Optional[str] = None  # separate embedding model
    base_url: Optional[str] = None
    timeout: float = 60.0

    # provider-specific
    azure_deployment: Optional[str] = None

    # secrets (either direct value or ref name)
    api_key: Optional[SecretStr] = None
    api_key_ref: Optional[str] = Field(
        default=None, description="Name in secret store, e.g. 'OPENAI_API_KEY'"
    )

class LLMSettings(BaseModel):
    enabled: bool = True
    default: LLMProfile = LLMProfile()
    profiles: Dict[str, LLMProfile] = Field(default_factory=dict)

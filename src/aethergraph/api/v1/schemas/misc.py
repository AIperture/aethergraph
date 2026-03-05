from pydantic import BaseModel, Field  # type: ignore


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigLLMProvider(BaseModel):
    name: str
    model: str | None = None
    enabled: bool = True


class ConfigResponse(BaseModel):
    version: str
    storage_backends: dict[str, str] = Field(default_factory=dict)
    llm_providers: list[ConfigLLMProvider] = Field(default_factory=list)
    features: dict[str, bool] = Field(default_factory=dict)

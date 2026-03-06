from pydantic import BaseModel, Field, RootModel  # type: ignore


class StatsOverview(BaseModel):
    llm_calls: int = Field(0, description="Total LLM calls in the window")
    llm_prompt_tokens: int = Field(0, description="Total prompt tokens in the window")
    llm_completion_tokens: int = Field(0, description="Total completion tokens in the window")
    runs: int = Field(0, description="Total runs started in the window")
    runs_succeeded: int = Field(0, description="Runs that completed successfully")
    runs_failed: int = Field(0, description="Runs that failed")
    artifacts: int = Field(0, description="Total artifacts recorded in the window")
    artifact_bytes: int = Field(0, description="Total artifact payload size in bytes")
    events: int = Field(0, description="Total metered memory events in the window")
    embedding_calls: int = Field(0, description="Total embedding calls in the window")
    embedding_texts: int = Field(0, description="Total texts embedded in the window")
    embedding_tokens: int = Field(0, description="Total embedding tokens in the window")


class GraphStatsEntry(BaseModel):
    runs: int = Field(0)
    succeeded: int = Field(0)
    failed: int = Field(0)
    total_duration_s: float = Field(0.0)


class GraphStats(RootModel[dict[str, GraphStatsEntry]]):
    pass


class MemoryStats(RootModel[dict[str, dict[str, int]]]):
    pass


class ArtifactStatsEntry(BaseModel):
    count: int = 0
    bytes: int = 0
    pinned_count: int = 0
    pinned_bytes: int = 0


class ArtifactStats(RootModel[dict[str, ArtifactStatsEntry]]):
    pass


class LLMStatsEntry(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMStats(RootModel[dict[str, LLMStatsEntry]]):
    pass


class EmbeddingStatsEntry(BaseModel):
    calls: int = 0
    num_texts: int = 0
    tokens: int = 0


class EmbeddingStats(RootModel[dict[str, EmbeddingStatsEntry]]):
    pass

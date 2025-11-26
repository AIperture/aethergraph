# Schemas for request and response bodies used in the API.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


# --------- Graphs ---------
class GraphListItem(BaseModel):
    graph_id: str
    name: str
    description: str | None = None
    inputs: list[str] = []
    outputs: list[str] = []
    tags: list[str] = []


class GraphDetail(GraphListItem):
    nodes: list[dict[str, Any]]  # Simplified representation of nodes
    edges: list[dict[str, Any]]  # Simplified representation of edges


# --------- Runs ---------


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
    cancellation_requested = "cancellation_requested"


class RunSummary(BaseModel):
    run_id: str
    graph_id: str
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    tags: list[str] = []
    user_id: str | None = None
    org_id: str | None = None


class RunCreateRequest(BaseModel):
    run_id: str | None = None
    inputs: dict[str, Any]
    run_config: dict[str, Any] = {}
    tags: list[str] = []


class RunCreateResponse(BaseModel):
    run_id: str
    graph_id: str
    status: RunStatus
    outputs: dict[str, Any] | None = None
    has_waits: bool
    continuations: list[dict[str, Any]] = []
    started_at: datetime | None = None
    finished_at: datetime | None = None


class NodeSnapshot(BaseModel):
    node_id: str
    tool_name: str | None = None
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None


class RunSnapshot(BaseModel):
    run_id: str
    graph_id: str
    nodes: list[NodeSnapshot]
    edges: list[dict[str, Any]]  # Simplified representation of edges


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    next_cursor: str | None = None


# --------- Memory ---------
class MemoryEvent(BaseModel):
    event_id: str
    scope_id: str
    kind: str
    tages: list[str] = []
    created_at: datetime
    data: dict[str, Any]


class MemoryEventListResponse(BaseModel):
    events: list[MemoryEvent]
    next_cursor: str | None = None


class MemorySummaryEntry(BaseModel):
    summary_id: str
    scope_id: str
    summary_tag: str
    created_at: datetime
    time_from: datetime | None = None
    time_to: datetime | None = None
    text: str
    metadata: dict[str, Any] = {}


class MemorySummaryListResponse(BaseModel):
    summaries: list[MemorySummaryEntry]
    next_cursor: str | None = None


class MemorySearchRequest(BaseModel):
    query: str
    scope_id: str | None = None
    top_k: int = 5
    filters: dict[str, Any] = {}


class MemorySearchHit(BaseModel):
    score: float
    event: MemoryEvent | None = None
    summary: MemorySummaryEntry | None = None


class MemorySearchResponse(BaseModel):
    hits: list[MemorySearchHit]


# --------- Artifacts ---------
class ArtifactMeta(BaseModel):
    artifact_id: str
    kind: str
    mime_type: str | None = None
    size: int | None = None
    scope_id: str | None = None
    tags: list[str] = []
    created_at: datetime
    uri: str | None = None


class ArtifactListResponse(BaseModel):
    query: str
    scope_id: str | None = None
    top_k: int = 5
    filters: dict[str, Any] = {}


class ArtifactSearchRequest(BaseModel):
    query: str
    scope_id: str | None = None
    top_k: int = 10
    filters: dict[str, Any] = {}


class ArtifactSearchHit(BaseModel):
    score: float
    artifact: ArtifactMeta | None = None


class ArtifactSearchResponse(BaseModel):
    hits: list[ArtifactSearchHit]


# --------- channels ---------


class ChannelIngressRequest(BaseModel):
    kind: str = "chat_user"
    text: str | None = None
    metadata: dict[str, Any] = {}


class ChannelEvent(BaseModel):
    event_id: str
    channel_id: str
    kind: str
    created_at: datetime
    data: dict[str, Any]


class ChannelEventListResponse(BaseModel):
    events: list[ChannelEvent]
    next_cursor: str | None = None


# ---------- Misc ----------


class HealthResponse(BaseModel):
    status: str
    version: str


class ConfigLLMProvider(BaseModel):
    name: str
    model: str | None = None
    enabled: bool = True


class ConfigResponse(BaseModel):
    version: str
    storage_backends: dict[str, str] = {}
    llm_providers: list[ConfigLLMProvider] = []
    features: dict[str, bool] = {}

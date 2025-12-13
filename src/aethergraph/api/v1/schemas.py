# Schemas for request and response bodies used in the API.

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, RootModel

from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility, SessionKind


# --------- Graphs ---------
class GraphListItem(BaseModel):
    graph_id: str
    name: str
    description: str | None = None
    inputs: list[str] = []
    outputs: list[str] = []
    tags: list[str] = []
    kind: str | None = None  # "graph" | "graphfn"
    flow_id: str | None = None
    entrypoint: bool | None = None


class GraphNodeInfo(BaseModel):
    id: str
    type: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    expected_inputs: list[str] = []
    expected_outputs: list[str] = []
    output_keys: list[str] = []


class GraphEdgeInfo(BaseModel):
    source: str
    target: str


class GraphDetail(BaseModel):
    graph_id: str
    name: str
    description: str | None = None
    inputs: list[str]
    outputs: list[str]
    tags: list[str] = []

    kind: str | None = None  # "graph" | "graphfn"
    flow_id: str | None = None
    entrypoint: bool | None = None
    nodes: list[GraphNodeInfo] = []
    edges: list[GraphEdgeInfo] = []


# --------- Runs ---------


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    waiting = "waiting"
    canceled = "canceled"
    cancellation_requested = "cancellation_requested"


class RunSummary(BaseModel):
    run_id: str
    graph_id: str
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    session_id: str | None = None
    tags: list[str] = []
    user_id: str | None = None
    org_id: str | None = None
    graph_kind: str | None = None
    flow_id: str | None = None
    entrypoint: bool | None = None

    meta: dict[str, Any] = Field(default_factory=dict)

    # Python attribute names are snake_case, JSON keys are camelCase
    app_id: str | None = Field(default=None, alias="appId")
    app_name: str | None = Field(default=None, alias="appName")
    agent_id: str | None = Field(default=None, alias="agentId")

    # origin/visibility/importance could go here if desired
    origin: RunOrigin | None = None
    visibility: RunVisibility | None = None
    importance: RunImportance | None = None

    class Config:
        populate_by_name = True  # allows setting via app_id/app_name in Python


class RunCreateRequest(BaseModel):
    run_id: str | None = None
    inputs: dict[str, Any]
    run_config: dict[str, Any] = {}
    tags: list[str] = []
    session_id: str | None = None

    # origin/visibility/importance could go here if desired
    origin: RunOrigin | None = None
    visibility: RunVisibility | None = None
    importance: RunImportance | None = None

    # agent / app info
    agent_id: str | None = Field(default=None, alias="agentId")
    app_id: str | None = Field(default=None, alias="appId")
    app_name: str | None = Field(default=None, alias="appName")

    class Config:
        populate_by_name = True  # allows setting via app_id/app_name in Python


class RunCreateResponse(BaseModel):
    run_id: str
    graph_id: str
    status: RunStatus
    outputs: dict[str, Any] | None = None
    has_waits: bool
    continuations: list[dict[str, Any]] = []
    started_at: datetime | None = None
    finished_at: datetime | None = None


# for channel events emitted during a run
class RunChannelEvent(BaseModel):
    id: str
    run_id: str
    type: str  # original OutEvent.type
    text: str | None
    buttons: list[dict[str, Any]]
    file: dict[str, Any] | None
    meta: dict[str, Any]
    ts: float  # unix timestamp


class NodeSnapshot(BaseModel):
    node_id: str
    tool_name: str | None = None
    status: RunStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None


class EdgeSnapshot(BaseModel):
    source: str
    target: str


class RunSnapshot(BaseModel):
    run_id: str
    graph_id: str
    nodes: list[NodeSnapshot]
    edges: list[EdgeSnapshot]
    graph_kind: str | None = None
    flow_id: str | None = None
    entrypoint: bool | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummary]
    next_cursor: str | None = None


# --------- Memory ---------
class MemoryEvent(BaseModel):
    event_id: str
    scope_id: str
    kind: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    data: dict[str, Any] | None = None


class MemoryEventListResponse(BaseModel):
    events: list[MemoryEvent]
    next_cursor: str | None = None


# ---------- Summaries ----------


class MemorySummaryEntry(BaseModel):
    summary_id: str
    scope_id: str
    summary_tag: str
    created_at: datetime
    time_from: datetime
    time_to: datetime
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySummaryListResponse(BaseModel):
    summaries: list[MemorySummaryEntry]
    next_cursor: str | None = None


# ---------- Search ----------


class MemorySearchRequest(BaseModel):
    query: str
    scope_id: str | None = None
    top_k: int = 10


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
    pinned: bool = False
    preview_uri: str | None = None

    # Associated run / graph / node
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactMeta]
    next_cursor: str | None = None


class ArtifactSearchRequest(BaseModel):
    # Optional semantic / text query (for future embedding search)
    query: str | None = None

    # Common filters
    scope_id: str | None = None
    kind: str | None = None
    tags: list[str] | None = None

    # Extra label filters that map directly to Artifact.labels
    labels: dict[str, Any] = Field(default_factory=dict)

    # Metric-based ranking
    metric: str | None = None
    mode: Literal["max", "min"] | None = None

    # Pagination / result size
    limit: int = 10

    # If True, use index.best(...) and only return a single hit
    best_only: bool = False


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


# --------- Stats ---------
# ---------- Overview ----------


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


# ---------- Graph stats ----------


class GraphStatsEntry(BaseModel):
    runs: int = Field(0)
    succeeded: int = Field(0)
    failed: int = Field(0)
    total_duration_s: float = Field(0.0)


class GraphStats(RootModel[dict[str, GraphStatsEntry]]):
    """Map graph_id -> GraphStatsEntry"""

    # no extra fields needed; RootModel handles serialization


# ---------- Memory stats ----------


class MemoryStats(RootModel[dict[str, dict[str, int]]]):
    """Map memory_kind -> { 'count': int }"""


# ---------- Artifact stats ----------


class ArtifactStatsEntry(BaseModel):
    count: int = 0
    bytes: int = 0
    pinned_count: int = 0
    pinned_bytes: int = 0


class ArtifactStats(RootModel[dict[str, ArtifactStatsEntry]]):
    """Map artifact_kind -> ArtifactStatsEntry"""


# ---------- LLM stats ----------


class LLMStatsEntry(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMStats(RootModel[dict[str, LLMStatsEntry]]):
    """
    Map of LLM model name → stats.

    Example:
      {
        "gpt-4o-mini": {"calls": 5, "prompt_tokens": 1234, "completion_tokens": 567}
      }
    """


# ------ viz facade integration ------
VizKind = Literal["scalar", "vector", "matrix", "image"]
VizMode = Literal["append", "replace"]


class VizPoint(BaseModel):
    step: int = Field(..., description="Iteration / timestep")
    value: float | None = None
    vector: list[float] | None = None
    matrix: list[list[float]] | None = None
    artifact_id: str | None = Field(
        None,
        description="Artifact ID for image frames (or future 3D payloads).",
    )
    created_at: datetime | None = None


class VizTrack(BaseModel):
    track_id: str = Field(..., description="Developer-chosen ID, e.g. 'loss', 'design_shape'")
    figure_id: str | None = Field(None, description="Optional panel/group ID, e.g. 'metrics_panel'")
    node_id: str | None = Field(None, description="Node that emitted this track, if applicable")
    viz_kind: VizKind
    mode: VizMode = "append"
    meta: dict[str, Any] | None = None
    points: list[VizPoint]


class VizFigure(BaseModel):
    figure_id: str | None = Field(
        None,
        description="Panel/group identifier; tracks with same figure_id are shown together",
    )
    tracks: list[VizTrack]


class RunVizResponse(BaseModel):
    run_id: str
    figures: list[VizFigure]


# -------------- Session Schemas --------------
class Session(BaseModel):
    session_id: str
    kind: SessionKind
    title: str | None = None

    user_id: str | None = None
    org_id: str | None = None

    source: str = "webui"  # "webui", "sidecar", "api", "sdk", etc.
    external_ref: str | None = None  # e.g. chat ID, playground ID, notebook path, etc.

    created_at: datetime
    updated_at: datetime


class SessionCreateRequest(BaseModel):
    kind: SessionKind
    title: str | None = None
    external_ref: str | None = None


class SessionListResponse(BaseModel):
    items: list[Session]
    next_cursor: str | None = None


class SessionRunsResponse(BaseModel):
    items: list[RunSummary]

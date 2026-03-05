from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field  # type: ignore

from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility


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
    tags: list[str] = Field(default_factory=list)
    user_id: str | None = None
    org_id: str | None = None
    graph_kind: str | None = None
    flow_id: str | None = None
    entrypoint: bool | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    app_id: str | None = Field(default=None, alias="appId")
    app_name: str | None = Field(default=None, alias="appName")
    agent_id: str | None = Field(default=None, alias="agentId")
    origin: RunOrigin | None = None
    visibility: RunVisibility | None = None
    importance: RunImportance | None = None
    artifact_count: int | None = None
    last_artifact_at: datetime | None = None

    class Config:
        populate_by_name = True


class RunCreateRequest(BaseModel):
    run_id: str | None = None
    inputs: dict[str, Any]
    run_config: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    session_id: str | None = None
    origin: RunOrigin | None = None
    visibility: RunVisibility | None = None
    importance: RunImportance | None = None
    agent_id: str | None = Field(default=None, alias="agentId")
    app_id: str | None = Field(default=None, alias="appId")
    app_name: str | None = Field(default=None, alias="appName")

    class Config:
        populate_by_name = True


class RunCreateResponse(BaseModel):
    run_id: str
    graph_id: str
    status: RunStatus
    outputs: dict[str, Any] | None = None
    has_waits: bool
    continuations: list[dict[str, Any]] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunChannelEvent(BaseModel):
    id: str
    run_id: str
    type: str
    text: str | None
    buttons: list[dict[str, Any]] = Field(default_factory=list)
    file: dict[str, Any] | None
    meta: dict[str, Any] = Field(default_factory=dict)
    ts: float


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

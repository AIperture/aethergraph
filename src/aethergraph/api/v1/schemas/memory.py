from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field  # type: ignore


class MemoryEvent(BaseModel):
    event_id: str
    scope_id: str
    ts: str | float | None = None
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    node_id: str | None = None
    graph_id: str | None = None
    kind: str
    stage: str | None = None
    topic: str | None = None
    tool: str | None = None
    tags: list[str] = Field(default_factory=list)
    severity: int | None = None
    signal: float | None = None
    created_at: datetime
    snippet: str = ""
    text: str | None = None
    data: dict[str, Any] | None = None
    metrics: dict[str, float] | None = None
    inputs: list[dict[str, Any]] | None = None
    outputs: list[dict[str, Any]] | None = None


class MemoryEventListResponse(BaseModel):
    events: list[MemoryEvent]
    next_cursor: str | None = None


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

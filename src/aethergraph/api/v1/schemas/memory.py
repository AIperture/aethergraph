from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field  # type: ignore


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

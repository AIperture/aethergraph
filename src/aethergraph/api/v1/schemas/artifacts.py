from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field  # type: ignore


class ArtifactMeta(BaseModel):
    occurrence_id: str | None = None
    artifact_id: str
    kind: str
    mime_type: str | None = None
    size: int | None = None
    scope_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    uri: str | None = None
    pinned: bool = False
    preview_uri: str | None = None
    run_id: str | None = None
    graph_id: str | None = None
    node_id: str | None = None
    session_id: str | None = None
    filename: str | None = None


class ArtifactListResponse(BaseModel):
    artifacts: list[ArtifactMeta]
    next_cursor: str | None = None


class ArtifactSearchRequest(BaseModel):
    query: str | None = None
    scope_id: str | None = None
    kind: str | None = None
    tags: list[str] | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    metric: str | None = None
    mode: Literal["max", "min"] | None = None
    limit: int = 10
    best_only: bool = False


class ArtifactSearchHit(BaseModel):
    score: float
    artifact: ArtifactMeta | None = None


class ArtifactSearchResponse(BaseModel):
    hits: list[ArtifactSearchHit]

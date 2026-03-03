from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field  # type: ignore

VizKind = Literal["scalar", "vector", "matrix", "image"]
VizMode = Literal["append", "replace"]


class VizPoint(BaseModel):
    step: int = Field(..., description="Iteration or timestep")
    value: float | None = None
    vector: list[float] | None = None
    matrix: list[list[float]] | None = None
    artifact_id: str | None = Field(None, description="Artifact ID for image frames.")
    created_at: datetime | None = None


class VizTrack(BaseModel):
    track_id: str = Field(..., description="Developer-defined track id")
    figure_id: str | None = Field(None, description="Optional figure or panel id")
    node_id: str | None = Field(None, description="Node that emitted this track")
    viz_kind: VizKind
    mode: VizMode = "append"
    meta: dict[str, Any] | None = None
    points: list[VizPoint]


class VizFigure(BaseModel):
    figure_id: str | None = Field(None, description="Panel or group identifier")
    tracks: list[VizTrack]


class RunVizResponse(BaseModel):
    run_id: str
    figures: list[VizFigure]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

VizKind = Literal["scalar", "vector", "matrix", "image"]
VizMode = Literal["append", "replace"]


@dataclass
class VizEvent:
    # Provenance
    run_id: str
    graph_id: str
    node_id: str
    tool_name: str
    tool_version: str

    # Visualization fields
    track_id: str  # unique id for the trace (e.g., "loss", "accuracy")
    figure_id: str  # optional figure id for grouping traces, e.g. "metrics_panel"
    viz_kind: VizKind
    step: int  # iteration or step number
    mode: VizMode = "append"  # append or replace

    # Tenant-ish fields
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    app_id: str | None = None
    session_id: str | None = None

    # Payload
    value: float | None = None  # for scalar
    vector: list[float] | None = None  # for vector
    matrix: list[list[float]] | None = None  # for matrix
    artifact_id: str | None = None  # for image or other artifact-based viz

    # Optional metadata
    meta: dict[str, Any] | None = None  # {"label": "Training Loss", "color": "blue", ...}
    tags: list[str] | None = None  # arbitrary tags for filtering or grouping

    # Timestamp
    created_at: str | None = None  # ISO 8601 timestamp

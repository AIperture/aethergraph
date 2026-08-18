from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

VizKind = Literal["scalar", "vector", "matrix", "image"]
VizMode = Literal["append", "replace"]


@dataclass
class VizEvent:
    """Mutable public Viz input with explicitly deprecated compatibility identity.

    `app_id` and `client_id` remain optional compatibility metadata for the active
    pre-cut facade. Canonical persistence never places either value in provider scope,
    authorization, query keys, or indexes.
    """

    # Provenance
    run_id: str
    graph_id: str
    node_id: str
    tool_name: str
    tool_version: str

    # Visualization fields
    track_id: str  # unique id for the trace (e.g., "loss", "accuracy")
    figure_id: str | None  # optional figure id for grouping traces, e.g. "metrics_panel"
    viz_kind: VizKind
    step: int  # iteration or step number
    mode: VizMode = "append"  # append or replace

    # Tenant-ish fields
    org_id: str | None = None
    user_id: str | None = None
    client_id: str | None = field(
        default=None,
        metadata={"deprecated": True, "role": "optional compatibility metadata"},
    )
    app_id: str | None = field(
        default=None,
        metadata={"deprecated": True, "role": "optional compatibility metadata"},
    )
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


class VizEventSink(Protocol):
    """Append validated public Viz events without exposing persistence layout."""

    async def append(self, evt: VizEvent) -> None:
        """Persist one visualization event through the configured service boundary.

        Intro:
            Accepts the stable public Viz input and completes only after the selected
            service has made it authoritative. The sink exposes no provider cursor or
            physical storage handle.

        Examples:
            Append a scalar event:
                ```python
                await sink.append(scalar_event)
                ```

            Append an image reference:
                ```python
                await sink.append(image_event)
                ```

        Args:
            evt: Valid public visualization event to persist exactly once.

        Returns:
            None: The event is authoritative before the await completes.

        Notes:
            Implementations must not derive provider scope from deprecated `app_id`
            or `client_id` compatibility metadata.
        """
        ...

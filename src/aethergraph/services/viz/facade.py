from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.viz import VizEvent, VizMode
from aethergraph.services.artifacts.facade import Artifact, ArtifactFacade
from aethergraph.services.scope.scope import Scope
from aethergraph.services.viz.viz_service import VizService


@dataclass
class VizFacade:
    """
    High-level facade for visualization operations within a given Scope.

    - Wraps VizService and ArtifactFacade.
    - Knows about Scope to auto-fill provenance and tenant fields.

    Usage pattern in ctx.viz:
    # Scalars
    await ctx.viz.scalar("loss", step=iter, value=float(loss), figure_id="metrics")

    # Matrix (small heatmap)
    await ctx.viz.matrix("field_map", step=iter, matrix=field_2d, figure_id="fields")

    # Image (pre-rendered PNG)
    artifact = await ctx.artifacts.save_file(path="frame_17.png", kind="image")
    await ctx.viz.image_from_artifact("design_shape", step=17, artifact=artifact, figure_id="design")
    """

    run_id: str
    graph_id: str
    node_id: str
    tool_name: str
    tool_version: str

    viz_service: VizService
    scope: Scope | None = None
    artifacts: ArtifactFacade | None = None

    # ------- internal helpers -------
    def _scope_dims(self) -> dict[str, Any]:
        if not self.scope:
            return {}
        return self.scope.metering_dimensions()

    def _apply_scope(self, evt: VizEvent) -> VizEvent:
        dims = self._scope_dims()
        evt.org_id = evt.org_id or dims.get("org_id")
        evt.user_id = evt.user_id or dims.get("user_id")
        evt.client_id = evt.client_id or dims.get("client_id")
        evt.app_id = evt.app_id or dims.get("app_id")
        evt.session_id = evt.session_id or dims.get("session_id")
        return evt

    # ------- public API -------
    async def scalar(
        self,
        track_id: str,
        *,
        step: int,
        value: float,
        figure_id: str | None = None,
        mode: VizMode = "append",
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        evt = VizEvent(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            track_id=track_id,
            figure_id=figure_id,
            viz_kind="scalar",
            step=step,
            mode=mode,
            value=float(value),
            meta=meta,
            tags=(tags or []) + ["type:scalar"],
        )
        evt = self._apply_scope(evt)
        await self.viz_service.append(evt)

    async def vector(
        self,
        track_id: str,
        *,
        step: int,
        values: Sequence[float],
        figure_id: str | None = None,
        mode: VizMode = "append",
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        evt = VizEvent(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            track_id=track_id,
            figure_id=figure_id,
            viz_kind="vector",
            step=step,
            mode=mode,
            vector=[float(v) for v in values],
            meta=meta,
            tags=(tags or []) + ["type:vector"],
        )
        evt = self._apply_scope(evt)
        await self.viz_service.append(evt)

    async def matrix(
        self,
        track_id: str,
        *,
        step: int,
        matrix: Sequence[Sequence[float]],
        figure_id: str | None = None,
        mode: VizMode = "append",
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        # Convert to plain list[list[float]]
        m = [[float(x) for x in row] for row in matrix]
        evt = VizEvent(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            track_id=track_id,
            figure_id=figure_id,
            viz_kind="matrix",
            step=step,
            mode=mode,
            matrix=m,
            meta=meta,
            tags=(tags or []) + ["matrix"],
        )
        evt = self._apply_scope(evt)
        await self.viz_service.append(evt)

    async def image_from_artifact(
        self,
        track_id: str,
        *,
        step: int,
        artifact: Artifact,
        figure_id: str | None = None,
        mode: VizMode = "append",
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Log a reference to an existing Artifact as an image track.

        Caller is responsible for creating the artifact via ctx.artifacts().
        """
        evt = VizEvent(
            run_id=self.run_id,
            graph_id=self.graph_id,
            node_id=self.node_id,
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            track_id=track_id,
            figure_id=figure_id,
            viz_kind="image",
            step=step,
            mode=mode,
            artifact_id=artifact.artifact_id,
            meta=meta,
            tags=(tags or []) + ["image"],
        )
        evt = self._apply_scope(evt)
        await self.viz_service.append(evt)

    async def image_from_bytes(
        self,
        track_id: str,
        *,
        step: int,
        data: bytes,
        mime: str = "image/png",
        kind: str = "image",
        figure_id: str | None = None,
        mode: VizMode = "append",
        labels: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Artifact:
        """
        Convenience helper: save image bytes as an Artifact then log a viz event.
        Requires artifacts facade to be configured.
        """
        if not self.artifacts:
            raise RuntimeError("VizFacade.image_from_bytes requires an ArtifactFacade")

        # Save artifact using writer() so we get proper metering + labels

        # Use ArtifactFacade.writer to store the image
        async with self.artifacts.writer(kind=kind, planned_ext=".png") as w:
            w.write(data)
            if labels:
                w.add_labels(labels)
        art = self.artifacts.last_artifact
        if not art:
            raise RuntimeError("Artifact writer did not produce an artifact")

        await self.image_from_artifact(
            track_id=track_id,
            step=step,
            artifact=art,
            figure_id=figure_id,
            mode=mode,
            meta=meta,
            tags=tags,
        )
        return art

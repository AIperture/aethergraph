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
        """
        Record a single scalar value for visualization in the Aethergraph UI.

        This method standardizes the event format, auto-fills provenance fields,
        and dispatches the scalar data to the configured storage backend.

        Examples:
            Basic usage to log a loss metric:
            ```python
            await context.viz().scalar("loss", step=iteration, value=loss)
            ```

            Logging a scalar with extra metadata and custom tags:
            ```python
            await context.viz().scalar(
                "accuracy",
                step=42,
                value=0.98,
                figure_id="metrics",
                meta={"model": "resnet"},
                tags=["experiment:baseline"]
            )
            ```

        Args:
            track_id: Unique identifier for the scalar track (e.g., "loss").
            step: Integer step or iteration number for the data point.
            value: The scalar value to record (float).
            figure_id: Optional figure grouping for UI display.
            mode: Storage mode, typically "append".
            meta: Optional dictionary of extra metadata.
            tags: Optional list of string labels. The tag "type:scalar" is automatically appended.

        Returns:
            None. The event is persisted for later visualization.
        """
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
        """
        Record a single vector (1D array) for visualization in the Aethergraph UI.

        This method standardizes the event format, auto-fills provenance fields,
        and dispatches the vector data to the configured storage backend.

        Examples:
            Basic usage to log a vector:
            ```python
            await context.viz().vector("embedding", step=iteration, values=[0.1, 0.2, 0.3])
            ```

            Logging a vector with extra metadata and custom tags:
            ```python
            await context.viz().vector(
                "features",
                step=42,
                values=[1.0, 2.5, 3.7],
                figure_id="feature_tracks",
                meta={"source": "encoder"},
                tags=["experiment:baseline"]
            )
            ```

        Args:
            track_id: Unique identifier for the vector track (e.g., "embedding").
            step: Integer step or iteration number for the data point.
            values: Sequence of float values representing the vector.
            figure_id: Optional figure grouping for UI display.
            mode: Storage mode, typically "append".
            meta: Optional dictionary of extra metadata.
            tags: Optional list of string labels. The tag "type:vector" is automatically appended.

        Returns:
            None. The event is persisted for later visualization.
        """
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
        """
        Record a single matrix (2D array) for visualization in the Aethergraph UI.

        This method standardizes the event format, auto-fills provenance fields,
        and dispatches the matrix data to the configured storage backend.

        Examples:
            Basic usage to log a matrix:
            ```python
            await context.viz().matrix("confusion", step=iteration, matrix=[[1, 2], [3, 4]])
            ```

            Logging a matrix with extra metadata and custom tags:
            ```python
            await context.viz().matrix(
                "heatmap",
                step=42,
                matrix=[[0.1, 0.2], [0.3, 0.4]],
                figure_id="metrics",
                meta={"source": "model"},
                tags=["experiment:baseline"]
            )
            ```

        Args:
            track_id: Unique identifier for the matrix track (e.g., "confusion").
            step: Integer step or iteration number for the data point.
            matrix: Sequence of sequences of float values representing the 2D matrix.
            figure_id: Optional figure grouping for UI display.
            mode: Storage mode, typically "append".
            meta: Optional dictionary of extra metadata.
            tags: Optional list of string labels. The tag "matrix" is automatically appended.

        Returns:
            None. The event is persisted for later visualization.
        """
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
        Record a reference to an existing image Artifact for visualization in the Aethergraph UI.

        This method standardizes the event format, auto-fills provenance fields,
        and dispatches the image reference to the configured storage backend.

        Examples:
            Basic usage to log an image artifact:
            ```python
            await context.viz().image_from_artifact(
                "design_shape",
                step=17,
                artifact=artifact,
                figure_id="design"
            )
            ```

            Logging an image with extra metadata and custom tags:
            ```python
            await context.viz().image_from_artifact(
                "output_frame",
                step=42,
                artifact=artifact,
                figure_id="frames",
                meta={"source": "simulation"},
                tags=["experiment:baseline"]
            )
            ```

        Args:
            track_id: Unique identifier for the image track (e.g., "design_shape").
            step: Integer step or iteration number for the data point.
            artifact: The Artifact object referencing the stored image.
            figure_id: Optional figure grouping for UI display.
            mode: Storage mode, typically "append".
            meta: Optional dictionary of extra metadata.
            tags: Optional list of string labels. The tag "image" is automatically appended.

        Returns:
            None. The event is persisted for later visualization.
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
        Save image bytes as an Artifact and log a visualization event.

        This convenience method is accessed via `context.viz().image_from_bytes(...)` and is used by the Aethergraph UI to persist image data to storage. It stores the image as an Artifact using the configured ArtifactFacade, then logs a visualization event referencing the saved artifact.

        Examples:
            Saving a PNG image to the current visualization track:
            ```python
            await context.viz().image_from_bytes(
                track_id="experiment-123",
                step=42,
                data=image_bytes,
                mime="image/png",
                labels={"type": "output", "stage": "inference"},
                tags=["result", "png"]
            )
            ```

            Saving an image with custom metadata and figure association:
            ```python
            await context.viz().image_from_bytes(
                track_id="demo-track",
                step=7,
                data=img_bytes,
                figure_id="fig-1",
                meta={"caption": "Sample output"},
                mode="replace"
            )
            ```

        Args:
            track_id: The identifier for the visualization track to associate with the image.
            step: The step or index within the track for this image.
            data: Raw image bytes to be saved.
            mime: The MIME type of the image (default: "image/png").
            kind: The artifact kind (default: "image").
            figure_id: Optional identifier for the figure this image belongs to.
            mode: Visualization mode, e.g., "append" or "replace".
            labels: Optional dictionary of labels to attach to the artifact.
            meta: Optional dictionary of metadata for the visualization event.
            tags: Optional list of string tags for categorization.

        Returns:
            Artifact: The persisted `Artifact` object representing the saved image.

        Notes:
            - This method requires that `self.artifacts` is set to an `ArtifactFacade` instance.
            - The saved artifact is automatically linked to the visualization event.
            - To change the name of the saved artifact in UI, use `labels` to set a "filename" label.
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

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.viz.facade import VizFacade


class _SyncWriter:
    def __init__(self) -> None:
        self.payloads: list[bytes] = []
        self.labels: dict[str, Any] = {}

    def write(self, data: bytes) -> None:
        self.payloads.append(data)

    def add_labels(self, labels: dict[str, Any]) -> None:
        self.labels.update(labels)


class _AsyncWriter(_SyncWriter):
    async def write(self, data: bytes) -> None:
        self.payloads.append(data)


class _Artifacts:
    def __init__(self, writer: _SyncWriter) -> None:
        self.stream = writer
        self.last_artifact: Artifact | None = None

    @asynccontextmanager
    async def writer(self, **_: Any):
        yield self.stream
        self.last_artifact = Artifact(artifact_id="artifact-1", kind="image")


class _VizService:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def append(self, event: Any) -> None:
        self.events.append(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("writer_type", [_SyncWriter, _AsyncWriter])
async def test_image_from_bytes_supports_sync_and_canonical_async_writers(writer_type) -> None:
    writer = writer_type()
    artifacts = _Artifacts(writer)
    viz_service = _VizService()
    facade = VizFacade(
        run_id="run-1",
        graph_id="graph-1",
        node_id="node-1",
        tool_name="renderer",
        tool_version="1",
        viz_service=viz_service,  # type: ignore[arg-type]
        artifacts=artifacts,  # type: ignore[arg-type]
    )

    artifact = await facade.image_from_bytes(
        "images",
        step=1,
        data=b"png-data",
        labels={"filename": "frame.png"},
    )

    assert artifact.artifact_id == "artifact-1"
    assert writer.payloads == [b"png-data"]
    assert writer.labels == {"filename": "frame.png"}
    assert len(viz_service.events) == 1
    assert viz_service.events[0].artifact_id == "artifact-1"

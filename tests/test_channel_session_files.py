from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.core.tools.builtins.toolset import send_file as send_file_tool
from aethergraph.services.channel.session import ChannelSession, _artifact_to_chat_file
from aethergraph.utils.mime_types import mime_type_for_filename


class _Bus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class _Writer(AbstractAsyncContextManager):
    def __init__(self, artifacts: _Artifacts) -> None:
        self.artifacts = artifacts
        self.labels: dict[str, Any] = {}
        self.content = b""

    async def __aenter__(self) -> _Writer:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.artifacts.last_artifact = Artifact(
                artifact_id="artifact:file-1",
                bytes=len(self.content),
                mime=self.artifacts.writer_mime,
                labels=dict(self.labels),
            )

    async def write(self, content: bytes) -> None:
        self.content += content

    def add_labels(self, labels: dict[str, Any]) -> None:
        self.labels.update(labels)


class _Artifacts:
    def __init__(self) -> None:
        self.last_artifact: Artifact | None = None
        self.writer_mime: str | None = None

    def writer(self, *, mime: str, **_: Any) -> _Writer:
        self.writer_mime = mime
        return _Writer(self)

    async def get_by_id(self, artifact_id: str) -> Artifact | None:
        if self.last_artifact is not None and self.last_artifact.artifact_id == artifact_id:
            return self.last_artifact
        return None


class _Context:
    def __init__(self) -> None:
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.origin_binding = SimpleNamespace(channel_key="ui:test")
        self.services = SimpleNamespace(
            channels=_Bus(),
            continuation_store=None,
            memory_facade=None,
        )
        self._artifacts = _Artifacts()

    def artifacts(self) -> _Artifacts:
        return self._artifacts


@pytest.mark.parametrize(
    ("filename", "expected"),
    (
        ("probe.png", "image/png"),
        ("probe.webp", "image/webp"),
        ("probe.pdf", "application/pdf"),
        ("probe.csv", "text/csv"),
        ("probe.json", "application/json"),
        ("probe.jsonl", "application/x-ndjson"),
        ("probe.md", "text/markdown"),
        ("probe.yaml", "application/yaml"),
        ("probe.mp3", "audio/mpeg"),
        ("probe.mp4", "video/mp4"),
        (
            "probe.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        ("probe.zip", "application/zip"),
    ),
)
def test_filename_mime_registry_is_platform_independent(filename: str, expected: str) -> None:
    assert mime_type_for_filename(filename) == expected


def test_artifact_descriptor_prefers_specific_label_over_generic_storage_mime() -> None:
    descriptor = _artifact_to_chat_file(
        Artifact(
            artifact_id="artifact:report",
            mime="application/octet-stream",
            labels={"filename": "report.custom", "mimetype": "application/x-report"},
        )
    )

    assert descriptor["mimetype"] == "application/x-report"


@pytest.mark.asyncio
async def test_send_file_persists_and_publishes_one_canonical_mimetype() -> None:
    context = _Context()

    await ChannelSession(context).send_file(
        file_bytes=b"probe,status\nfile,ok\n",
        filename="conversation-contract-report.csv",
        title="Conversation contract report",
        artifact_labels={"mimetype": "text/csv", "contract_probe": "file"},
    )

    assert context._artifacts.writer_mime == "text/csv"
    artifact = context._artifacts.last_artifact
    assert artifact is not None
    assert artifact.mime == "text/csv"
    assert artifact.labels == {
        "contract_probe": "file",
        "filename": "conversation-contract-report.csv",
        "mimetype": "text/csv",
    }
    assert len(context.services.channels.events) == 1
    assert context.services.channels.events[0].file is None
    assert context.services.channels.events[0].attachments[0]["mimetype"] == "text/csv"


@pytest.mark.asyncio
async def test_send_file_tool_forwards_authored_mimetype() -> None:
    channel = SimpleNamespace(send_file=AsyncMock())
    context = SimpleNamespace(channel=lambda _: channel)

    result = await send_file_tool(
        file_bytes=b"report",
        filename="report.csv",
        mimetype="text/csv",
        context=context,
    )

    assert result == {"ok": True}
    channel.send_file.assert_awaited_once_with(
        url=None,
        file_bytes=b"report",
        filename="report.csv",
        title=None,
        mimetype="text/csv",
    )

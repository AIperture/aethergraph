from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.services.channel import (
    ChannelAction,
    ChannelAttachment,
    ChannelMessage,
)
from aethergraph.services.channel.session import ChannelSession


class _Bus:
    def __init__(self) -> None:
        self.events = []

    async def publish(self, event):
        self.events.append(event)
        return {"event_cursors": [4, 5], "provider_delivery_ids": ["provider-1"]}


class _Artifacts:
    def __init__(self) -> None:
        self.items = {
            "artifact-image": Artifact(
                artifact_id="artifact-image",
                bytes=67,
                mime="image/png",
                labels={"filename": "probe.png"},
            ),
            "artifact-file": Artifact(
                artifact_id="artifact-file",
                bytes=21,
                mime="text/csv",
                labels={"filename": "probe.csv"},
            ),
        }

    async def get_by_id(self, artifact_id: str):
        return self.items.get(artifact_id)


class _Context:
    def __init__(self) -> None:
        self.run_id = "run-1"
        self.node_id = "node-1"
        self.session_id = "session-1"
        self.graph_id = "graph-1"
        self.agent_id = "agent-1"
        self.app_id = "app-1"
        self.origin_binding = SimpleNamespace(channel_key="endpoint:sessions/session-1")
        self.services = SimpleNamespace(
            channels=_Bus(),
            continuation_store=None,
            memory_facade=None,
        )
        self._artifacts = _Artifacts()

    def artifacts(self) -> _Artifacts:
        return self._artifacts


def test_channel_message_rejects_duplicate_attachments_and_unsafe_actions() -> None:
    with pytest.raises(ValueError, match="duplicate attachment"):
        ChannelMessage(
            message_id="message-1",
            attachments=(
                ChannelAttachment("artifact-1"),
                ChannelAttachment("artifact-1"),
            ),
        )

    with pytest.raises(ValueError, match="absolute HTTP"):
        ChannelAction(kind="external_link", label="Open", href="file:///tmp/report")


@pytest.mark.asyncio
async def test_send_message_hydrates_all_artifacts_before_one_publish() -> None:
    context = _Context()

    receipt = await ChannelSession(context).send_message(
        ChannelMessage(
            message_id="message-1",
            text_markdown="Results are ready.",
            attachments=(
                ChannelAttachment(
                    artifact_id="artifact-image",
                    presentation="image",
                    alt_text="Contract image",
                ),
                ChannelAttachment(
                    artifact_id="artifact-file",
                    presentation="file",
                    title="Contract report",
                ),
            ),
            actions=(
                ChannelAction(
                    kind="suggested_reply",
                    label="Continue",
                    value="continue",
                ),
                ChannelAction(
                    kind="external_link",
                    label="Documentation",
                    href="https://example.test/docs",
                ),
            ),
        )
    )

    assert receipt.message_id == "message-1"
    assert receipt.event_cursors == (4, 5)
    assert receipt.provider_delivery_ids == ("provider-1",)
    assert len(context.services.channels.events) == 1
    event = context.services.channels.events[0]
    assert event.upsert_key == "message-1"
    assert [item["artifact_id"] for item in event.attachments] == [
        "artifact-image",
        "artifact-file",
    ]
    assert event.attachments[0]["mimetype"] == "image/png"
    assert [action.kind for action in event.actions] == [
        "suggested_reply",
        "external_link",
    ]


@pytest.mark.asyncio
async def test_send_message_rejects_missing_artifact_before_publish() -> None:
    context = _Context()

    with pytest.raises(FileNotFoundError, match="missing-artifact"):
        await ChannelSession(context).send_message(
            ChannelMessage(
                message_id="message-1",
                text_markdown="Unavailable",
                attachments=(ChannelAttachment("missing-artifact"),),
            )
        )

    assert context.services.channels.events == []


@pytest.mark.asyncio
async def test_send_message_rejects_non_image_presentation_before_publish() -> None:
    context = _Context()

    with pytest.raises(ValueError, match="not an image"):
        await ChannelSession(context).send_message(
            ChannelMessage(
                message_id="message-1",
                attachments=(ChannelAttachment("artifact-file", presentation="image"),),
            )
        )

    assert context.services.channels.events == []

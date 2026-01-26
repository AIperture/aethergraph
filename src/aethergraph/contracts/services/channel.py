from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

EventType = Literal[
    "agent.message",
    "agent.message.update",  # simple text messages
    "agent.stream.start",
    "agent.stream.delta",
    "agent.stream.end",  # streaming messages
    "agent.progress.start",
    "agent.progress.update",
    "agent.progress.end",  # progress bar
    "session.need_input",
    "session.need_approval",
    "session.waiting",
    "file.upload",
    "link.buttons",  # link preview with buttons
]


class PhaseRich(TypedDict, total=False):
    kind: Literal["phase"]
    phase: str  # "routing", "planning", "reasoning", "tools", "reply"
    status: Literal["pending", "active", "done", "failed", "skipped"]
    label: str | None  # short human label
    detail: str | None  # optional extra text
    code: str | None  # internal code like "routing.planning"


class ProgressRich(TypedDict, total=False):
    kind: Literal["progress"]
    label: str | None
    current: float | int | None
    total: float | int | None
    unit: str | None  # "%", "steps", etc.


class FileRef(TypedDict, total=False):
    id: str  # platform file id (e.g., Slack file ID)
    name: str  # suggested filename
    mimetype: str  # MIME type, e.g., "image/png", "application/pdf"
    size: int  # size in bytes
    uri: str  # URL to download the file (artifact storage or platform URL)
    url_private: str  # private URL if applicable (e.g., Slack private URL)
    platform: str  # platform name, e.g., "slack", "telegram", "console"
    channel_key: str  # normalized channel key where the file was sent, e.g., "slack:team/T:chan/C"
    ts: str | float  # timestamp of the file upload


@dataclass
class Button:
    label: str
    value: str | None = None
    url: str | None = None
    style: Literal["primary", "danger", "default"] | None = None  # for slack buttons


@dataclass
class OutEvent:
    type: EventType  # "agent.message", "session.need_input", "session.need_approval", "agent.stream.*"
    channel: str  # routing key, e.g., "console:stdout" or "slack:team/T:chan/C[:thread/TS]"
    text: str | None = None
    rich: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    # Optional structured extras most adapters can use, e.g., for buttons, attachments, files, etc.
    buttons: list[Button] | None = None
    image: dict[str, Any] | None = None  # e.g., {"url": "...", "alt": "...", "title": "..."}
    file: dict[str, Any] | None = (
        None  # e.g., {"bytes" b"...", "filename": "...", "mimetype": "..."} or {"url": "...", "filename": "...", "mimetype": "..."}
    )
    upsert_key: str | None = None  # for idempotent upserts, e.g., message ID to update same message

    def to_printable(self) -> str:
        """Only contains printable parts of the event."""
        return (
            f"Event(type={self.type}, channel={self.channel}, text={self.text}, meta={self.meta})"
        )


class ChannelAdapter(Protocol):
    # Capabilities helper
    capabilities: set[str]  # e.g. {"text", "image", "file", "buttons", "rich"}

    async def send(self, event: OutEvent) -> None:
        """
        Send an outgoing event to the appropriate channel.
        E.g., print to console, post to Slack, enqueue in UI, etc.
        """
        pass

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Literal, Set, TypedDict

EventType = Literal[
    "agent.message", "agent.message.update", # simple text messages
    "agent.stream.start", "agent.stream.delta", "agent.stream.end", # streaming messages
    "agent.progress.start", "agent.progress.update", "agent.progress.end", # progress bar
    "session.need_input", "session.need_approval", "session.waiting", 
    "file.upload", 
    "link.buttons" # link preview with buttons
]

class FileRef(TypedDict, total=False):
    id: str                # platform file id (e.g., Slack file ID)
    name: str              # suggested filename
    mimetype: str          # MIME type, e.g., "image/png", "application/pdf"
    size: int              # size in bytes
    uri: str               # URL to download the file (artifact storage or platform URL)
    url_private: str       # private URL if applicable (e.g., Slack private URL)
    platform: str          # platform name, e.g., "slack", "telegram", "console"
    channel_key: str       # normalized channel key where the file was sent, e.g., "slack:team/T:chan/C"
    ts: str | float        # timestamp of the file upload


@dataclass 
class Button:
    label: str 
    value: Optional[str] = None
    url: Optional[str] = None
    style: Optional[Literal["primary", "danger", "default"]] = None  # for slack buttons



@dataclass
class OutEvent:
    type: EventType                    # "agent.message", "session.need_input", "session.need_approval", "agent.stream.*"
    channel: str                       # routing key, e.g., "console:stdout" or "slack:team/T:chan/C[:thread/TS]"
    text: Optional[str] = None
    rich: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    # Optional structured extras most adapters can use, e.g., for buttons, attachments, files, etc.
    buttons: Optional[Dict[str, Button]] = None  # for approvals or link actions 
    image: Optional[Dict[str, Any]] = None  # e.g., {"url": "...", "alt": "...", "title": "..."}
    file: Optional[Dict[str, Any]] = None  # e.g., {"bytes" b"...", "filename": "...", "mimetype": "..."} or {"url": "...", "filename": "...", "mimetype": "..."}
    upsert_key: Optional[str] = None  # for idempotent upserts, e.g., message ID to update same message

    def to_printable(self) -> str: 
        """Only contains printable parts of the event."""
        return f"Event(type={self.type}, channel={self.channel}, text={self.text}, meta={self.meta})"

class ChannelAdapter(Protocol):
    # Capabilities helper 
    capabilities: Set[str] # e.g. {"text", "image", "file", "buttons", "rich"}

    async def send(self, event: OutEvent) -> None:
        """
        Send an outgoing event to the appropriate channel.
        E.g., print to console, post to Slack, enqueue in UI, etc.
        """
        pass

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict
from urllib.parse import urlparse

EventType = Literal[
    "agent.message",
    "agent.message.update",  # simple text messages
    "agent.stream.start",
    "agent.stream.delta",
    "agent.stream.end",  # streaming messages
    "agent.progress.start",
    "agent.progress.update",
    "agent.progress.end",  # progress bar
    "agent.tool.activity",
    "session.need_input",
    "session.need_approval",
    "session.waiting",
    "structured.output",
    "file.upload",
]


class ChannelRoutingError(RuntimeError):
    """Structured failure raised when an exact channel route cannot be resolved."""

    def __init__(
        self,
        *,
        code: Literal["channel.origin_required", "channel.adapter_not_found"],
        message: str,
        channel_key: str | None = None,
        known_prefixes: tuple[str, ...] = (),
    ) -> None:
        """Create a stable channel-routing failure.

        Examples:
            Report a missing run origin:
            ```python
            ChannelRoutingError(
                code="channel.origin_required",
                message="A run origin is required.",
            )
            ```

            Report an unavailable adapter:
            ```python
            ChannelRoutingError(
                code="channel.adapter_not_found",
                message="No adapter is registered for 'slack'.",
                channel_key="slack:team/T:chan/C",
                known_prefixes=("console", "ui"),
            )
            ```

        Args:
            code: Stable machine-readable failure code.
            message: Human-readable failure explanation.
            channel_key: Exact channel address involved, when available.
            known_prefixes: Sorted adapter prefixes available to the bus.

        Returns:
            None.

        Notes:
            The structured fields are intended for host and endpoint error
            projection. The exception message is retained for logs and tests.
        """
        self.code = code
        self.channel_key = channel_key
        self.known_prefixes = known_prefixes
        super().__init__(message)


class PhaseRich(TypedDict, total=False):
    kind: Literal["phase"]
    phase: str  # "routing", "planning", "reasoning", "tools", "reply"
    phase_key: str | None
    phase_key_source: Literal["explicit", "event"] | None
    status: Literal["pending", "active", "done", "failed", "skipped"]
    label: str | None  # short human label
    detail: str | None  # optional extra text
    code: str | None  # internal code like "routing.planning"
    phase_group_id: str | None
    phase_seq: int | None
    phase_event_id: str | None
    phase_updated_at: float | None


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


@dataclass(frozen=True, slots=True)
class ChoiceResult:
    """Typed normalized result returned by a Channel choice interaction."""

    choice: str | None
    choice_label: str | None
    text: str = ""
    matched: bool = False


@dataclass(frozen=True, slots=True)
class FileInteractionResult:
    """Typed normalized result returned by a Channel file interaction."""

    text: str = ""
    files: tuple[FileRef, ...] = ()


@dataclass
class Button:
    label: str
    value: str | None = None
    url: str | None = None
    style: Literal["primary", "danger", "default"] | None = None  # for slack buttons


@dataclass(frozen=True, slots=True)
class ChannelAttachment:
    """Artifact-backed attachment in one transport-neutral Channel message."""

    artifact_id: str
    presentation: Literal["auto", "file", "image"] = "auto"
    title: str = ""
    alt_text: str = ""

    def __post_init__(self) -> None:
        artifact_id = _bounded_text(self.artifact_id, "artifact_id", maximum=512)
        title = _bounded_text(self.title, "title", maximum=512, allow_empty=True)
        alt_text = _bounded_text(self.alt_text, "alt_text", maximum=2_000, allow_empty=True)
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "alt_text", alt_text)


@dataclass(frozen=True, slots=True)
class ChannelAction:
    """Non-blocking action rendered with one logical Channel message."""

    kind: Literal["suggested_reply", "external_link"]
    label: str
    value: str = ""
    href: str = ""
    style: Literal["primary", "danger", "default"] = "default"

    def __post_init__(self) -> None:
        label = _bounded_text(self.label, "label", maximum=255)
        value = _bounded_text(self.value, "value", maximum=4_000, allow_empty=True)
        href = _bounded_text(self.href, "href", maximum=4_000, allow_empty=True)
        if self.kind == "suggested_reply":
            if not value:
                raise ValueError("suggested_reply requires a non-empty value")
            if href:
                raise ValueError("suggested_reply forbids href")
        elif self.kind == "external_link":
            if value:
                raise ValueError("external_link forbids value")
            parsed = urlparse(href)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("external_link requires an absolute HTTP(S) href")
        else:
            raise ValueError(f"Unsupported Channel action kind: {self.kind}")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "href", href)


@dataclass(frozen=True, slots=True)
class ChannelMessage:
    """One logical assistant message independent of physical channel fan-out."""

    message_id: str
    text_markdown: str = ""
    attachments: tuple[ChannelAttachment, ...] = ()
    actions: tuple[ChannelAction, ...] = ()

    def __post_init__(self) -> None:
        message_id = _bounded_text(self.message_id, "message_id", maximum=512)
        text_markdown = _bounded_text(
            self.text_markdown,
            "text_markdown",
            maximum=200_000,
            allow_empty=True,
        )
        attachments = tuple(self.attachments)
        actions = tuple(self.actions)
        if not text_markdown and not attachments and not actions:
            raise ValueError("ChannelMessage requires text, attachments, or actions")
        attachment_ids = [item.artifact_id for item in attachments]
        if len(attachment_ids) != len(set(attachment_ids)):
            raise ValueError("ChannelMessage contains a duplicate attachment artifact_id")
        action_ids = [(item.kind, item.label, item.value, item.href) for item in actions]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("ChannelMessage contains a duplicate action")
        if len(attachments) > 64:
            raise ValueError("ChannelMessage cannot contain more than 64 attachments")
        if len(actions) > 32:
            raise ValueError("ChannelMessage cannot contain more than 32 actions")
        object.__setattr__(self, "message_id", message_id)
        object.__setattr__(self, "text_markdown", text_markdown)
        object.__setattr__(self, "attachments", attachments)
        object.__setattr__(self, "actions", actions)


@dataclass(frozen=True, slots=True)
class ChannelDeliveryReceipt:
    """Durable identities returned after one logical Channel message is delivered."""

    message_id: str
    event_cursors: tuple[int, ...] = ()
    provider_delivery_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_id",
            _bounded_text(self.message_id, "message_id", maximum=512),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.event_cursors
        ):
            raise ValueError("event_cursors must contain non-negative integers")
        provider_ids = tuple(
            _bounded_text(value, "provider_delivery_id", maximum=512)
            for value in self.provider_delivery_ids
        )
        object.__setattr__(self, "event_cursors", tuple(self.event_cursors))
        object.__setattr__(self, "provider_delivery_ids", provider_ids)


def _bounded_text(
    value: str,
    name: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    if len(normalized) > maximum:
        raise ValueError(f"{name} cannot exceed {maximum} characters")
    return normalized


@dataclass(frozen=True)
class ChoiceOption:
    id: str
    label: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


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
    attachments: list[dict[str, Any]] | None = None
    actions: list[ChannelAction] | None = None
    upsert_key: str | None = None  # for idempotent upserts, e.g., message ID to update same message

    def to_printable(self) -> str:
        """Only contains printable parts of the event."""
        return (
            f"Event(type={self.type}, channel={self.channel}, text={self.text}, meta={self.meta})"
        )


class ChannelAdapter(Protocol):
    # Capabilities helper
    capabilities: set[str]  # e.g. {"text", "image", "file", "buttons", "rich"}

    async def send(self, event: OutEvent) -> dict | None:
        """
        Send an outgoing event to the appropriate channel.
        E.g., print to console, post to Slack, enqueue in UI, etc.
        """
        pass

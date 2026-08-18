"""Provider-neutral runtime continuation values."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ContinuationStatus(StrEnum):
    """Runtime lifecycle state for a durable continuation."""

    WAITING = "waiting"
    RESUMED = "resumed"
    CANCELED = "canceled"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class Correlator:
    """Platform-agnostic correlation key for continuations."""

    scheme: str
    channel: str
    thread: str = ""
    message: str = ""

    def key(self) -> tuple[str, str, str, str]:
        """Return the normalized exact lookup identity.

        Intro:
            Normalizes optional transport components without changing their meaning.

        Examples:
            Build a complete key:
            ```python
            key = Correlator("slack", "C1", "T1", "M1").key()
            ```

            Build a thread-root key:
            ```python
            key = Correlator("slack", "C1", "T1").key()
            ```

        Args:
            None.

        Returns:
            tuple[str, str, str, str]: Exact normalized correlator components.

        Notes:
            The tuple is suitable for an indexed equality lookup; it is not a prefix.
        """
        return (self.scheme, self.channel, self.thread or "", self.message or "")


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationDraft:
    """Tokenless immutable content submitted for atomic continuation creation."""

    run_id: str
    node_id: str
    kind: str
    continuation_id: str = field(default_factory=lambda: f"cont-{uuid4().hex}")
    prompt: str | None = None
    resume_schema: dict[str, Any] | None = None
    deadline: datetime | None = None
    poll: dict[str, Any] | None = None
    next_wakeup_at: datetime | None = None
    attempts: int = 0
    channel: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] | None = None
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = field(
        default=None,
        metadata={"deprecated": True, "compatibility_only": True},
    )
    graph_id: str | None = None
    correlators: tuple[Correlator, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Continuation:
    """Tokenless revisioned continuation returned by a runtime store."""

    continuation_id: str
    revision: int
    run_id: str
    node_id: str
    kind: str
    status: ContinuationStatus = ContinuationStatus.WAITING
    prompt: str | None = None
    resume_schema: dict[str, Any] | None = None
    deadline: datetime | None = None
    poll: dict[str, Any] | None = None
    next_wakeup_at: datetime | None = None
    attempts: int = 0
    channel: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    closed_at: datetime | None = None
    payload: dict[str, Any] | None = None
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = field(
        default=None,
        metadata={"deprecated": True, "compatibility_only": True},
    )
    graph_id: str | None = None
    correlators: tuple[Correlator, ...] = ()

    @property
    def closed(self) -> bool:
        """Report whether this record is terminal.

        Intro:
            Derives legacy closed semantics from the explicit lifecycle status.

        Examples:
            Inspect a waiting record:
            ```python
            assert waiting.closed is False
            ```

            Inspect a resumed record:
            ```python
            assert resumed.closed is True
            ```

        Args:
            None.

        Returns:
            bool: `True` for terminal states and `False` while waiting.

        Notes:
            Closed is derived and is never persisted as an independent flag.
        """
        return self.status is not ContinuationStatus.WAITING

    def to_dict(self) -> dict[str, Any]:
        """Serialize a continuation without bearer-token material.

        Intro:
            Produces the provider-neutral runtime document used by legacy stores.

        Examples:
            Serialize a waiting record:
            ```python
            payload = waiting.to_dict()
            ```

            Confirm token material is absent:
            ```python
            assert "token" not in waiting.to_dict()
            ```

        Args:
            None.

        Returns:
            dict[str, Any]: JSON-ready tokenless continuation content.

        Notes:
            Deprecated App identity is emitted only inside explicit compatibility metadata.
        """
        metadata: dict[str, Any] = {}
        if self.app_id:
            metadata["compatibility_metadata"] = {
                "app_id": {
                    "value": self.app_id,
                    "deprecated": True,
                    "scheduled_removal": "future breaking release",
                }
            }
        return {
            "continuation_id": self.continuation_id,
            "revision": self.revision,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "status": self.status.value,
            "prompt": self.prompt,
            "resume_schema": self.resume_schema,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "poll": self.poll,
            "next_wakeup_at": self.next_wakeup_at.isoformat() if self.next_wakeup_at else None,
            "attempts": self.attempts,
            "channel": self.channel,
            "created_at": self.created_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "payload": self.payload,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "graph_id": self.graph_id,
            "correlators": [
                {
                    "scheme": value.scheme,
                    "channel": value.channel,
                    "thread": value.thread,
                    "message": value.message,
                }
                for value in self.correlators
            ],
            "metadata": metadata,
        }


@dataclass(slots=True)
class CreatedContinuation:
    """Atomic creation result containing a tokenless record and one-time raw token."""

    record: Continuation
    token: str

    def __getattr__(self, name: str) -> Any:
        """Expose record fields while issuance code still needs the one-time token.

        Intro:
            Delegates non-token attributes to the immutable continuation record.

        Examples:
            Read the issued run identity:
            ```python
            run_id = created.run_id
            ```

            Read the issued prompt:
            ```python
            prompt = created.prompt
            ```

        Args:
            name: Requested continuation record attribute.

        Returns:
            Any: Value from the immutable tokenless record.

        Notes:
            Only the issuance envelope carries the raw token; persistence receives `record`.
        """
        return getattr(self.record, name)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContinuationQuery:
    """Bounded indexed runtime continuation query."""

    session_id: str | None = None
    statuses: tuple[ContinuationStatus, ...] = (ContinuationStatus.WAITING,)
    kinds: tuple[str, ...] = ()
    correlator: Correlator | None = None
    due_at_or_before: datetime | None = None
    open_at: datetime | None = None
    limit: int = 100
    cursor: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not 1 <= self.limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        for name in ("due_at_or_before", "open_at"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ContinuationPage:
    """One bounded runtime continuation result page."""

    items: tuple[Continuation, ...]
    next_cursor: str | None = None

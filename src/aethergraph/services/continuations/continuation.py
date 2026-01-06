from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Correlator:
    """Platform-agnostic correlation key for continuations."""

    scheme: str  # e.g., "slack", "web", "email"
    channel: str  # e.g., channel ID, email address
    thread: str  # e.g., thread ID, conversation ID
    message: str  # e.g., message ID, timestamp

    def key(self) -> tuple[str, str, str, str]:
        return (self.scheme, self.channel, self.thread or "", self.message or "")


@dataclass
class Continuation:
    """Represents a continuation of a process or workflow."""

    run_id: str
    node_id: str
    kind: str
    token: str
    prompt: str | None = None
    resume_schema: dict | None = None
    deadline: datetime | None = None
    poll: dict | None = None
    next_wakeup_at: datetime | None = None
    attempts: int = 0
    channel: str | None = None
    created_at: datetime = datetime.utcnow()
    closed: bool = False  # ← NEW
    payload: dict[str, Any] | None = None  # set at creation time

    # new session, etc.
    session_id: str | None = None
    agent_id: str | None = None
    app_id: str | None = None
    graph_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "token": self.token,
            "prompt": self.prompt,
            "resume_schema": self.resume_schema,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "poll": self.poll,
            "next_wakeup_at": self.next_wakeup_at.isoformat() if self.next_wakeup_at else None,
            "attempts": self.attempts,
            "channel": self.channel,
            "created_at": self.created_at.isoformat(),
            "closed": self.closed,
            "payload": self.payload,
        }

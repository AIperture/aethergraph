from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

@dataclass(frozen=True)
class Correlator:
    """Platform-agnostic correlation key for continuations."""
    scheme: str  # e.g., "slack", "web", "email"
    channel: str  # e.g., channel ID, email address
    thread: str  # e.g., thread ID, conversation ID
    message: str  # e.g., message ID, timestamp

    def key(self) -> Tuple[str, str, str, str]:
        return (self.scheme, self.channel, self.thread or "", self.message or "")

@dataclass
class Continuation:
    """Represents a continuation of a process or workflow."""
    run_id: str
    node_id: str
    kind: str  
    token: str
    prompt: Optional[str] = None
    resume_schema: Optional[Dict] = None
    deadline: Optional[datetime] = None
    poll: Optional[Dict] = None
    next_wakeup_at: Optional[datetime] = None
    attempts: int = 0
    channel: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    closed: bool = False  # ← NEW
    payload: Optional[Dict[str, Any]] = None  # set at creation time

    def to_dict(self) -> Dict[str, Any]:
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

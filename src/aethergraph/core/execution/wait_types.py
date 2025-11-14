from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_WAIT_KEY = "__wait__"


@dataclass
class WaitSpec:
    kind: str = "external"  # "human" | "ask_text" | "external" | "time" | "event" | ... This is more generic than channel wait kinds
    prompt: dict[str, Any] | str | None = None  # for human/robot
    resume_schema: dict[str, Any] | None = None  # for human/robot validation
    channel: str | None = None  # for external/event
    deadline: datetime | str | None = None  # ISO timestamp or datetime
    poll: dict[str, Any] | None = (
        None  # {"interval_sec": 30, "endpoint": "...", "extract": "$.path"}
    )

    # resume handles
    token: str | None = None  # internal opaque continuation id (do NOT expose to untrusted clients)
    resume_key: str | None = None  # short alias safe to surface in UI/buttons
    notified: bool = False  # internal flag: whether continuation notification has been sent out
    inline_payload: dict[str, Any] | None = (
        None  # internal: optional inline payload returned from notification step
    )

    # Optional grab-bag for extensions; avoids new fields churn later
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        # Only include non-None fields to preserve backward compatibility with consumers
        d = {
            "kind": self.kind,
            "prompt": self.prompt,
            "resume_schema": self.resume_schema,
            "channel": self.channel,
            "deadline": self.deadline,
            "poll": self.poll,
            "token": self.token,
            "resume_key": self.resume_key,
            "notified": self.notified,
            "inline_payload": self.inline_payload,
            "meta": self.meta or None,
        }
        return {k: v for k, v in d.items() if v is not None}

    def sanitized_for_transport(self) -> dict[str, Any]:
        """
        Strip sensitive fields for UI/adapters/webhooks.
        Prefer exposing `resume_key` (short alias) over raw `token`.
        """
        d = self.to_dict()
        d.pop("token", None)
        return d


def wait_sentinel(spec: WaitSpec | dict[str, Any]) -> dict[str, Any]:
    """Return the canonical sentinel the executor understands as 'please wait'."""
    return {_WAIT_KEY: spec if isinstance(spec, dict) else spec.__dict__}


class WaitRequested(RuntimeError):
    """Exception to raise from a tool to indicate it wants to wait."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        super().__init__(f"Wait requested: {spec}")

    def to_dict(self):
        return self.spec if isinstance(self.spec, dict) else self.spec.to_dict()

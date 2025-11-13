from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime

_WAIT_KEY = "__wait__"

@dataclass
class WaitSpec:
    kind: str = "external"                         # "human" | "ask_text" | "external" | "time" | "event" | ... This is more generic than channel wait kinds
    prompt: Optional[Dict[str, Any] | str] = None  # for human/robot
    resume_schema: Optional[Dict[str, Any]] = None # for human/robot validation
    channel: Optional[str] = None                  # for external/event
    deadline: Optional[datetime | str] = None      # ISO timestamp or datetime
    poll: Optional[Dict[str, Any]] = None          # {"interval_sec": 30, "endpoint": "...", "extract": "$.path"}

    # resume handles
    token: Optional[str] = None                    # internal opaque continuation id (do NOT expose to untrusted clients)
    resume_key: Optional[str] = None               # short alias safe to surface in UI/buttons
    notified: bool = False                           # internal flag: whether continuation notification has been sent out
    inline_payload: Optional[Dict[str, Any]] = None  # internal: optional inline payload returned from notification step

    # Optional grab-bag for extensions; avoids new fields churn later
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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

    def sanitized_for_transport(self) -> Dict[str, Any]:
        """
        Strip sensitive fields for UI/adapters/webhooks.
        Prefer exposing `resume_key` (short alias) over raw `token`.
        """
        d = self.to_dict()
        d.pop("token", None)
        return d

def wait_sentinel(spec: WaitSpec | Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical sentinel the executor understands as 'please wait'."""
    return {_WAIT_KEY: spec if isinstance(spec, dict) else spec.__dict__}

class WaitRequested(RuntimeError):
    """ Exception to raise from a tool to indicate it wants to wait. """
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec
        super().__init__(f"Wait requested: {spec}")

    def to_dict(self): 
        return self.spec if isinstance(self.spec, dict) else self.spec.to_dict()
from dataclasses import dataclass, field
from typing import Any, Literal

NodeWaitingKind = Literal["human", "robot", "external", "time", "event"]


class NodeStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    CANCELLED = "CANCELLED"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"
    WAITING_HUMAN = "WAITING_HUMAN"
    WAITING_ROBOT = "WAITING_ROBOT"
    WAITING_EXTERNAL = "WAITING_EXTERNAL"
    WAITING_TIME = "WAITING_TIME"
    WAITING_EVENT = "WAITING_EVENT"

    @classmethod
    def from_kind(cls, kind: NodeWaitingKind) -> str:
        """Map waiting kind to status."""
        return {
            "human": cls.WAITING_HUMAN,
            "approval": cls.WAITING_HUMAN,
            "user_approval": cls.WAITING_HUMAN,  # alias to keep backward compatibility
            "user_input": cls.WAITING_HUMAN,
            "user_files": cls.WAITING_HUMAN,
            "robot": cls.WAITING_ROBOT,
            "external": cls.WAITING_EXTERNAL,
            "time": cls.WAITING_TIME,
            "event": cls.WAITING_EVENT,
        }[kind]

    @classmethod
    def is_waiting(cls, status: str) -> bool:
        return status.startswith("WAITING_")


TERMINAL_STATES = {NodeStatus.DONE, NodeStatus.FAILED, NodeStatus.SKIPPED}
WAITING_STATES = {
    NodeStatus.WAITING_HUMAN,
    NodeStatus.WAITING_ROBOT,
    NodeStatus.WAITING_EXTERNAL,
    NodeStatus.WAITING_TIME,
    NodeStatus.WAITING_EVENT,
}


@dataclass
class TaskNodeState:
    status: NodeStatus = NodeStatus.PENDING
    outputs: dict[str, any] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    next_wakeup_at: str | None = None  # ISO timestamp
    wait_token: str | None = None  # for external wait/resume with Continuation
    wait_spec: dict[str, Any] | None = None  # spec for waiting (kind, channel, meta, etc.)
    started_at: str | None = None  # ISO timestamp
    finished_at: str | None = None  # ISO timestamp

    @property
    def output(self):
        # convenience for single-output nodes
        return self.outputs.get("result")

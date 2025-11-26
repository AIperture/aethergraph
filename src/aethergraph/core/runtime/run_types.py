from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# used to represent the status of a run, primiarily used in endpoint with RunManager


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
    cancellation_requested = "cancellation_requested"


@dataclass
class RunRecord:
    """
    Core-level representation of a run.

    This is independent from any Pydantic model used by the HTTP API.
    """

    run_id: str
    graph_id: str
    kind: str  # "taskgraph" | "graphfn" | other in the future
    status: RunStatus
    started_at: datetime
    finished_at: datetime | None = None
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None
    org_id: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

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
    waiting = "waiting"
    canceled = "canceled"
    cancellation_requested = "cancellation_requested"


class RunOrigin(str, Enum):
    app = "app"  # launched from an application UI
    agent = "agent"  # launched by an AI agent
    chat = "chat"  # launched from a chat interface
    playground = "playground"  # launched from a playground environment (sidecar/SDK/local dev)
    api = "api"  # launched from an API call
    system = "system"  # launched from a system process (internal maintenance, cron job, etc.)


class RunVisibility(str, Enum):
    normal = "normal"  # visible to all users with access to the org/client
    inline = "inline"  # hidden from run listings, only shown in session / debug views
    hidden = "hidden"  # hidden from all UIs (used in quick tests, ephemeral runs, etc.)


class RunImportance(str, Enum):
    normal = "normal"  # standard run
    ephemeral = "ephemeral"  # low-importance run, noisy or temporary (may be pruned sooner)


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

    session_id: str | None = None
    origin: RunOrigin = RunOrigin.app
    visibility: RunVisibility = RunVisibility.normal
    importance: RunImportance = RunImportance.normal

    # optional agent/app linkage
    agent_id: str | None = None
    app_id: str | None = None

    # Artifact statistics
    artifact_count: int = 0
    first_artifact_at: datetime | None = None
    last_artifact_at: datetime | None = None

    # Optional: keep a small rolling window of recent artifact IDs
    recent_artifact_ids: list[str] = field(default_factory=list)

    def __item__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


# Session-related run types
class SessionKind(str, Enum):
    chat = "chat"
    playground = "playground"
    notebook = "notebook"
    pipline = "pipeline"  # future

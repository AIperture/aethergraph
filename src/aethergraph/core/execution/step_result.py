from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.continuations.continuation import Continuation


@dataclass
class StepResult:
    status: str  # NodeStatus
    outputs: dict[str, Any] | None = None  # outputs if completed
    artifacts: list[Artifact] = field(default_factory=list)
    error: str | None = None  # error message if failed
    continuation: Continuation | None = None  # continuation if waiting
    next_wakeup_at: datetime | None = None  # ISO timestamp for next wakeup (for time-based waits)

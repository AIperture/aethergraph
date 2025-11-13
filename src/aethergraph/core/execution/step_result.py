from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from aethergraph.contracts.services.artifacts import Artifact 
from aethergraph.services.continuations.continuation import Continuation

@dataclass
class StepResult:
    status: str  # NodeStatus
    outputs: Optional[Dict[str, Any]] = None  # outputs if completed
    artifacts: List[Artifact] = field(default_factory=list)
    error: Optional[str] = None  # error message if failed
    continuation: Optional[Continuation] = None  # continuation if waiting
    next_wakeup_at: Optional[datetime] = None  # ISO timestamp for next wakeup (for time-based waits)

"""Project connected-runtime events into canonical operator trace DTOs.

This package translates AG-owned runtime records into the canonical operator-UI
read model served only by ``/api/trace/v2``. It is not part of agent execution,
``NodeContext``, or the Studio backend API.
"""

from __future__ import annotations

from .models import (
    AgentSegment,
    ContextSnapshot,
    Cycle,
    Dispatch,
    PlanSnapshot,
    PlanTimeline,
    ResourceSlotTrace,
    RunNode,
    RunOutcome,
    ToolExecution,
    TraceGraph,
    TraceSessionGroup,
    TurnDetail,
    TurnSummary,
)
from .reader import TraceReader
from .service import TraceProjectionService

__all__ = [
    "AgentSegment",
    "ContextSnapshot",
    "Cycle",
    "Dispatch",
    "PlanSnapshot",
    "PlanTimeline",
    "ResourceSlotTrace",
    "RunNode",
    "RunOutcome",
    "ToolExecution",
    "TraceGraph",
    "TraceProjectionService",
    "TraceReader",
    "TraceSessionGroup",
    "TurnDetail",
    "TurnSummary",
]

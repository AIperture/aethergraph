# aethergraph/persist/interfaces.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Protocol

@dataclass
class GraphSnapshot:
    run_id: str
    graph_id: str
    rev: int
    created_at: float  # epoch seconds
    spec_hash: str     # detect spec drift
    state: Dict[str, Any]  # JSON-serializable TaskGraphState

@dataclass
class StateEvent:
    run_id: str
    graph_id: str
    rev: int
    ts: float
    kind: str  # "STATUS" | "OUTPUT" | "INPUTS_BOUND" | "PATCH"
    payload: Dict[str, Any]

class GraphStateStore(Protocol):
    async def save_snapshot(self, snap: GraphSnapshot) -> None: ...
    async def load_latest_snapshot(self, run_id: str) -> Optional[GraphSnapshot]: ...
    async def append_event(self, ev: StateEvent) -> None: ...
    async def load_events_since(self, run_id: str, from_rev: int) -> List[StateEvent]: ...
    async def list_run_ids(self, graph_id: Optional[str] = None) -> List[str]: ...


# aethergraph/persist/interfaces.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class GraphSnapshot:
    run_id: str
    graph_id: str
    rev: int
    created_at: float  # epoch seconds
    spec_hash: str  # detect spec drift
    state: dict[str, Any]  # JSON-serializable TaskGraphState


@dataclass
class StateEvent:
    run_id: str
    graph_id: str
    rev: int
    ts: float
    kind: str  # "STATUS" | "OUTPUT" | "INPUTS_BOUND" | "PATCH"
    payload: dict[str, Any]


class GraphStateStore(Protocol):
    async def save_snapshot(self, snap: GraphSnapshot) -> None: ...
    async def load_latest_snapshot(self, run_id: str) -> GraphSnapshot | None: ...
    async def append_event(self, ev: StateEvent) -> None: ...
    async def load_events_since(self, run_id: str, from_rev: int) -> list[StateEvent]: ...
    async def list_run_ids(self, graph_id: str | None = None) -> list[str]: ...

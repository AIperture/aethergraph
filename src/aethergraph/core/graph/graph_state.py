from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from .node_state import TaskNodeState
from .node_spec import TaskNodeSpec

@dataclass 
class GraphPatch:
    # Used for mutation of a graph's topology; NOT USED YET
    op: str # "add_node", "remove_node", "update_node", "add_edge", "remove_edge"
    payload: Dict[str, Any]  # details depend on op type -> To be defined later 


@dataclass
class TaskGraphState:
    run_id : str | None = None # unique run identifier, used when in agent or program execution, and run_id is known from agent/program level
    nodes: Dict[str, TaskNodeState] = field(default_factory=dict)  # node_id -> TaskNodeState
    # node_status: Dict[str, str] = field(default_factory=dict)  # node_id -> status ("pending", "running", "completed", "failed", etc.)
    # node_outputs: Dict[str, Any] = field(default_factory=dict)  # node_id -> output data
    _bound_inputs: Optional[Dict[str, Any]] = field(default=None, repr=False)  # inputs bound at runtime
    rev: int = 0  # revision number, incremented on each mutation
    patches: List[GraphPatch] = field(default_factory=list, repr=False)  # list of patches applied to the graph

    def default_node_states(self, spec: TaskNodeSpec):
        # Initialize node states based on the given spec
        for nid in spec.nodes.keys():
            if nid not in self.nodes:
                self.nodes[nid] = TaskNodeState()
                self.node_status[nid] = "PENDING"

    def summary_line(self) -> str:
        from collections import Counter
        sc = Counter(self.node_status.values())
        counts = ", ".join(f"{k}={v}" for k,v in sorted(sc.items())) or "—"
        bound = list(self._bound_inputs.keys()) if self._bound_inputs else "—"
        return f"bound_inputs={bound}, node_outputs={len(self.node_outputs)}, status_counts: {counts}"

    @property
    def node_statuses(self) -> Dict[str, str]:
        return {nid: ns.status for nid, ns in self.nodes.items()}
    
    # alias to node_statuses
    @property
    def node_status(self) -> Dict[str, str]:
        return {nid: ns.status for nid, ns in self.nodes.items()}
    
    @property
    def node_outputs(self) -> Dict[str, Any]:
        return {nid: ns.outputs for nid, ns in self.nodes.items() if ns.outputs}
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import inspect

from .graph_io import IOBindings, IOSpec
from .node_spec import TaskNodeSpec


@dataclass
class TaskGraphSpec:
    graph_id: str 
    version: str = "0.1.0"
    nodes: Dict[str, TaskNodeSpec] = field(default_factory=dict)  # node_id -> TaskNodeSpec
    io: IOSpec = field(default_factory=IOSpec)  # inputs/outputs
    bindings: Optional[IOBindings] = None  # input/output bindings
    meta: Dict[str, Any] = field(default_factory=dict)  # additional metadata

    def canonical(self) -> str:
        return f"graph:{self.graph_id}@{self.version}"
    
    @property
    def inputs_required(self) -> set[str]: return set(self.io.required.keys())
    @property
    def inputs_optional(self) -> Dict[str, Any]: return {k: p.default for k,p in self.io.optional.items()}
    @property
    def outputs(self) -> Dict[str, Any]: return {k: p.default for k,p in self.io.outputs.items()}

    def io_summary_lines(self) -> List[str]:
        return [
            f"required: {_fmt_set(self.inputs_required)}",
            f"optional: {_fmt_opt_map(self.inputs_optional, show_values=False)}",
            f"outputs:  {_fmt_outputs_map(self.outputs)}",
        ]


@dataclass
class GraphView:
    """ A read-only view of the graph's spec and state. """
    graph_id: str
    nodes: Dict[str, Any]  # node_id -> TaskNodeRuntime
    metadata: Dict[str, Any] = field(default_factory=dict)  # Optional metadata

    # helpers, no mutation
    def get_dependents(self, nid: str) -> List[str]:
        """ Get list of node_ids that depend on the given node_id. """
        return [x.node_id for x in self.nodes.values() if nid in x.dependencies]
    
    def get_root_nodes(self) -> List[str]:
        """ Get list of root node_ids (no dependencies). """
        return [x.node_id for x in self.nodes.values() if not x.dependencies]
    
    

# ---------- helpers for printing and debugging ----------

def _short(x: Any, maxlen: int = 42) -> str:
    s = str(x)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"

def _status_label(s: Any) -> str:
    # Accept Enum-like (with .name), strings, or None
    if s is None:
        return "-"
    return getattr(s, "name", str(s))

def _logic_label(logic: Any) -> str:
    # Show a dotted path when possible; fall back to repr/str
    if isinstance(logic, str):
        return logic
    # Unwrap @tool proxies if present
    impl = getattr(logic, "__aether_impl__", logic)
    if inspect.isfunction(impl) or inspect.ismethod(impl):
        mod = getattr(impl, "__module__", None) or ""
        name = getattr(impl, "__name__", None) or "tool"
        return f"{mod}.{name}".strip(".")
    return _short(repr(logic), 80)

def _fmt_set(xs: Optional[set]) -> str:
    return ", ".join(sorted(map(str, xs))) if xs else "—"

def _fmt_opt_map(d: Optional[dict], *, show_values: bool = False, maxval: int = 26) -> str:
    if not d:
        return "—"
    if show_values:
        items = [f"{k}={_short(v, maxval)}" for k, v in d.items()]
    else:
        items = list(map(str, d.keys()))
    return ", ".join(sorted(items)) if items else "—"

def _fmt_outputs_map(d: Optional[dict]) -> str:
    """
    Show graph outputs mapping; if a value looks like a Ref(node_id, key),
    render as 'out_key ← node_id.key'. Otherwise, just list keys.
    """
    if not d:
        return "—"
    parts = []
    for out_k, v in d.items():
        # duck-typed Ref
        if hasattr(v, "node_id") and hasattr(v, "key"):
            parts.append(f"{out_k} ← {getattr(v, 'node_id')}.{getattr(v, 'key')}")
        else:
            parts.append(str(out_k))
    return ", ".join(sorted(parts))

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
import itertools
from typing import Any
import uuid

from .graph_refs import GRAPH_INPUTS_NODE_ID, RESERVED_INJECTABLES
from .graph_spec import TaskGraphSpec
from .node_spec import TaskNodeSpec
from .task_graph import TaskGraph

_GRAPH_CTX: ContextVar[GraphBuilder | None] = ContextVar("_GRAPH_CTX", default=None)  # Async-safe


def current_builder() -> GraphBuilder | None:
    return _GRAPH_CTX.get()


class GraphBuilder:
    _auto_counter = itertools.count(1)

    def __init__(self, *, name: str = "default_graph"):
        self.spec = TaskGraphSpec(graph_id=name, nodes={}, meta={})
        self.graph = TaskGraph(spec=self.spec)
        self.graph.ensure_inputs_node()

        self._auto_counter_by_logic = {}  # logic_name -> counter

        # index for quick lookup
        self._alias_index: dict[str, str] = {}  # alias -> node_id
        self._logic_index: dict[str, list[str]] = {}  # logic -> [node_id, ...]
        self._label_index: dict[str, list[str]] = {}  # label -> {node_id, ...}

    def add_node(self, node_spec: TaskNodeSpec) -> str:
        if node_spec.node_id in self.spec.nodes:
            raise ValueError(
                f"Node ID '{node_spec.node_id}' already exists in graph '{self.spec.graph_id}'"
            )
        self.spec.nodes[node_spec.node_id] = node_spec
        return self

    def add_tool_node(
        self,
        *,
        node_id: str,
        logic: str,
        inputs: dict,
        expected_input_keys: Iterable[str] | None = None,
        expected_output_keys: Iterable[str] | None = None,
        after: Iterable[str] | None = None,
        inject: list[str] | None = None,
        tool_name: str | None = None,
        tool_version: str | None = None,
    ) -> GraphBuilder:
        """Add a tool node to the graph."""

        if node_id in self.spec.nodes:
            raise ValueError(f"Node with id {node_id} already exists in the graph.")

        # Initialize injection and pure input mappings. Injection is for reserved keywords that should be passed from the context.
        deps = set(after or [])
        inject = inject or []
        pure_inputs = {}
        has_arg = False

        for k, v in list(inputs.items()):
            if k in RESERVED_INJECTABLES:
                inject.append(k)
            else:
                pure_inputs[k] = v

        # infer dependencies from input Refs
        def _walk_refs(x):
            # Recursively walk input bindings to find Ref dependencies
            nonlocal has_arg
            if isinstance(x, dict):
                if x.get("_type") == "ref" and "from" in x:
                    yield x["from"]
                elif x.get("_type") == "arg":
                    has_arg = True
                else:
                    for v in x.values():
                        yield from _walk_refs(v)
            elif isinstance(x, list | tuple):
                for v in x:
                    yield from _walk_refs(v)

        deps = set(_walk_refs(pure_inputs))
        if has_arg:
            deps.add(GRAPH_INPUTS_NODE_ID)  # ensure inputs node is a dependency
        if after:
            for a in after:
                deps.add(a.node_id if hasattr(a, "node_id") else a)

        node = TaskNodeSpec(
            node_id=node_id,
            type="tool",
            logic=logic,
            inputs=inputs,
            dependencies=list(deps),
            expected_input_keys=expected_input_keys,
            expected_output_keys=expected_output_keys,
            metadata={},
            tool_name=tool_name or logic or "unknown_tool",
            tool_version=tool_version,  # could be set to a version string if available
        )
        return self.add_node(node)

    def ensure_inputs_node(self):
        """Ensure the special inputs node exists in the graph."""
        if GRAPH_INPUTS_NODE_ID not in self.spec.nodes:
            self.spec.nodes[GRAPH_INPUTS_NODE_ID] = TaskNodeSpec(
                node_id=GRAPH_INPUTS_NODE_ID,
                type="inputs",
                logic=None,
                inputs={},
                dependencies=[],
                expected_input_keys=[],
                expected_output_keys=[],
                metadata={"synthetic": True},
            )
        return self

    def freeze(self) -> TaskGraphSpec:
        """Frozen dataclass / validate topo order"""
        return self.spec

    def expose(self, name: str, value: Any):
        self.graph.expose(name, value)

    # ---- ids and utils ----
    def next_id(self, logic_name: str | None = None) -> str:
        """Generate a unique node ID."""
        base = (logic_name or "node").rstrip("_")
        return f"{base}_{next(self._auto_counter)}_{uuid.uuid4().hex[:6]}"

    def _next_readable_id(self, logic_name: str | None = None) -> str:
        """Generate a more human-readable node ID, but may not be unique."""
        n = self._auto_counter_by_logic.get(logic_name, 0) + 1
        self._auto_counter_by_logic[logic_name] = n
        return f"{logic_name}_{n}"  # deterministic and readable

    def to_graph(self) -> TaskGraph:
        self.graph.spec.metadata["graph_io"] = self.graph.io_signature()
        return self.graph

    def register_alias(self, alias: str, node_id: str):
        if alias in self._alias_index and self._alias_index[alias] != node_id:
            raise ValueError(
                f"Alias '{alias}' already registered for node '{self._alias_index[alias]}', cannot re-register for '{node_id}'"
            )
        self._alias_index[alias] = node_id

    def register_logic_name(self, logic_name: str, node_id: str):
        self._logic_index.setdefault(logic_name, []).append(node_id)

    def register_labels(self, labels: Iterable[str], node_id: str):
        for label in labels or []:
            self._label_index.setdefault(label, set()).add(node_id)

    # ergonomic accessors
    def find_by_alias(self, alias: str) -> str | None:
        return self._alias_index.get(alias)

    def find_by_logic(self, logic_prefix: str) -> list[str]:
        exact = self._logic_index.get(logic_prefix, [])
        if exact:
            return list(exact)
        # fuzzy match: logic_name contained in key
        out = []
        for k, v in self._logic_index.items():
            if k.startswith(logic_prefix):
                out.extend(v)
        return out

    def find_by_label(self, label: str) -> list[str]:
        return sorted(self._label_index.get(label, set()))


@contextmanager
def graph(*, name: str = "default_graph"):
    """Context manager that yields a GraphBuilder to build a TaskGraph."""
    builder = GraphBuilder(name=name)
    token = _GRAPH_CTX.set(builder)
    try:
        yield builder.graph
    finally:
        builder.graph.__post_init__()  # reify runtime nodes
        _GRAPH_CTX.reset(token)

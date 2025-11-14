# aethergraph/core/execution/context.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aethergraph.core.graph.task_node import TaskNodeRuntime

from aethergraph.services.clock.clock import SystemClock
from aethergraph.services.logger.std import StdLoggerService
from aethergraph.services.resume.router import ResumeRouter

from ..graph.graph_refs import GRAPH_INPUTS_NODE_ID, RESERVED_INJECTABLES
from .bound_memory import BoundMemoryAdapter
from .node_context import NodeContext
from .node_services import NodeServices


@dataclass
class ExecutionContext:
    run_id: str
    graph_id: str | None
    graph_inputs: dict[str, Any]
    outputs_by_node: dict[str, dict[str, Any]]
    services: NodeServices
    logger_factory: StdLoggerService
    clock: SystemClock
    resume_payload: dict[str, Any] | None = None
    should_run_fn: Callable[[], bool] | None = None
    resume_router: ResumeRouter | None = None  # ResumeRouter

    # Back-compat shim
    bound_memory: BoundMemoryAdapter | None = None

    def create_node_context(self, node: TaskNodeRuntime) -> NodeContext:
        return NodeContext(
            run_id=self.run_id,
            graph_id=self.graph_id or "",
            node_id=node.node_id,
            services=self.services,
            resume_payload=self.resume_payload,
            # back-compat for old ctx.mem()
            bound_memory=self.bound_memory,
        )

    # def as_node_context(self, ad) -> "NodeContext":
    #     """ Create a NodeContext representing this execution context itself as a node.
    #         Useful for ad-hoc contexts that don't have real nodes.
    #     """
    #     return NodeContext(
    #         run_id=self.run_id,
    #         graph_id=self.graph_id or "",
    #         node_id="ad_",
    #         services=self.services,
    #         resume_payload=self.resume_payload,
    #         # back-compat for old ctx.mem()
    #         bound_memory=self.bound_memory,
    #     )

    # ----- helpers used by step forward() -----
    def now(self) -> datetime:
        return self.clock.now()

    def resolve(self, logic_ref: str):
        """Resolve a logic reference to a callable.
        NOTE: This is not used anymore; prefer get_logic().
        """
        # fallback dotted import
        mod, _, attr = logic_ref.rpartition(".")
        return getattr(importlib.import_module(mod), attr)

    def get_logic(self, logic_ref):
        """Resolve a logic reference to a callable.
        If a registry is available and the ref looks like a registry key, use it.
        Otherwise, if a dotted path, import it.
        Otherwise, return as-is (assumed callable).
        Args:
            logic_ref: A callable, dotted path string, or registry key string.
        Returns:
            The resolved callable.
        """
        if isinstance(logic_ref, str) and logic_ref.startswith("registry:") and self.registry:
            # registry key
            return self.registry.get_logic_ref(logic_ref)
        if isinstance(logic_ref, str):
            # dotted path fallback
            mod, _, attr = logic_ref.rpartition(".")
            return getattr(importlib.import_module(mod), attr)
        return logic_ref

    async def resolve_inputs(self, node) -> dict[str, Any]:
        """
        Materialize a node's input mapping by resolving:
        - {"_type":"arg","key":K} → graph input value (or optional default)
        - {"_type":"ref","from":NODE_ID,"key":OUT} → upstream node's output value
        - {"_type":"context","key":K,"default":D} → memory value (or D if missing)
        Works recursively over dicts/lists/tuples.

        The function works as follows:
         - If the value is a dict with "_type" of "arg", it looks up the graph input.
         - If the value is a dict with "_type" of "ref", it looks up
        the specified node's output.
         - If the value is a dict without special keys, it recursively resolves
           each key-value pair.
         - If the value is a list or tuple, it recursively resolves each element.
         - Otherwise, it returns the value as-is (assumed to be a constant).

        Args:
            node: The TaskNodeRuntime whose inputs to resolve.
        Returns:
            The fully resolved inputs dict for the node.
        Raises:
            KeyError: If a referenced graph input or node output is missing.
        """
        raw = getattr(node, "inputs", {}) or {}
        # Grab optional defaults from the graph spec if available
        opt_defaults: dict[str, Any] = {}
        parent_graph = getattr(node, "_parent_graph", None)
        if parent_graph and getattr(parent_graph, "spec", None):
            # _io_inputs_optional is a dict[str, Any]
            opt_defaults = getattr(parent_graph.spec, "inputs_optional", {}) or {}

        # Allow a fallback to graph.state.node_outputs if scheduler hasn't copied yet
        fallback_outputs = {}
        if parent_graph and getattr(parent_graph, "state", None):
            fallback_outputs = getattr(parent_graph.state, "node_outputs", {}) or {}

        def _err_path(msg: str, path: str):
            raise KeyError(
                f"{msg} (node={getattr(node, 'node_id', getattr(node, 'id', '?'))}, path={path})"
            )

        def _resolve_arg(marker: dict[str, Any], path: str):
            k = marker.get("key")
            if k is None:
                _err_path("Bad arg marker (missing 'key')", path)
            if k in self.graph_inputs:
                return self.graph_inputs[k]
            if k in opt_defaults:
                return opt_defaults[k]
            # Helpful error: show known keys
            known = list(self.graph_inputs.keys())
            _err_path(f"Graph input '{k}' not provided (known inputs: {known})", path)

        def _resolve_ref(marker: dict[str, Any], path: str):
            src = marker.get("from")
            out_key = marker.get("key")
            if src is None or out_key is None:
                _err_path("Bad ref marker (need 'from' and 'key')", path)

            # Tolerate someone emitting a ref to the inputs sentinel
            if src == GRAPH_INPUTS_NODE_ID:
                # Interpret as an 'arg' reference to graph inputs
                return _resolve_arg({"_type": "arg", "key": out_key}, path + ".__graph_inputs__")

            # Primary source: env.outputs_by_node (scheduler publishes here)
            if src in self.outputs_by_node:
                outs = self.outputs_by_node[src] or {}
                if out_key in outs:
                    return outs[out_key]
                _err_path(
                    f"Upstream node '{src}' has no output key '{out_key}'. "
                    f"Available: {list(outs.keys())}",
                    path,
                )

            # Fallback: graph state (useful during tests or if scheduler filled it there)
            if src in fallback_outputs:
                outs = fallback_outputs[src] or {}
                if out_key in outs:
                    return outs[out_key]
                _err_path(
                    f"(fallback) Upstream node '{src}' has no output key '{out_key}'. "
                    f"Available: {list(outs.keys())}",
                    path,
                )

            _err_path(f"Upstream node '{src}' outputs not available yet", path)

        def _resolve_any(val: Any, path: str):
            # Handle dict markers
            if isinstance(val, dict):
                t = val.get("_type")
                if t == "arg":
                    return _resolve_arg(val, path)
                if t == "ref":
                    return _resolve_ref(val, path)
                if t == "context":
                    return self.memory.read(val["key"], val.get("default"))
                # regular dict: recurse keys
                return {k: _resolve_any(v, f"{path}.{k}") for k, v in val.items()}

            # Handle list/tuple
            if isinstance(val, list):
                return [_resolve_any(v, f"{path}[{i}]") for i, v in enumerate(val)]
            if isinstance(val, tuple):
                return tuple(_resolve_any(v, f"{path}[{i}]") for i, v in enumerate(val))

            # Pass-through literal
            return val

        # Make sure we don't mutate node.inputs
        # materialized = _resolve_any(copy.deepcopy(raw), path="inputs")
        materialized = _resolve_any(raw, path="inputs")

        # Strip framework-reserved injectables from *user* inputs.
        # We always inject these later from the execution context.
        if isinstance(materialized, dict):
            for k in list(materialized.keys()):
                if k in RESERVED_INJECTABLES:
                    materialized.pop(k, None)

            # If someone put arguments under "kwargs", keep them;
            # build_call_kwargs will flatten and then drop "kwargs".
            # (No change needed here beyond not touching it.)
        return materialized

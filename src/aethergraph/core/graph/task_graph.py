from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, is_dataclass
import datetime
import inspect
from typing import Any
import uuid

from .graph_refs import GRAPH_INPUTS_NODE_ID, Ref, normalize_binding, resolve_binding
from .graph_spec import GraphView, TaskGraphSpec
from .graph_state import GraphPatch, TaskGraphState
from .node_spec import TaskNodeSpec
from .node_state import NodeStatus, TaskNodeState
from .task_node import TaskNodeRuntime
from .utils import _logic_label, _short, _status_label
from .visualize import ascii_overview, to_dot, visualize


# small helper to turn a dataclass (spec) into plain dict safely
def _dataclass_to_plain(d):
    return asdict(d) if is_dataclass(d) else d


def _utc_ts() -> float:
    return datetime.datetime.now(tz=datetime.timezone.utc).timestamp()


@dataclass
class TaskGraph:
    spec: TaskGraphSpec
    state: TaskGraphState = field(
        default_factory=TaskGraphState
    )  # mutable state, including node states, patches
    observers: list[Any] = field(default_factory=list)

    # Expose graph_id as convenient alias; source of truth is spec.graph_id
    graph_id: str = field(init=False, repr=True)

    # Ephemeral runtime table (not serialized)
    _runtime_nodes: dict[str, TaskNodeRuntime] = field(default_factory=dict, init=False, repr=False)

    # Inverted indexeds for quick lookup (by alias, logic, label) ephemeral
    _idx_ready: bool = field(default=False, init=False, repr=False)
    _by_alias: dict[str, str] = field(
        default_factory=dict, init=False, repr=False
    )  # alias -> node_id
    _by_logic: dict[str, list[str]] = field(
        default_factory=dict, init=False, repr=False
    )  # logic -> [node_id, ...]
    _by_label: dict[str, list[str]] = field(
        default_factory=dict, init=False, repr=False
    )  # label -> {node_id, ...}
    _by_name: dict[str, list[str]] = field(
        default_factory=dict, init=False, repr=False
    )  # display name -> [node_id, ...]: TODO: decided if we need this

    @classmethod
    def new_run(self, spec: TaskGraphSpec, *, run_id: str | None = None, **kwargs) -> TaskGraph:
        """Create a new TaskGraph instance for a new run."""
        run_id = run_id or str(uuid.uuid4())
        # initialize empty node states
        nodes = {nid: TaskNodeState() for nid in spec.nodes}
        state = TaskGraphState(run_id=run_id, nodes=nodes)
        graph = self.from_spec(spec, state=state, **kwargs)
        return graph

    @classmethod
    def from_spec(cls, spec: TaskGraphSpec, *, state: TaskGraphState | None = None):
        """Create a TaskGraph instance from a TaskGraphSpec and optional state and memory."""
        graph = cls(spec=spec, state=state or TaskGraphState())
        # Set back-references in nodes
        for node in graph.spec.nodes.values():
            node._parent_graph = graph
        graph.ensure_inputs_node()
        graph.__post_init__()
        graph.ensure_inputs_node()

        # Set the inputs node state to DONE
        input_node = graph.node(GRAPH_INPUTS_NODE_ID)
        input_node.state.status = NodeStatus.DONE
        return graph

    # Publich read-only view
    def node(self, node_id: str) -> TaskNodeRuntime:
        return self._runtime_nodes[node_id]

    @property
    def nodes(self) -> list[TaskNodeRuntime]:
        return list(self._runtime_nodes.values())

    def _apply_patches(self) -> dict[str, TaskNodeSpec]:
        """Compute a patched node spec dict for the *view*.
        The original spec stays immutable
        """
        node_specs = dict(self.spec.nodes)  # shallow copy of mapping

        # The following is used when graph mutations are supported. It is just a sketch now.
        for p in self.state.patches:
            if p.op == "add_or_replace_node":
                ns = TaskNodeSpec(**p.payload)  # validate payload
                node_specs[ns.node_id] = ns
            elif p.op == "remove_node":
                node_specs.pop(p.payload["node_id"], None)
            elif p.op == "add_dependency":
                nid = p.payload["node_id"]
                dep = p.payload["dependency_id"]
                old = node_specs.get(nid)
                # create a new frozen spec with updated deps
                node_specs[nid] = TaskNodeSpec(
                    **{**old.__dict__, "dependencies": [*old.dependencies, dep]}
                )
            # TODO: add more patch type
            pass

        return node_specs

    def _reify_runtime_nodes(self):
        """Create TaskNodeRuntime instances for all nodes in the graph spec."""
        effective_specs = self._apply_patches()  # get patched specs
        table = {}
        for nid, nspec in effective_specs.items():
            nstate = self.state.nodes.get(nid)
            if nstate is None:
                nstate = TaskNodeState()  # in case a patch add a node
                self.state.nodes[nid] = nstate  # persist its state in TaskGraphState
            table[nid] = TaskNodeRuntime(spec=nspec, state=nstate, _parent_graph=self)
        self._runtime_nodes = table

    def __post_init__(self):
        # establish graph_id as alias to spec.graph_id
        self.graph_id = self.spec.graph_id

        # establish back-references in nodes
        if not getattr(self.state, "nodes", None):
            self.state.nodes = {
                nid: TaskNodeState() for nid in self.spec.nodes
            }  # GraphSpec.nodes is Dict[str, TaskNodeSpec]

        # establish back-references in nodes
        self._reify_runtime_nodes()

        # index for quick lookup
        self._reindex()

    def _reindex(self):
        self._by_alias.clear()
        self._by_logic.clear()
        self._by_label.clear()
        self._by_name.clear()
        for nid, node in self._runtime_nodes.items():
            metadata = getattr(node.spec, "metadata", {}) or {}
            alias = metadata.get("alias")
            labels = metadata.get("labels", [])
            display = metadata.get("display_name")
            logic_name = node.spec.tool_name or (
                node.spec.logic if isinstance(node.spec.logic, str) else _short(node.spec.logic)
            )

            if alias:
                self._by_alias[alias] = nid
            for label in labels:
                self._by_label.setdefault(label, set()).add(nid)
            if logic_name:
                self._by_logic.setdefault(logic_name, []).append(nid)
            if display:
                self._by_name.setdefault(display, []).append(nid)

        self._idx_ready = True

    def _ensure_index(self):
        if not self._idx_ready:
            self._reindex()

    # Call when mutate spec.nodes in PatchFlow
    def index_touch(self):
        self._idx_ready = False

    # Node access
    def node_ids(self) -> list[str]:
        """Get list of all node IDs in the graph."""
        return list(self._runtime_nodes.keys())

    # Node finder
    def get_by_id(self, node_id: str) -> str:
        """Get node ID by ID (identity function)."""
        if node_id not in self._runtime_nodes:
            raise ValueError(f"Node ID '{node_id}' not found in graph '{self.graph_id}'")
        return node_id

    def get_by_alias(self, alias: str) -> str | None:
        """Get node ID by alias."""
        self._ensure_index()
        node_id = self._by_alias.get(alias)
        if not node_id:
            raise KeyError(f"Alias '{alias}' not found in graph '{self.graph_id}'")
        return node_id

    def find_by_label(self, label: str) -> list[str]:
        """Find node IDs by label."""
        self._ensure_index()
        return sorted(self._by_label.get(label, set()))

    def find_by_logic(self, logic_prefix: str, *, first: bool = False) -> list[str] | str | None:
        """Find node IDs by logic name.
        If first=True, return only the first match or None if not found.

        Usage:
        graph.find_by_logic("my_tool")          # all nodes with logic name "my_tool"
        graph.find_by_logic("my_tool", first=True)  # first node with logic name "my_tool" or None
        graph.find_by_logic("my_tool_v")        # all nodes with logic name starting with "my_tool_v"
        graph.find_by_logic("my_tool_v", first=True) # first node with logic name starting with "my_tool_v" or None
        """
        self._ensure_index()
        if logic_prefix in self._by_logic:
            ids = list(self._by_logic[logic_prefix])
        else:
            ids = []
            for k, vs in self._by_logic.items():
                if k.startswith(logic_prefix):
                    ids.extend(vs)
        ids.sort()
        return (ids[0] if (first and ids) else ids) or ([] if not first else None)

    def find_by_display(self, name_prefix: str, *, first: bool = False) -> list[str] | str | None:
        """Find node IDs by display name.
        If first=True, return only the first match or None if not found.

        Usage:
        graph.find_by_display("My Node")          # all nodes with display name "My Node"
        graph.find_by_display("My Node", first=True)  # first node with display name "My Node" or None
        graph.find_by_display("My Node V")        # all nodes with display name starting with "My Node V"
        graph.find_by_display("My Node V", first=True) # first node with display name starting with "My Node V" or None
        """
        self._ensure_index()
        if name_prefix in self._by_name:
            ids = list(self._by_name[name_prefix])
        else:
            ids = []
            for k, vs in self._by_name.items():
                if k.startswith(name_prefix):
                    ids.extend(vs)
        ids.sort()
        return (ids[0] if (first and ids) else ids) or ([] if not first else None)

    # ---------- Unified selector ----------
    # Mini-DSL:
    #   "@alias"       -> by alias
    #   "#label"       -> by label (many)
    #   "id:<id>"      -> exact id
    #   "logic:<pref>" -> logic name prefix
    #   "name:<pref>"  -> display name prefix
    #   "/regex/"      -> regex on node_id

    def select(self, selector: str, *, first: bool = False) -> str | list[str] | None:
        selector = selector.strip()
        if selector.startswith("@"):
            return self.get_by_alias(selector[1:])
        elif selector.startswith("#"):
            ids = self.find_by_label(selector[1:])
            return ids[0] if (first and ids) else ids

        elif selector.startswith("id:"):
            return self.get_by_id(selector[3:])
        elif selector.startswith("logic:"):
            ids = self.find_by_logic(selector[6:], first=first)
            return ids
        elif selector.startswith("name:"):
            ids = self.find_by_display(selector[5:], first=first)
            return ids
        elif len(selector) >= 2 and selector[0] == "/" and selector[-1] == "/":
            import re

            pattern = re.compile(selector[1:-1])
            ids = [nid for nid in self.node_ids() if pattern.search(nid)]
            ids.sort()
            return ids[0] if (first and ids) else ids
        else:
            # fallback: prefix on node_id
            ids = [nid for nid in self.node_ids() if nid.startswith(selector)]
            ids.sort()
            return ids[0] if (first and ids) else ids

    def pick_one(self, selector: str) -> str | None:
        """Pick one node ID by selector, or None if not found."""
        res = self.select(selector, first=True)
        if not res:
            raise KeyError(f"No node found for selector '{selector}' in graph '{self.graph_id}'")
        return res

    def pick_all(self, selector: str) -> list[str]:
        """Pick all node IDs by selector, or empty list if none found."""
        res = self.select(selector, first=False)
        if isinstance(res, str):
            return [res]
        return res or []

    # --------- Read-only views ---------
    def view(self) -> GraphView:
        """Get a read-only view of the graph's spec and state."""
        return GraphView(
            graph_id=self.spec.graph_id,
            nodes=self.spec.nodes,
            node_status=self.state.node_status,  # state.node_status is a property in TaskGraphState derived from self.state.nodes
            metadata=self.spec.metadata,
        )

    # -------- Graph mutation APIs ---------
    def patch_add_or_replace_node(self, node_spec: dict[str, Any]):
        """Patch the graph by adding or replacing a node."""
        patch = GraphPatch(op="add_or_replace_node", payload=node_spec)
        self.state.patches.append(patch)
        self.state.rev += 1
        # awaitable = None
        for obs in self.observers:
            cb = getattr(obs, "on_patch_applied", None)
            if cb:
                cb(self, patch)  # r = cb(self, patch) if awaitable is needed
                # if hasattr(r, "__await__"):
                #     awaitable = r  # keep last; or gather all

        self._reify_runtime_nodes()

    def patch_remove_node(self, node_id: str):
        """Patch the graph by removing a node."""
        patch = GraphPatch(op="remove_node", payload={"node_id": node_id})
        self.state.patches.append(patch)
        self.state.rev += 1
        # awaitable = None
        for obs in self.observers:
            cb = getattr(obs, "on_patch_applied", None)
            if cb:
                cb(self, patch)  # r = cb(self, patch) if awaitable is needed
                # if hasattr(r, "__await__"):
                #     awaitable = r  # keep last; or gather all
        self._reify_runtime_nodes()

    def patch_add_dependency(self, node_id: str, dependency_id: str):
        """Patch the graph by adding a dependency to a node."""
        patch = GraphPatch(
            op="add_dependency", payload={"node_id": node_id, "dependency_id": dependency_id}
        )
        self.state.patches.append(patch)
        self.state.rev += 1
        # awaitable = None
        for obs in self.observers:
            cb = getattr(obs, "on_patch_applied", None)
            if cb:
                cb(self, patch)  # r = cb(self, patch) if awaitable is needed
                # if hasattr(r, "__await__"):
                #     awaitable = r  # keep last; or gather all
        self._reify_runtime_nodes()

    # --------- Introspection APIs ---------
    def list_nodes(self, exclude_internal=True) -> list[str]:
        """List all node IDs in the graph."""
        return (
            list(self.spec.nodes.keys())
            if not exclude_internal
            else [nid for nid in self.spec.nodes if not nid.startswith("_")]
        )

    # --------- Topology helpers ---------
    def dependents(self, node_id: str) -> list[str]:
        """Get list of node_ids that depend on the given node_id."""
        return [x.node_id for x in self.spec.nodes.values() if node_id in x.dependencies]

    def topological_order(self) -> list[str]:
        """Get nodes in topological order. Raises error if cycles are detected."""
        import networkx as nx

        G = nx.DiGraph()
        for n in self.spec.nodes.values():
            G.add_node(n.node_id)
            for dep in n.dependencies:
                G.add_edge(dep, n.node_id)
        try:
            order = list(nx.topological_sort(G))
            return order
        except nx.NetworkXUnfeasible:
            raise ValueError(
                "Graph has at least one cycle; topological sort not possible."
            ) from None

    def get_subgraph_nodes(self, start_node_id: str) -> list[str]:
        """Get all nodes reachable from the given start_node_id (including itself)."""
        seen, stack = set(), [start_node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(self.dependents(nid))
        return list(seen)

    def get_upstream_nodes(self, start_node_id: str) -> list[str]:
        """Get all upstream nodes that the given node_id depends on (including itself)."""
        seen, stack = set(), [start_node_id]
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            stack.extend(self.spec.nodes[nid].dependencies)
        return list(seen)

    # --------- State mutation APIs ---------
    async def set_status(self, node_id: str, status: NodeStatus):
        """Set the status of a node and notify observers."""
        raise NotImplementedError(
            "set_status() is not implemented yet. Use set_node_status() instead."
        )

    async def set_outputs(self, node_id: str, outputs: dict[str, Any]):
        """Set the outputs of a node."""
        raise NotImplementedError(
            "set_outputs() is not implemented yet. Use set_node_outputs() instead."
        )

    # async def set_node_status(self, node_id: str, status: NodeStatus) -> None:
    #     state = self.state.nodes.get(node_id)
    #     if state.status is status:
    #         return
    #     state.status = status
    #     self.state.rev += 1
    #     await self._notify_status_change(node_id)

    async def set_node_status(self, node_id: str, status: NodeStatus) -> None:
        state = self.state.nodes.get(node_id)
        if state is None:
            # or however you want to handle missing nodes
            raise KeyError(f"Unknown node_id: {node_id}")

        # no-op if unchanged
        if state.status is status or state.status == status:
            return

        # --- timestamps ---

        # 1) First time we go to RUNNING → set started_at (but don't overwrite on resume)
        if status == NodeStatus.RUNNING and getattr(state, "started_at", None) is None:
            state.started_at = _utc_ts()

        # 2) Terminal states → set finished_at if not already set
        TERMINAL_STATES = {
            getattr(NodeStatus, "DONE", "DONE"),
            getattr(NodeStatus, "FAILED", "FAILED"),
            getattr(NodeStatus, "SKIPPED", "SKIPPED"),
            getattr(NodeStatus, "CANCELLED", None) or getattr(NodeStatus, "CANCELED", None),
        }
        # filter out any Nones in case some names don't exist
        TERMINAL_STATES = {s for s in TERMINAL_STATES if s is not None}

        if status in TERMINAL_STATES and getattr(state, "finished_at", None) is None:
            state.finished_at = _utc_ts()

        # --- actual status change + rev bump ---
        state.status = status
        self.state.rev += 1
        await self._notify_status_change(node_id)

    async def set_node_outputs(self, node_id: str, outputs: dict[str, Any]) -> None:
        state = self.state.nodes.get(node_id)
        state.outputs = outputs
        self.state.rev += 1
        await self._notify_output_change(node_id)

    async def _notify_status_change(self, node_id: str):
        runtime_node = self._runtime_nodes.get(node_id)  # runtime view points at same state object
        for obs in self.observers:
            cb = getattr(obs, "on_node_status_change", None)
            if cb:
                out = cb(runtime_node)
                if hasattr(out, "__await__"):
                    await out

    def _notify_inputs_bound(self):
        for obs in self.observers:
            cb = getattr(obs, "on_inputs_bound", None)
            if cb:
                out = cb(self)
                if hasattr(out, "__await__"):
                    # fire-and-forget is okay; await here to keep ordering
                    # (code already awaits in other notify paths)
                    pass

    async def _notify_output_change(self, node_id: str):
        runtime_node = self._runtime_nodes.get(node_id)  # runtime view points at same state object
        for obs in self.observers:
            cb = getattr(obs, "on_node_output_change", None)
            if cb:
                out = cb(runtime_node)
                if hasattr(out, "__await__"):
                    await out

    # --------- Rest paths ---------
    async def reset_node(self, node_id: str, *, preserve_outputs: bool = False):
        """Reset a node to PENDING state. Optionally preserve outputs."""
        if node_id not in self.spec.nodes:
            raise ValueError(f"Node with id {node_id} does not exist in the graph.")

        if node_id == GRAPH_INPUTS_NODE_ID:
            raise ValueError("Cannot reset the special graph inputs node.")

        node = self.state.nodes[node_id]
        await node.reset_node(preserve_outputs=preserve_outputs)

    async def reset(
        self,
        node_ids: list[str] | None = None,
        *,
        recursive=True,
        direction="forward",
        preserve_outputs: bool = False,
    ):
        """
        Reset the graph or a subgraph to PENDING state.
        If node_id is None, reset the entire graph.
        If recursive is True, reset all dependent nodes (forward) or dependencies (backward).
        """
        if not node_ids:
            # Reset the entire graph
            for nid in list(self.spec.nodes.keys()):
                if nid == GRAPH_INPUTS_NODE_ID:
                    continue
                await self.reset_node(nid, preserve_outputs=preserve_outputs)

        # partial reset
        target_ids = []
        for nid in node_ids:
            if recursive:
                if direction == "forward":
                    target_ids.extend(self.get_subgraph_nodes(nid))
                elif direction == "backward":
                    target_ids.extend(self.get_upstream_nodes(nid))
                else:
                    raise ValueError("direction must be 'forward' or 'backward'")
            else:
                target_ids.append(nid)

        for nid in set(target_ids):
            await self.reset_node(nid, preserve_outputs=preserve_outputs)

        return {
            "status": "partial_reset",
            "graph_id": self.spec.graph_id,
            "nodes_reset": list(set(target_ids)),
        }

    # --------- Observers and hooks ---------
    def add_observer(self, observer: Any):
        self.observers.append(observer)

    # --------- Difference APIs ---------
    def diff(self, other: TaskGraph) -> dict[str, Any]:
        """
        Compute the difference between this graph and another graph.
        Returns a dict with added, removed, and modified nodes.
        """
        if self.spec.graph_id != other.spec.graph_id:
            raise ValueError("Can only diff graphs with the same graph_id.")

        diff_result = {"added": [], "removed": [], "modified": []}

        # Check for added and modified nodes
        for nid, node in other.spec.nodes.items():
            if nid not in self.spec.nodes:
                diff_result["added"].append(nid)
            else:
                # Check for modifications (dependencies or metadata changes)
                old_node = self.spec.nodes[nid]
                if (set(old_node.dependencies) != set(node.dependencies)) or (
                    old_node.metadata != node.metadata
                ):
                    diff_result["modified"].append(nid)

        # Check for removed nodes
        for nid in self.spec.nodes:
            if nid not in other.spec.nodes:
                diff_result["removed"].append(nid)

        return diff_result

    # --------- IO definition APIs ---------
    def declare_inputs(
        self, *, required: Iterable[str] | None = None, optional: dict[str, Any] | None = None
    ) -> None:
        """Declare graph-level inputs."""
        # if required: self.spec._io_inputs_required.update(required)
        # if optional: self.spec._io_inputs_optional.update(optional or {})

        from .graph_io import ParamSpec

        required_spec = {
            k: ParamSpec() for k in (required or [])
        }  # currently we don't support detailed param spec. Only names are used.
        optional_spec = {k: ParamSpec(default=v) for k, v in (optional or {}).items()}
        if required:
            self.spec.io.required.update(required_spec)
        if optional:
            self.spec.io.optional.update(optional_spec)

    def expose(self, name: str, value: Ref | Any) -> None:
        """Expose a graph-level output.
        In graph IO, outputs can be references to node outputs or constant values.
        """
        if name not in self.spec.io.expose:
            self.spec.io.expose.append(name)
        self.spec.io.set_expose(name, normalize_binding(value))

    def require_outputs(self, *names: str) -> None:
        """Require certain graph-level outputs to be present."""
        missing = [n for n in names if n not in self.spec._io_outputs]
        if missing:
            raise ValueError(f"Missing required outputs: {', '.join(missing)}")

    def io_signature(self, include_values: bool = False) -> dict[str, Any]:
        """Get the graph's IO signature as a dict.
        The signature includes:
            - inputs: {required: [...], optional: {...}}
            - outputs: {keys: [...], bindings: {...}}
        Note: Disable include_values when initializing a graph to avoid resolving unbound refs.
        """
        if hasattr(self.spec.io, "get_expose_names"):
            names: list[str] = self.spec.io.get_expose_names()
        else:
            names = list(getattr(self.spec.io, "expose", []) or [])

        if hasattr(self.spec.io, "get_expose_bindings"):
            bindings: dict[str, Any] = self.spec.io.get_expose_bindings()
        else:
            bindings = dict(getattr(self.spec, "meta", {}).get("expose_bindings", {}))

        # Build the signature dict with concrete iterables / dicts
        out = {
            "inputs": {
                "required": sorted(self.spec.inputs_required),
                "optional": dict(self.spec.inputs_optional),
            },
            "outputs": {
                "keys": list(names),
                "bindings": {n: bindings.get(n) for n in names},
            },
        }

        if include_values:
            out["outputs"]["values"] = {
                n: resolve_binding(bindings.get(n), self.state.node_outputs) for n in names
            }
        return out

    def ensure_inputs_node(self):
        if GRAPH_INPUTS_NODE_ID not in self.spec.nodes:
            node_spec = TaskNodeSpec(
                node_id=GRAPH_INPUTS_NODE_ID,
                type="inputs",
                logic=None,
                inputs={},
                dependencies=[],
                metadata={"synthetic": True},
                expected_input_keys=[],
                expected_output_keys=[],
            )
            self.spec.nodes[GRAPH_INPUTS_NODE_ID] = node_spec

            node_state = self.state.nodes.setdefault(GRAPH_INPUTS_NODE_ID, TaskNodeState())
            node_state.status = NodeStatus.DONE

    def _validate_and_bind_inputs(self, provided: dict[str, Any]) -> dict[str, Any]:
        """Validate and bind provided inputs against the graph's IO signature."""
        req = self.spec.inputs_required
        # opt = set(self.spec.inputs_optional.keys())
        missing = [k for k in req if k not in provided]
        if missing:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")

        merged = dict(self.spec.inputs_optional)  # start with optional defaults
        merged.update(provided)  # override with provided
        self.state._bound_inputs = merged

        # bump rev; persist an event
        self.state.rev += 1
        # notify
        out = self._notify_inputs_bound()
        if hasattr(out, "__await__"):
            # optional: await if later want strict ordering
            pass
        return merged

    def _resolve_ref(self, r: Any, node_outputs: dict[str, dict[str, Any]]) -> Any:
        """Resolve a Ref or return the value as-is."""
        if not (isinstance(r, dict) and r.get("_type") == "ref"):
            return r

        src, key = r.get("from"), r.get("key")
        if src == GRAPH_INPUTS_NODE_ID:
            if self.state._bound_inputs is None:
                raise RuntimeError("Graph inputs not bound. Call graph(...) or bind explicitly.")
            return self.state._bound_inputs.get(key)
        return node_outputs.get(src, {}).get(key)

    # --------- Execution APIs ---------
    # Here we have temporary APIs, later we will use tools and scheduler to manage execution
    def _load_logic(self, logic: Any):
        # If logic is a callable, return it as-is
        if callable(logic):
            return logic
        # If logic is a string, check if it starts with "registry:"
        if isinstance(logic, str) and logic.startswith("registry:"):
            # If it does, look it up in the registry
            return self._lookup_registry(logic)
        # If we reach here, logic is not valid
        raise ValueError(f"Invalid logic: {logic}")

    async def _run_tool(self, logic: Any, **kwargs):
        fn = self._load_logic(logic)
        res = fn(**kwargs)
        if inspect.isawaitable(res):
            res = await res
        return res if isinstance(res, dict) else {"result": res}

    # -------- Print and Debug ---------
    def pretty(self, *, max_nodes: int = 20, max_width: int = 100) -> str:
        """
        Human-friendly summary of this TaskGraph.
        """
        lines: list[str] = []

        # Header
        lines.append(
            f"TaskGraph[{self.spec.graph_id}]  "
            f"nodes={len(self.spec.nodes)}  "
            f"observers={len(self.observers)}"
        )

        # IO signature
        lines.append("IO Signature:")
        for s in self.spec.io_summary_lines():
            lines.append(f"  {s}")

        # State summary
        lines.append(f"State: {self.state.summary_line()}")

        # Nodes table (compact)
        lines.append("Nodes:")
        header = f"{'id':<22}  {'type':<10}  {'status':<12}  {'#deps':<5}  logic"
        lines.append("  " + header)
        lines.append("  " + "-" * (len(header) + 4))

        def _safe_get(node, attr, default):
            return getattr(node, attr, default)

        n_items = list(self.spec.nodes.items())
        for idx, (nid, node) in enumerate(n_items):
            if idx >= max_nodes:
                lines.append(f"  … ({len(n_items) - max_nodes} more)")
                break

            ntype = _safe_get(node, "node_type", "?")
            status = _status_label(self.state.nodes.get(nid, TaskNodeState()).status)
            deps = _safe_get(node, "dependencies", None) or []
            logic = _logic_label(_safe_get(node, "logic", None))

            # Width control: keep table tidy
            row = f"  {_short(nid, 22):<22}  {_short(ntype, 10):<10}  {_short(status, 12):<12}  {len(deps):<5}  {_short(logic, max_width)}"
            lines.append(row)

        return "\n".join(lines)

    # Optional: make print(graph) show a compact version
    def __str__(self) -> str:
        return self.pretty(max_nodes=12, max_width=96)

    # -------- Persistence conveniences (opt-in) --------
    def spec_json(self) -> dict[str, Any]:
        """
        JSON-safe representation of the graph spec.
        Keeps TaskGraph storage-agnostic; callers can write to file/db/etc.
        """
        return _dataclass_to_plain(self.spec)


# --------- Visualization ---------
TaskGraph.to_dot = to_dot
TaskGraph.visualize = visualize
TaskGraph.ascii_overview = ascii_overview

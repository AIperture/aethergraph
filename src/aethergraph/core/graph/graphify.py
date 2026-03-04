from __future__ import annotations

import inspect
from typing import Any, get_origin, get_type_hints

from aethergraph.core.graph.action_spec import _map_py_type_to_json_type
from aethergraph.services.registry.agent_app_meta import (
    AgentConfig,
    AppConfig,
    build_agent_meta,
    build_app_meta,
)

from ..runtime.runtime_registry import current_registry
from .task_graph import TaskGraph


def _normalize_type_hint(ann: Any) -> str:
    """Convert a Python annotation into a simple string for IO types."""
    if ann is inspect._empty:
        return "any"

    origin = get_origin(ann)
    # args = get_args(ann)

    # Builtins
    if ann is str:
        return "string"
    if ann is int:
        return "int"
    if ann is float:
        return "float"
    if ann is bool:
        return "bool"
    if ann is dict or origin is dict:
        # e.g. dict[str, float]
        return "object"
    if ann in (list, tuple) or origin in (list, tuple):
        # e.g. list[int] / list[dict[str, float]]
        return "array"
    if ann is Any:
        return "any"

    # Fallback: stringified type name
    return getattr(ann, "__name__", str(ann))


def graphify(
    name="default_graph",
    inputs=(),
    outputs=None,
    version="0.1.0",
    *,
    entrypoint: bool = False,
    flow_id: str | None = None,
    tags: list[str] | None = None,
    as_agent: AgentConfig | None = None,
    as_app: AppConfig | None = None,
    description: str | None = None,
):
    """
    Decorator to define a `TaskGraph` and optionally register it as an agent or app.

    This decorator wraps a Python function as a `TaskGraph`, enabling it to be executed
    as a node-based graph with runtime context, retry policy, and concurrency controls.
    It also supports rich metadata registration for agent and app discovery.

    Examples:
        Basic usage:
        ```python
        @graphify(
            name="add_numbers",
            inputs=["a", "b"],
            outputs=["sum"],
        )
        async def add_numbers(a: int, b: int):
            return {"sum": a + b}
        ```

        Registering as an agent with metadata:
        ```python
        @graphify(
            name="chat_agent",
            inputs=["message", "files", "context_refs", "session_id", "user_meta"],
            outputs=["response"],
            as_agent={
                "id": "chatbot",
                "title": "Chat Agent",
                "description": "Conversational AI agent.",
                "mode": "chat_v1",
                "icon": "chat",
                "tags": ["chat", "nlp"],
            },
        )
        async def chat_agent(...):
            ...
        ```

        Registering as an app:
        ```python
        @graphify(
            name="summarizer",
            inputs=[],
            outputs=["summary"],
            as_app={
                "id": "summarizer-app",
                "name": "Text Summarizer",
                "description": "Summarizes input text.",
                "category": "Productivity",
                "tags": ["nlp", "summary"],
            },
        )
        async def summarizer():
            ...
        ```

        Typed app inputs with defaults + UI schema override:
        ```python
        @graphify(
            name="generic_typed_workflow_v1",
            inputs={
                "limit": 100,                 # optional with default
                "temperature": 0.2,           # optional with default
                "options": {"mode": "fast"},  # optional object default
                "required_like": None,        # workaround: treat as required in function validation
            },
            outputs=["result", "metrics", "artifacts"],
            as_app={
                "id": "generic-typed-app",
                "name": "Generic Typed App",
                "input_schema": [
                    {"name": "limit", "label": "Limit", "widget": "number", "default": 250},
                    {"name": "temperature", "label": "Temperature", "widget": "number"},
                    {"name": "enabled", "label": "Enabled", "widget": "switch", "default": True},
                    {"name": "options", "label": "Options", "widget": "json"},
                    {"name": "required_like", "label": "Required Value", "widget": "text"},
                ],
            },
        )
        def generic_typed_workflow(
            limit: int,
            temperature: float,
            options: dict[str, str],
            required_like: str | None,
            enabled: bool = False,
        ):
            if required_like is None:
                raise ValueError("required_like must be provided")
            ...
        ```

    Args:
        name: Unique name for the graph function.
        inputs: Graph input declaration. Supports either:
            - `list[str]`: required input names.
            - `dict[str, Any]`: optional input names mapped to default values.
            If `as_agent` is provided with `mode="chat_v1"`, this must match
            `["message", "files", "context_refs", "session_id", "user_meta"]`.
            Type inference uses function annotations for declared input names
            (e.g. `int/float -> number`, `str -> string`, `bool -> boolean`,
            `dict -> object`, `list -> array`).
        outputs: List of output keys returned by the function.
        version: Version string for the graph function (default: "0.1.0").
        entrypoint: If True, marks this graph as the main entrypoint for a flow.  [Currently unused]
        flow_id: Optional flow identifier for grouping related graphs.
        tags: List of string tags for discovery and categorization.
        as_agent: Optional dictionary defining agent metadata. Used when running through Aethergraph UI. See additional information below.
        as_app: Optional dictionary defining app metadata. Used when running through Aethergraph UI.
            Supports optional `input_schema` UI overrides. Each entry is matched
            by `name` and can provide UI hints such as `label`, `placeholder`,
            `widget`, `description`, and `default`.
        description: Optional human-readable description of the graph function.

    Returns:
        TaskGraph: A decorator that transforms a function into a TaskGraph with the specified configuration.

    Notes:
        - as_agent and as_app are not needed to define a graph; they are only for registration purposes for use in Aethergraph UI.
        - When registering as an agent, the `as_agent` dictionary should include at least an "id" key.
        - When registering as an app, the `as_app` dictionary should include at least an "id" key.
        - The decorated function is a sync function (generate the TaskGraph), despite the underlying `@tool` can be async.
        - For `graphify`, required vs optional is controlled by `inputs`:
          `list[str]` means required; `dict[str, default]` means optional with defaults.
        - `graphify` does not currently support mixing required and optional in a
          single `inputs` declaration. To have required inputs, use `list[str]`.
          If you must use `dict` only, a common workaround is a sentinel default
          (for example `None`) and explicit validation inside the function body.
        - Runtime optional defaults declared in `inputs` are applied by graph binding.
          UI defaults are shown only when surfaced via API input schema and can be
          overridden by `as_app.input_schema` defaults.
        - Field types are inferred from function annotations for declared inputs and
          are exposed as JSON-like types in registry metadata.
    """

    def _wrap(fn):
        fn_sig = inspect.signature(fn)
        fn_params = list(fn_sig.parameters.keys())

        # Normalize declared inputs into a list of names
        required_inputs = list(inputs.keys()) if isinstance(inputs, dict) else list(inputs)

        # Optional: validate the signature matches declared inputs
        # (or keep permissive: inject only the overlap)
        overlap = [p for p in fn_params if p in required_inputs]

        def _build() -> TaskGraph:
            from .graph_builder import graph
            from .graph_refs import arg

            agent_id = as_agent.get("id") if as_agent else None
            app_id = as_app.get("id") if as_app else None

            with graph(name=name, agent_id=agent_id, app_id=app_id) as g:
                # declarations unchanged...
                if isinstance(inputs, dict):
                    g.declare_inputs(required=[], optional=inputs)
                else:
                    g.declare_inputs(required=required_inputs, optional={})

                # --- Inject args: map fn params -> arg("<name>")
                injected_kwargs = {p: arg(p) for p in overlap}

                # Run user body
                ret = fn(**injected_kwargs)

                # expose logic (fixed typo + single-output collapse)
                def _is_ref(x):
                    return (
                        isinstance(x, dict)
                        and x.get("_type") == "ref"
                        and "from" in x
                        and "key" in x
                    )

                def _expose_from_handle(prefix, handle):
                    oks = list(getattr(handle, "output_keys", []))
                    if prefix and len(oks) == 1:
                        g.expose(prefix, getattr(handle, oks[0]))
                    else:
                        for k in oks:
                            g.expose(f"{prefix}.{k}" if prefix else k, getattr(handle, k))

                if isinstance(ret, dict):
                    for k, v in ret.items():
                        if _is_ref(v):
                            g.expose(k, v)
                        elif hasattr(v, "node_id"):
                            _expose_from_handle(k, v)
                        else:
                            g.expose(k, v)
                elif hasattr(ret, "node_id"):
                    _expose_from_handle("", ret)
                else:
                    if outputs:
                        if len(outputs) != 1:
                            raise ValueError(
                                "Returning a single literal but multiple outputs are declared."
                            )
                        g.expose(outputs[0], ret)
                    else:
                        raise ValueError(
                            "Returning a single literal but no output name is declared."
                        )
            return g

        _build.__ag_builder__ = True
        _build.__name__ = fn.__name__
        _build.build = _build  # alias
        _build.graph_name = name
        _build.version = version

        def _spec():
            g = _build()
            return g.spec

        _build.spec = _spec

        def _io():
            g = _build()
            return g.io_signature()

        _build.io = _io

        # ---- Register graph + optional agent ----

        registry = current_registry()
        if registry is None:
            return _build

        base_tags = tags or []

        # Effective description
        doc_desc = inspect.getdoc(fn) or None
        eff_description = description or doc_desc or name

        # Infer IO types from annotations if possible
        try:
            resolved_hints = get_type_hints(fn)
        except Exception:
            # Fallback: use raw __annotations__ if get_type_hints blows up
            resolved_hints = getattr(fn, "__annotations__", {}) or {}

        # Infer IO types from annotations in a JSON-ish schema
        input_type_map: dict[str, str] = {}
        for pname in required_inputs:
            param = fn_sig.parameters.get(pname)
            if param is None:
                continue

            # Prefer resolved type hint; fall back to the raw annotation
            ann = resolved_hints.get(pname, param.annotation)
            if ann is inspect._empty:
                continue

            j = _map_py_type_to_json_type(ann)
            if j is not None:
                input_type_map[pname] = j

        # for outputs, we only have the return annotation as a whole
        output_names = list(outputs or [])
        output_type_map: dict[str, str] = {n: "any" for n in output_names}

        graph_meta: dict[str, Any] = {
            "kind": "graph",
            "entrypoint": entrypoint,
            "flow_id": flow_id,
            "tags": base_tags,
            "description": eff_description,
            "inputs": inputs,
            "outputs": outputs,
            "io_types": {
                "inputs": input_type_map,
                "outputs": output_type_map,
            },
        }

        registry.register(
            nspace="graph",
            name=name,
            version=version,
            obj=_build,
            meta=graph_meta,
        )

        # Agent meta (if any)
        agent_meta = build_agent_meta(
            graph_name=name,
            version=version,
            graph_meta=graph_meta,
            agent_cfg=as_agent,
        )
        if agent_meta is not None:
            registry.register(
                nspace="agent",
                name=agent_meta["id"],
                version=version,
                obj=_build,
                meta=agent_meta,
            )

        # App meta (if any)
        app_meta = build_app_meta(
            graph_name=name,
            version=version,
            graph_meta=graph_meta,
            app_cfg=as_app,
        )
        if app_meta is not None:
            registry.register(
                nspace="app",
                name=app_meta["id"],
                version=version,
                obj=_build,
                meta=app_meta,
            )

        return _build

    return _wrap

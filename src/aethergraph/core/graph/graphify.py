from __future__ import annotations

import inspect
from typing import Any

from aethergraph.services.registry.agent_app_meta import build_agent_meta, build_app_meta

from ..runtime.runtime_registry import current_registry
from .task_graph import TaskGraph


def graphify(
    name="default_graph",
    inputs=(),
    outputs=None,
    version="0.1.0",
    *,
    entrypoint: bool = False,
    flow_id: str | None = None,
    tags: list[str] | None = None,
    as_agent: dict[str, Any] | None = None,
    as_app: dict[str, Any] | None = None,
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

    Args:
        name: Unique name for the graph function.
        inputs: List of input parameter names. If `as_agent` is provided with `mode="chat_v1"`,
            this must match `["message", "files", "context_refs", "session_id", "user_meta"]`.
        outputs: List of output keys returned by the function.
        version: Version string for the graph function (default: "0.1.0").
        entrypoint: If True, marks this graph as the main entrypoint for a flow.  [Currently unused]
        flow_id: Optional flow identifier for grouping related graphs.
        tags: List of string tags for discovery and categorization.
        as_agent: Optional dictionary defining agent metadata. Used when running through Aethergraph UI. See additional information below.
        as_app: Optional dictionary defining app metadata. Used when running through Aethergraph UI. See additional information below.
        description: Optional human-readable description of the graph function.

    Returns:
        TaskGraph: A decorator that transforms a function into a TaskGraph with the specified configuration.

    Notes:
        - as_agent and as_app are not needed to define a graph; they are only for registration purposes for use in Aethergraph UI.
        - When registering as an agent, the `as_agent` dictionary should include at least an "id" key.
        - When registering as an app, the `as_app` dictionary should include at least an "id" key.
        - The decorated function is a sync function (generate the TaskGraph), despite the underlying `@tool` can be async.
        - Fields `inputs` and `outputs` are can be inferred from the function signature if not explicitly provided, but it's recommended to declare them for clarity.
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

        doc_desc = inspect.getdoc(fn) or None
        eff_description = description or doc_desc or name

        graph_meta: dict[str, Any] = {
            "kind": "graph",
            "entrypoint": entrypoint,
            "flow_id": flow_id,
            "tags": base_tags,
            "description": eff_description,
            "inputs": inputs,
            "outputs": outputs,
        }

        registry.register(
            nspace="graph",
            name=name,
            version=version,
            obj=_build(),
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
                obj=_build(),
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
                obj=_build(),
                meta=app_meta,
            )

        return _build

    return _wrap

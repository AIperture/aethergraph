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
):
    """
    Decorator that builds a TaskGraph from a function body using the builder context.
    The function author writes sequential code with tool calls returning NodeHandles.

    Usage:
    @graphify(name="my_graph", inputs=["input1", "input2"], outputs=["output"])
    def my_graph(input1, input2):
        # function body using graph builder API
        pass
        return {"output": some_node_handle}

    The decorated function returns a builder function that constructs the TaskGraph.

    To build the graph, call the returned function:
    graph_instance = my_graph.build()
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
        graph_meta: dict[str, Any] = {
            "kind": "graph",
            "entrypoint": entrypoint,
            "flow_id": flow_id or name,
            "tags": base_tags,
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

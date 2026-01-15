from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, get_type_hints

from aethergraph.core.graph.action_spec import IOSlot, _map_py_type_to_json_type
from aethergraph.core.runtime.run_registration import RunRegistrationGuard
from aethergraph.services.registry.agent_app_meta import (
    build_agent_meta,
    build_app_meta,
)

from ..execution.retry_policy import RetryPolicy
from ..runtime.runtime_env import RuntimeEnv
from ..runtime.runtime_registry import current_registry  # ContextVar accessor
from .graph_builder import graph  # context manager
from .graph_refs import GRAPH_INPUTS_NODE_ID
from .interpreter import Interpreter
from .node_spec import TaskNodeSpec


class GraphFunction:
    def __init__(
        self,
        name: str,
        fn: Callable,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        version: str = "0.1.0",
        agent_id: str | None = None,
        app_id: str | None = None,
    ):
        self.graph_id = name
        self.name = name
        self.fn = fn
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.version = version
        self.registry_key: str | None = None
        self.last_graph = None
        self.last_context = None
        self.last_memory_snapshot = None
        self.agent_id = agent_id
        self.app_id = app_id

    async def run(
        self,
        *,
        env: RuntimeEnv | None = None,
        retry: RetryPolicy | None = None,
        max_concurrency: int | None = None,
        **inputs,
    ):
        """
        Build a fresh TaskGraph and execute this function via the Interpreter.
        If 'context' is declared in the fn signature, inject a NodeContext.
        """
        # Build env if not provided (use runner’s builder for consistency)
        if env is None:
            from ..runtime.graph_runner import _build_env  # internal helper

            env, retry, max_concurrency = await _build_env(self, inputs)
        if retry is None:
            retry = RetryPolicy()

        node_spec = TaskNodeSpec(
            node_id=GRAPH_INPUTS_NODE_ID, type="inputs", metadata={"synthetic": True}
        )
        runtime_ctx = env.make_ctx(
            node=node_spec, resume_payload=getattr(env, "resume_payload", None)
        )
        node_ctx = runtime_ctx.create_node_context(node=node_spec)

        with graph(name=self.graph_id, agent_id=self.agent_id, app_id=self.app_id) as G:
            interp = Interpreter(G, env, retry=retry, max_concurrency=max_concurrency)
            run_id = env.run_id

            # Register the scheduler for this run_id
            with RunRegistrationGuard(
                run_id=run_id, scheduler=interp.scheduler, container=env.container
            ):
                sig = inspect.signature(self.fn)
                call_kwargs = dict(inputs)
                if "context" in sig.parameters:
                    call_kwargs["context"] = node_ctx

                with interp.enter():
                    res = self.fn(**call_kwargs)
                    if inspect.isawaitable(res):
                        res = await res

            self.last_graph = G

        res = _normalize_and_expose(G, res, self.outputs)
        return res

    # --- Syntactic sugar ---
    async def __call__(self, **inputs):
        """Async call to run the graph function.
        Usage:
           result = await my_graph_fn(input1=value1, input2=value2)
        """
        from ..runtime.graph_runner import run_async

        return await run_async(self, inputs)

    def sync(self, **inputs):
        """Synchronous wrapper around async run(). Useful for quick tests or scripts.
        Usage:
           result = my_graph_fn.sync(input1=value1, input2=value2)
        """
        from ..runtime.graph_runner import run

        return run(self, inputs)

    def io_signature(self) -> dict[str, list[IOSlot]]:
        """
        Infer typed IO based on decorator inputs/outputs and Python annotations.

        Rule of thumbs:
        - Inputs: use decorator list; if missing, use all params except 'context'.
        - Outputs: use decorator list; if missing, use return annotation if any.

        """
        sig = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)

        # --- Inputs ---
        if self.inputs is not None:
            input_names = list(self.inputs)
        else:
            # fallback: all params except 'context'
            input_names = [p for p in sig.parameters if p != "context"]

        input_slots: list[IOSlot] = []
        for name in input_names:
            param = sig.parameters.get(name)
            if param is None:
                # decorator said it's a input but not in signature
                # treat as unknown, required
                input_slots.append(IOSlot(name=name, type=None, required=True))
                continue

            anno = hints.get(name)
            j_type = _map_py_type_to_json_type(anno) if anno is not None else None
            required = param.default is inspect._empty
            default = None if required else param.default

            input_slots.append(
                IOSlot(
                    name=name,
                    type=j_type,
                    required=required,
                    default=default,
                )
            )

        # --- Outputs ---
        output_slots: list[IOSlot] = []
        out_names = list(self.outputs or [])

        ret_anno = hints.get("return")
        if out_names:
            if len(out_names) == 1 and ret_anno is not None:
                j_type = _map_py_type_to_json_type(ret_anno)
                output_slots.append(IOSlot(name=out_names[0], type=j_type, required=True))
            else:
                # Multi-output or no return type: names only
                for name in out_names:
                    output_slots.append(IOSlot(name=name, type=None, required=True))
        elif ret_anno is not None:
            # No explicit output names but we *do* know the type: create a single output
            j_type = _map_py_type_to_json_type(ret_anno)
            output_slots.append(IOSlot(name="result", type=j_type, required=True))

        return {"inputs": input_slots, "outputs": output_slots}


def _is_ref(x: object) -> bool:
    return isinstance(x, dict) and x.get("_type") == "ref" and "from" in x and "key" in x


def _is_nodehandle(x: object) -> bool:
    return hasattr(x, "node_id") and hasattr(x, "output_keys")


def _expose_from_handle(G, prefix: str, handle) -> dict:
    oks = list(getattr(handle, "output_keys", []))
    if not oks:
        raise ValueError(f"NodeHandle '{getattr(handle, 'node_id', '?')}' has no output_keys")
    out = {}
    if prefix and len(oks) == 1:
        # collapse single output to the provided key
        k = oks[0]
        ref = getattr(handle, k)
        G.expose(prefix, ref)
        out[prefix] = ref
    else:
        # multi-output (or top-level handle)
        for k in oks:
            key = f"{prefix}.{k}" if prefix else k
            ref = getattr(handle, k)
            G.expose(key, ref)
            out[key] = ref
    return out


def _normalize_and_expose(G, ret, declared_outputs: list[str] | None) -> dict:
    """
    Normalize user return into {key: Ref or literal}.
    - Dict of NodeHandles/Refs/literals supported
    - Single NodeHandle supported
    - Single literal supported (needs 1 declared output)
    Also exposes Refs on G as boundary outputs.

    Examples:
    - return {"result": ref(...), "summary": node_handle(...), "count": 42}
    - return node_handle(...)
    """
    result = {}

    if isinstance(ret, dict):
        for k, v in ret.items():
            if _is_ref(v):
                G.expose(k, v)
                result[k] = v
            elif _is_nodehandle(v):
                result.update(_expose_from_handle(G, k, v))
            else:
                # literal stays literal; no expose
                result[k] = v

    elif _is_nodehandle(ret):
        result.update(_expose_from_handle(G, "", ret))

    else:
        # single literal/ref case
        if declared_outputs and len(declared_outputs) == 1:
            key = declared_outputs[0]
            if _is_ref(ret):
                G.expose(key, ret)
            result[key] = ret
        else:
            raise ValueError(
                "Returning a single literal but outputs are not declared or >1. "
                "Declare exactly one output or return a dict."
            )

    # If outputs were declared, restrict to those keys (keep order)
    if declared_outputs:
        result = {k: result[k] for k in declared_outputs if k in result}

        # Validate presence
        missing = [k for k in declared_outputs if k not in result]
        if missing:
            raise ValueError(f"Missing declared outputs: {missing}")

    return result


def graph_fn(
    name: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    version: str = "0.1.0",
    *,
    entrypoint: bool = False,
    flow_id: str | None = None,
    tags: list[str] | None = None,
    as_agent: dict[str, Any] | None = None,
    as_app: dict[str, Any] | None = None,
    description: str | None = None,
) -> Callable[[Callable], GraphFunction]:
    """
    Decorator to define a graph function and optionally register it as an agent or app.

    This decorator wraps a Python function as a `GraphFunction`, enabling it to be executed
    as a node-based graph with runtime context, retry policy, and concurrency controls.
    It also supports rich metadata registration for agent and app discovery.

    Examples:
        Basic usage:
        ```python
        @graph_fn(
            name="add_numbers",
            inputs=["a", "b"],
            outputs=["sum"],
        )
        async def add_numbers(a: int, b: int):
            return {"sum": a + b}
        ```

        Registering as an agent with metadata:
        ```python
        @graph_fn(
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
        @graph_fn(
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
        Callable: A decorator that wraps the function as a `GraphFunction` and registers it
        in the runtime registry, with agent/app metadata if provided.

    Notes:
        - as_agent and as_app are not needed to define a graph; they are only for registration purposes for use in Aethergraph UI.
        - When registering as an agent, the `as_agent` dictionary should include at least an "id" key.
        - When registering as an app, the `as_app` dictionary should include at least an "id" key.
        - The decorated function can be either synchronous or asynchronous.
        - Fields `inputs` and `outputs` are can be inferred from the function signature if not explicitly provided, but it's recommended to declare them for clarity.
    """

    def decorator(fn: Callable) -> GraphFunction:
        agent_id = as_agent.get("id") if as_agent else None
        app_id = as_app.get("id") if as_app else None
        gf = GraphFunction(
            name=name,
            fn=fn,
            inputs=inputs,
            outputs=outputs,
            version=version,
            agent_id=agent_id,
            app_id=app_id,
        )
        registry = current_registry()

        if registry is None:
            # no registry available, just return the graph function
            return gf

        base_tags = tags or []

        # prefer explicit description; then docstring; else name
        doc_desc = inspect.getdoc(fn) or None
        eff_description = description or doc_desc or name

        graph_meta: dict[str, Any] = {
            "kind": "graphfn",
            "entrypoint": entrypoint,
            "flow_id": flow_id,
            "tags": base_tags,
            "description": eff_description,
            "inputs": inputs,
            "outputs": outputs,
        }

        registry.register(
            nspace="graphfn",
            name=name,
            version=version,
            obj=gf,
            meta=graph_meta,
        )

        # Register as agent if requested
        # 4Agent meta (if any)
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
                obj=gf,
                meta=agent_meta,
            )

        # 5) App meta (if any)
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
                obj=gf,
                meta=app_meta,
            )

        return gf

    return decorator

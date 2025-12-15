from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from aethergraph.core.runtime.run_registration import RunRegistrationGuard

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
) -> Callable[[Callable], GraphFunction]:
    """Decorator to define a graph function."""

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
        graph_meta: dict[str, Any] = {
            "kind": "graphfn",
            "entrypoint": entrypoint,
            "flow_id": flow_id or name,
            "tags": base_tags,
        }

        registry.register(
            nspace="graphfn",
            name=name,
            version=version,
            obj=gf,
            meta=graph_meta,
        )

        # Register as agent if requested
        if as_agent is not None:
            agent_meta = dict(as_agent)

            agent_id = agent_meta.get("id", name)
            agent_title = agent_meta.get("title", f"Agent for {name}")
            agent_flow_id = agent_meta.get("flow_id", graph_meta["flow_id"])
            agent_tags = agent_meta.get("tags", base_tags)

            extra = {
                k: v for k, v in agent_meta.items() if k not in {"id", "title", "flow_id", "tags"}
            }

            full_agent_meta: dict[str, Any] = {
                "kind": "agent",
                "id": agent_id,
                "title": agent_title,
                "flow_id": agent_flow_id,
                "tags": agent_tags,
                "backing": {"type": "graphfn", "name": name, "version": version},
                **extra,
            }

            registry.register(
                nspace="agent",
                name=agent_id,
                version=version,
                obj=gf,
                meta=full_agent_meta,
            )

        # Register as app if requested
        if as_app is not None:
            app_meta = dict(as_app)

            app_id = app_meta.get("id", name)
            app_flow_id = app_meta.get("flow_id", graph_meta["flow_id"])
            app_name = app_meta.get("name", f"App for {name}")
            app_tags = app_meta.get("tags", base_tags)

            extra = {
                k: v for k, v in app_meta.items() if k not in {"id", "name", "flow_id", "tags"}
            }

            full_app_meta: dict[str, Any] = {
                "kind": "app",
                "id": app_id,
                "name": app_name,
                "graph_id": name,
                "flow_id": app_flow_id,
                "tags": app_tags,
                "backing": {"type": "graphfn", "name": name, "version": version},
                **extra,
            }

            registry.register(
                nspace="app",
                name=app_id,
                version=version,
                obj=gf,
                meta=full_app_meta,
            )

        return gf

    return decorator

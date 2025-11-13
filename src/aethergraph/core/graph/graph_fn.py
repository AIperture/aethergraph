from __future__ import annotations
import inspect
from typing import Any, Callable, Optional, List

from aethergraph.core.runtime.run_registration import RunRegistrationGuard

from .node_spec import TaskNodeSpec
from .graph_refs import GRAPH_INPUTS_NODE_ID
from .interpreter import Interpreter  
from .graph_builder import graph # context manager
from ..runtime.runtime_env import RuntimeEnv
from ..runtime.runtime_registry import current_registry     # ContextVar accessor

from .. execution.retry_policy import RetryPolicy

import inspect, uuid


class GraphFunction:
    def __init__(self, name: str, fn: Callable,
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None,
                 version: str = "0.1.0"):
        self.graph_id = name
        self.name = name
        self.fn = fn
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.version = version
        self.registry_key: Optional[str] = None
        self.last_graph = None
        self.last_context = None
        self.last_memory_snapshot = None

    async def run(self, *, env: RuntimeEnv | None = None, retry: RetryPolicy = RetryPolicy(), max_concurrency: int | None = None, **inputs):
        """
        Build a fresh TaskGraph and execute this function via the Interpreter.
        If 'context' is declared in the fn signature, inject a NodeContext.
        """
        # Build env if not provided (use runner’s builder for consistency)
        if env is None:
            from ..runtime.graph_runner import _build_env  # internal helper
            env, retry, max_concurrency = await _build_env(self, inputs)

        node_spec = TaskNodeSpec(node_id=GRAPH_INPUTS_NODE_ID, type="inputs", metadata={"synthetic": True})
        runtime_ctx = env.make_ctx(node=node_spec, resume_payload=getattr(env, "resume_payload", None))
        node_ctx = runtime_ctx.create_node_context(node=node_spec)

        with graph(name=self.graph_id) as G:
            interp = Interpreter(G, env, retry=retry, max_concurrency=max_concurrency)
            run_id = env.run_id 

            # Register the scheduler for this run_id
            with RunRegistrationGuard(run_id=run_id, scheduler=interp.scheduler, container=env.container):  
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
        """ Async call to run the graph function. 
         Usage: 
            result = await my_graph_fn(input1=value1, input2=value2)
        """
        from ..runtime.graph_runner import run_async 
        return await run_async(self, inputs)

    def sync(self, **inputs):
        """ Synchronous wrapper around async run(). Useful for quick tests or scripts.
         Usage: 
            result = my_graph_fn.sync(input1=value1, input2=value2)
        """
        from ..runtime.graph_runner import run, run_async
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



def graph_fn(name: str,
             inputs: Optional[List[str]] = None,
             outputs: Optional[List[str]] = None,
             version: str = "0.1.0",
             agent: Optional[str] = None, # if agent is set, register this graph fn as an agent with given name
             ) -> Callable[[Callable], GraphFunction]:
    """ Decorator to define a graph function. """
    def decorator(fn: Callable):
        gf = GraphFunction(name=name, fn=fn, inputs=inputs, outputs=outputs, version=version)
        # Register in registry if given
        registry = current_registry()

        if registry is not None:
            registry.register(
                nspace="graphfn",
                name=name,
                version=version,
                obj=gf # we register GraphFunction directly without spec -- graph function is already a runtime object
            )

        if agent:
            assert registry is not None, "No registry available to register agent, make sure to have a current_registry() set up."
            registry.register(
                nspace="agent",
                name=agent,
                version=version,
                obj=gf
            )
        return gf
    return decorator


from functools import wraps
import importlib
import inspect
from typing import Any, Callable, List, Optional
import uuid

from .waitable import DualStageTool, waitable_tool 
from ..execution.step_forward import _normalize_result
from ..graph.graph_builder import current_builder 
from ..graph.interpreter import AwaitableResult, current_interpreter, SimpleNS
from ..runtime.runtime_registry import current_registry
from ..graph.node_handle import NodeHandle 

def _infer_inputs_from_signature(fn: Callable) -> List[str]:
    sig = inspect.signature(fn)
    keys = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        keys.append(p.name)
    return keys

def _normalize_result_to_dict(res: Any) -> dict:
    """ Normalize function result into a dict of outputs.
    Supports:
        - None -> {}
        - dict -> as-is
        - tuple -> {"out0": v0, "out1": v1, ...}
        - single value -> {"result": value}
    """
    if res is None: return {}
    if isinstance(res, dict): return res
    if isinstance(res, tuple): return {f"out{i}": v for i, v in enumerate(res)}
    return {"result": res}

def _check_contract(outputs, out, impl):
    missing = [k for k in outputs if k not in out]
    if missing:
        raise ValueError(f"Tool {getattr(impl, '__name__', type(impl).__name__)} missing outputs: {missing}")
    

def resolve_dotted(path: str):
    """ Resolve a dotted path to a callable. """
    # "pkg.mod:symbol" or "pkg.mod.symbol"
    if ":" in path:
        mod, _, sym = path.partition(":")
        return getattr(importlib.import_module(mod), sym)
    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr)

CONTROL_KW = ("_after", "_name", "_condition", "_id", "_alias", "_labels")

def _split_control_kwargs(kwargs: dict):
    ctrl = {k: kwargs.pop(k) for k in CONTROL_KW if k in kwargs}
    return ctrl, kwargs


def tool(outputs: List[str],
         inputs: Optional[List[str]] = None,
         *,
         name: Optional[str] = None,
         version: str = "0.1.0",
         ):
    """
    Dual-mode decorator for plain functions and DualStageTool classes.
    - Graph mode: builds node (returns NodeHandle)
    - Immediate mode: executes (sync returns dict; async returns awaitable)
    - Registry: if provided, we always record a registry key on the proxy.
      - persist=None -> register_callable (dev hot reload)
      - persist="file" -> persist code under project/tools/... and register_file
    """
    def _wrap(obj):
        # -- normalize impl --
        waitable = inspect.isclass(obj) and issubclass(obj, DualStageTool)
        impl = waitable_tool(obj) if waitable else obj
        sig = inspect.signature(impl)
        declared_inputs = inputs or [
            p.name for p in sig.parameters.values()
            if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]
        is_async = inspect.iscoroutinefunction(impl) or (
            hasattr(impl, "__call__") and inspect.iscoroutinefunction(impl.__call__)
        )

        # -- proxy --
        if is_async:
            async def _immediate(call_kwargs):
                res = await impl(**call_kwargs)
                out = _normalize_result(res)
                _check_contract(outputs, out, impl)
                return out

            @wraps(impl)
            def proxy(*args, **kwargs):
                ctrl, kwargs = _split_control_kwargs(dict(kwargs))  # copy+strip control

                bound = sig.bind_partial(*args, **kwargs); bound.apply_defaults()
                call_kwargs = dict(bound.arguments)
                if current_builder() is not None:
                    return call_tool(proxy, **call_kwargs, **ctrl)
                return _immediate(call_kwargs)
        else:
            @wraps(impl)
            def proxy(*args, **kwargs):
                ctrl, kwargs = _split_control_kwargs(dict(kwargs))  # copy+strip control
                bound = sig.bind_partial(*args, **kwargs); bound.apply_defaults()
                call_kwargs = dict(bound.arguments)
                if current_builder() is not None:
                    return call_tool(proxy, **call_kwargs, **ctrl)
                out = _normalize_result(impl(**call_kwargs))
                _check_contract(outputs, out, impl)
                return out

        # annotate
        proxy.__aether_inputs__  = list(declared_inputs)
        proxy.__aether_outputs__ = list(outputs)
        proxy.__aether_impl__    = impl

        if waitable:
            proxy.__aether_waitable__ = True
            proxy.__aether_tool_class__ = obj  # original class 

        # registry behavior
        registry = current_registry()
        if registry is not None:
            registry.register(
                nspace="tool",
                name=name or getattr(impl, "__name__", "tool"),
                version=version,
                obj=impl,
            )

        return proxy
    return _wrap


def _id_of(x):
    return getattr(x, "node_id", x)  # accepts NodeHandle or str

def _ensure_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return list(x)
    return [x]

def call_tool_old(fn_or_path, **kwargs):   
    builder = current_builder()
    interp = current_interpreter() 
    # --- extract control-plane early

    
    ctrl, kwargs = _split_control_kwargs(kwargs)
    after_raw = ctrl.get("_after", None) 
    name_hint = ctrl.get("_name", None)
    alias = ctrl.get("_alias", None)
    node_id_kw = ctrl.get("_id", None) # hard override for node_id
    labels = _ensure_list(ctrl.get("_labels", None))
    condition = ctrl.get("_condition", None)  # not implemented yet 

    after_ids = [_id_of(a) for a in _ensure_list(after_raw)]

    if interp is not None:
        """ running under an interpreter (graph execution) 
        1. if builder is present, we are in graph-building mode inside a graph(...) context
        2. if no builder, we are in immediate execution mode; run the node directly
        """
        if isinstance(fn_or_path, str):
            logic = fn_or_path
            logic_name  = logic.rsplit(".", 1)[-1]
            inputs_decl  = list(kwargs.keys())
            outputs_decl = ["result"]
        else:
            impl = getattr(fn_or_path, "__aether_impl__", fn_or_path)
            reg_key = getattr(fn_or_path, "__aether_registry_key__", None)

            # Prefer registry for portability (esp. waitables)
            if reg_key:
                logic = f"registry:{reg_key}"            # e.g. "registry:tool:approve_report@0.1.0"
                logic_name = reg_key.split(":")[1].split("@")[0]
                logic_version = reg_key.split("@")[1] if "@" in reg_key else None   
            else:
                logic = f"{impl.__module__}.{getattr(impl,'__name__','tool')}"
                logic_name  = getattr(impl, "__name__", "tool")
                logic_version = getattr(impl, "__version__", None)

            inputs_decl  = getattr(fn_or_path, "__aether_inputs__",  _infer_inputs_from_signature(impl))
            outputs_decl = getattr(fn_or_path, "__aether_outputs__", ["result"])
        
        # add node to the (fresh) graph; dependencies are enforced by the schedule order we create
        node_id = (builder.next_id(logic_name=logic_name) if builder
                   else f"{logic_name}_{uuid.uuid4().hex[:6]}")
        
        if builder is None:
            raise RuntimeError("Interpreter expects a TaskGraph builder context; missing `with graph(...)`")

        if node_id_kw:
            node_id = node_id_kw  # override if provided
        elif alias:
            node_id = alias  # override if provided
        else:
            node_id = builder.next_id(logic_name=logic_name)

        if node_id in builder.spec.nodes:
            raise ValueError(f"Node ID '{node_id}' already exists in graph '{builder.spec.graph_id}'")

        builder.add_tool_node(
            node_id=node_id,
            logic=logic,
            inputs=kwargs,
            expected_input_keys=inputs_decl,
            expected_output_keys=outputs_decl,
            # after=kwargs.pop("_after", None),
            after=after_ids,
            tool_name=logic_name,
            tool_version=logic_version ,
        )
        builder.graph.__post_init__()  # reify runtime nodes
        builder.register_logic_name(logic_name, node_id) # register logic name for reverse lookup
        builder.register_labels(labels, node_id) # register labels for reverse lookup
        if alias:
            builder.register_alias(alias, node_id)
            
        # stash alias/labels in metadata for downstream (promotion, audit)
        builder.spec.nodes[node_id].metadata.update({
            "alias": alias,
            "labels": labels,
            "display_name": name_hint or logic_name,
        })         

        async def _runner():
            outs = await interp.run_one(node=builder.graph.node(node_id))
            # persist on node for audit/trace
            # n = builder.graph.node(node_id)
            # await builder.graph.set_node_outputs(node_id, outs)
            return SimpleNS(outs, node_id=node_id)
        return AwaitableResult(_runner)

    if builder is not None:
        """ building a static graph from within a graph(...) context
        Add a tool node to the current builder graph and return a NodeHandle.
        """

        if isinstance(fn_or_path, str):
            logic = fn_or_path
            logic_name  = logic.rsplit(".", 1)[-1]
            inputs_decl  = list(kwargs.keys())
            outputs_decl = ["result"]
        else:
            impl = getattr(fn_or_path, "__aether_impl__", fn_or_path)
            reg_key = getattr(fn_or_path, "__aether_registry_key__", None)

            # Prefer registry for portability (esp. waitables)
            if reg_key:
                logic = f"registry:{reg_key}"            # e.g. "registry:tool:approve_report@0.1.0"
                logic_name = reg_key.split(":")[1].split("@")[0]
                logic_version = reg_key.split("@")[1] if "@" in reg_key else None
            else:
                logic = f"{impl.__module__}.{getattr(impl,'__name__','tool')}"
                logic_name  = getattr(impl, "__name__", "tool")
                logic_version = getattr(impl, "__version__", None)

            inputs_decl  = getattr(fn_or_path, "__aether_inputs__",  _infer_inputs_from_signature(impl))
            outputs_decl = getattr(fn_or_path, "__aether_outputs__", ["result"])


        if node_id_kw:
            node_id = node_id_kw  # override if provided
        elif alias:
            node_id = alias  # override if provided
        else:
            node_id = builder.next_id(logic_name=logic_name)

        if node_id in builder.spec.nodes:
            raise ValueError(f"Node ID '{node_id}' already exists in graph '{builder.spec.graph_id}'")
        
        builder.add_tool_node(
            node_id=node_id,
            logic=logic,
            inputs=kwargs,
            expected_input_keys=inputs_decl,
            expected_output_keys=outputs_decl,
            after=after_ids,
            tool_name=logic_name,
            tool_version=logic_version ,
        )
        builder.register_logic_name(logic_name, node_id)
        builder.register_labels(labels, node_id)
        if alias:
            builder.register_alias(alias, node_id)

        # stash alias/labels in metadata for downstream (promotion, audit)
        builder.spec.nodes[node_id].metadata.update({
            "alias": alias,
            "labels": labels,
            "display_name": name_hint or logic_name,
        })            
        return NodeHandle(node_id=node_id, output_keys=outputs_decl)



    # immediate mode
    fn = resolve_dotted(fn_or_path) if isinstance(fn_or_path, str) else fn_or_path
    return _normalize_result_to_dict(fn(**kwargs))


def call_tool(fn_or_path, **kwargs):
    builder = current_builder()
    interp = current_interpreter()

    # --- extract control-plane early
    ctrl, kwargs = _split_control_kwargs(kwargs)
    after_raw = ctrl.get("_after", None)
    name_hint = ctrl.get("_name", None)
    alias = ctrl.get("_alias", None)
    node_id_kw = ctrl.get("_id", None)  # hard override for node_id
    labels = _ensure_list(ctrl.get("_labels", None))
    condition = ctrl.get("_condition", None)  # TODO

    after_ids = [_id_of(a) for a in _ensure_list(after_raw)]

    # ---------- Interpreter (reactive) mode ----------
    if interp is not None:
        if isinstance(fn_or_path, str):
            logic = fn_or_path
            logic_name  = logic.rsplit(".", 1)[-1]
            inputs_decl  = list(kwargs.keys())
            outputs_decl = ["result"]
            logic_version = None                       # ✅ ensure defined
        else:
            impl = getattr(fn_or_path, "__aether_impl__", fn_or_path)
            reg_key = getattr(fn_or_path, "__aether_registry_key__", None)
            if reg_key:
                logic = f"registry:{reg_key}"
                logic_name = reg_key.split(":")[1].split("@")[0]
                logic_version = reg_key.split("@")[1] if "@" in reg_key else None
            else:
                logic = f"{impl.__module__}.{getattr(impl,'__name__','tool')}"
                logic_name  = getattr(impl, "__name__", "tool")
                logic_version = getattr(impl, "__version__", None)

            inputs_decl  = getattr(fn_or_path, "__aether_inputs__",  _infer_inputs_from_signature(impl))
            outputs_decl = getattr(fn_or_path, "__aether_outputs__", ["result"])

        if builder is None:
            raise RuntimeError("Interpreter expects a TaskGraph builder context; missing `with graph(...)`")

        # node_id selection
        if node_id_kw:
            node_id = node_id_kw
        elif alias:
            node_id = alias
        else:
            node_id = builder.next_id(logic_name=logic_name)

        if node_id in builder.spec.nodes:
            raise ValueError(f"Node ID '{node_id}' already exists in graph '{builder.spec.graph_id}'")

        builder.add_tool_node(
            node_id=node_id,
            logic=logic,
            inputs=kwargs,
            expected_input_keys=inputs_decl,
            expected_output_keys=outputs_decl,
            after=after_ids,
            tool_name=logic_name,
            tool_version=logic_version,
        )

        # ✅ flush (reify) incrementally instead of calling __post_init__ directly
        if hasattr(builder, "flush"):
            builder.flush()
        else:
            builder.graph.__post_init__()  # fallback

        builder.register_logic_name(logic_name, node_id)
        builder.register_labels(labels, node_id)
        if alias:
            builder.register_alias(alias, node_id)

        builder.spec.nodes[node_id].metadata.update({
            "alias": alias,
            "labels": labels,
            "display_name": name_hint or logic_name,
        })

        async def _runner():
            outs = await interp.run_one(node=builder.graph.node(node_id))
            return SimpleNS(outs, node_id=node_id)      # ✅ include node_id

        return AwaitableResult(_runner, node_id=node_id)

    # ---------- Static build mode (no interpreter, inside graph(...)) ----------
    if builder is not None:
        if isinstance(fn_or_path, str):
            logic = fn_or_path
            logic_name  = logic.rsplit(".", 1)[-1]
            inputs_decl  = list(kwargs.keys())
            outputs_decl = ["result"]
            logic_version = None                       # ✅ ensure defined
        else:
            impl = getattr(fn_or_path, "__aether_impl__", fn_or_path)
            reg_key = getattr(fn_or_path, "__aether_registry_key__", None)
            if reg_key:
                logic = f"registry:{reg_key}"
                logic_name = reg_key.split(":")[1].split("@")[0]
                logic_version = reg_key.split("@")[1] if "@" in reg_key else None
            else:
                logic = f"{impl.__module__}.{getattr(impl,'__name__','tool')}"
                logic_name  = getattr(impl, "__name__", "tool")
                logic_version = getattr(impl, "__version__", None)

            inputs_decl  = getattr(fn_or_path, "__aether_inputs__",  _infer_inputs_from_signature(impl))
            outputs_decl = getattr(fn_or_path, "__aether_outputs__", ["result"])

        if node_id_kw:
            node_id = node_id_kw
        elif alias:
            node_id = alias
        else:
            node_id = builder.next_id(logic_name=logic_name)

        if node_id in builder.spec.nodes:
            raise ValueError(f"Node ID '{node_id}' already exists in graph '{builder.spec.graph_id}'")

        builder.add_tool_node(
            node_id=node_id,
            logic=logic,
            inputs=kwargs,
            expected_input_keys=inputs_decl,
            expected_output_keys=outputs_decl,
            after=after_ids,
            tool_name=logic_name,
            tool_version=logic_version,
        )
        builder.register_logic_name(logic_name, node_id)
        builder.register_labels(labels, node_id)
        if alias:
            builder.register_alias(alias, node_id)

        builder.spec.nodes[node_id].metadata.update({
            "alias": alias,
            "labels": labels,
            "display_name": name_hint or logic_name,
        })

        # Return a build-time handle; ensure it carries node_id
        return NodeHandle(node_id=node_id, output_keys=outputs_decl)  # or SimpleNS({}, node_id=node_id)

    # ---------- Immediate mode (outside graph & interpreter) ----------
    fn = resolve_dotted(fn_or_path) if isinstance(fn_or_path, str) else fn_or_path
    if inspect.iscoroutinefunction(fn):
        async def _run_async():
            return _normalize_result_to_dict(await fn(**kwargs))
        return AwaitableResult(_run_async)               # caller can await
    else:
        return _normalize_result_to_dict(fn(**kwargs))

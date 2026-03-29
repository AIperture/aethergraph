from collections.abc import Callable
from functools import wraps
import importlib
import inspect
from typing import Any

from ..execution.execution_guard import is_tool_execution_active
from ..execution.wait_types import WaitRequested
from ..graph.graph_builder import current_builder
from ..graph.node_handle import NodeHandle
from ..runtime.injection import resolve_node_context_param
from ..runtime.runtime_registry import current_registry
from .waitable import DualStageTool, waitable_tool


def _infer_inputs_from_signature(fn: Callable) -> list[str]:
    sig = inspect.signature(fn)
    context_param = resolve_node_context_param(fn)
    keys = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        if p.name == context_param:
            continue
        keys.append(p.name)
    return keys


def _normalize_result_to_dict(res: Any) -> dict:
    """Normalize function result into a dict of outputs."""
    if res is None:
        return {}
    if isinstance(res, dict):
        return res
    if isinstance(res, tuple):
        return {f"out{i}": v for i, v in enumerate(res)}
    return {"result": res}


def _check_contract(outputs, out, impl):
    missing = [k for k in outputs if k not in out]
    if missing:
        raise ValueError(
            f"Tool {getattr(impl, '__name__', type(impl).__name__)} missing outputs: {missing}"
        )


def resolve_dotted(path: str):
    """Resolve a dotted path to a callable."""
    if ":" in path:
        mod, _, sym = path.partition(":")
        return getattr(importlib.import_module(mod), sym)
    mod, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(mod), attr)


CONTROL_KW = ("_after", "_name", "_condition", "_id", "_alias", "_labels")


def _split_control_kwargs(kwargs: dict):
    ctrl = {k: kwargs.pop(k) for k in CONTROL_KW if k in kwargs}
    return ctrl, kwargs


def _id_of(x):
    return getattr(x, "node_id", x)


def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list | tuple | set):
        return list(x)
    return [x]


def _raise_immediate_control_kw_error(ctrl: dict[str, Any]) -> None:
    if not ctrl:
        return
    names = ", ".join(sorted(ctrl))
    raise RuntimeError(
        "tool_control_kwargs_build_mode_only: tool control kwargs are only supported inside "
        f"@graphify build mode. Received: {names}"
    )


def _execute_immediate_tool(fn_or_path, kwargs: dict[str, Any]):
    fn = resolve_dotted(fn_or_path) if isinstance(fn_or_path, str) else fn_or_path
    impl = getattr(fn, "__aether_impl__", fn)
    outputs = getattr(fn, "__aether_outputs__", ["result"])

    if getattr(fn, "__aether_waitable__", False):
        raise RuntimeError(
            "waitable_tool_immediate_mode_unsupported: DualStageTool-based tools are only "
            "supported during @graphify/@tool node execution."
        )

    async def _run_async():
        try:
            res = await impl(**kwargs)
        except WaitRequested as exc:
            raise RuntimeError(
                "waitable_tool_immediate_mode_unsupported: tools that request waits are only "
                "supported during @graphify/@tool node execution."
            ) from exc
        out = _normalize_result_to_dict(res)
        _check_contract(outputs, out, impl)
        return out

    if inspect.iscoroutinefunction(impl) or (
        callable(impl) and inspect.iscoroutinefunction(impl.__call__)
    ):
        return _run_async()

    try:
        res = impl(**kwargs)
    except WaitRequested as exc:
        raise RuntimeError(
            "waitable_tool_immediate_mode_unsupported: tools that request waits are only "
            "supported during @graphify/@tool node execution."
        ) from exc
    out = _normalize_result_to_dict(res)
    _check_contract(outputs, out, impl)
    return out


def tool(
    outputs: list[str],
    inputs: list[str] | None = None,
    *,
    name: str | None = None,
    version: str = "0.1.0",
):
    """
    Dual-mode decorator for plain functions and DualStageTool classes.

    - Graph mode: builds a tool node inside `@graphify`
    - Immediate mode: executes native Python directly everywhere else
    """

    def _wrap(obj):
        waitable = inspect.isclass(obj) and issubclass(obj, DualStageTool)
        impl = waitable_tool(obj) if waitable else obj
        sig = inspect.signature(impl)
        declared_inputs = (
            inputs
            or getattr(impl, "__aether_inputs__", None)
            or [
                p.name
                for p in sig.parameters.values()
                if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
            ]
        )

        @wraps(impl)
        def proxy(*args, **kwargs):
            ctrl, kwargs = _split_control_kwargs(dict(kwargs))
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            call_kwargs = dict(bound.arguments)

            if current_builder() is not None:
                return call_tool(proxy, **call_kwargs, **ctrl)

            _raise_immediate_control_kw_error(ctrl)

            if is_tool_execution_active():
                raise RuntimeError(
                    "tool_nested_tool_call_disallowed: nested @tool calls inside a graph-executed "
                    "tool body are not supported. Move orchestration into @graphify or plain Python."
                )

            return _execute_immediate_tool(proxy, call_kwargs)

        proxy.__aether_inputs__ = list(declared_inputs)
        proxy.__aether_outputs__ = list(outputs)
        proxy.__aether_impl__ = impl

        if waitable:
            proxy.__aether_waitable__ = True
            proxy.__aether_tool_class__ = obj

        registry = current_registry()
        if registry is not None:
            meta = {
                "kind": "tool",
                "tags": [],
            }
            registry.register(
                nspace="tool",
                name=name or getattr(impl, "__name__", "tool"),
                version=version,
                obj=impl,
                meta=meta,
            )

        return proxy

    return _wrap


def call_tool(fn_or_path, **kwargs):
    builder = current_builder()

    ctrl, kwargs = _split_control_kwargs(kwargs)
    after_raw = ctrl.get("_after", None)
    name_hint = ctrl.get("_name", None)
    alias = ctrl.get("_alias", None)
    node_id_kw = ctrl.get("_id", None)
    labels = _ensure_list(ctrl.get("_labels", None))
    condition = ctrl.get("_condition", None)

    if builder is not None:
        after_ids = [_id_of(a) for a in _ensure_list(after_raw)]
        if isinstance(fn_or_path, str):
            logic = fn_or_path
            logic_name = logic.rsplit(".", 1)[-1]
            inputs_decl = list(kwargs.keys())
            outputs_decl = ["result"]
            logic_version = None
        else:
            impl = getattr(fn_or_path, "__aether_impl__", fn_or_path)
            reg_key = getattr(fn_or_path, "__aether_registry_key__", None)
            if reg_key:
                logic = f"registry:{reg_key}"
                logic_name = reg_key.split(":")[1].split("@")[0]
                logic_version = reg_key.split("@")[1] if "@" in reg_key else None
            else:
                logic = f"{impl.__module__}.{getattr(impl, '__name__', 'tool')}"
                logic_name = getattr(impl, "__name__", "tool")
                logic_version = getattr(impl, "__version__", None)

            inputs_decl = getattr(
                fn_or_path, "__aether_inputs__", _infer_inputs_from_signature(impl)
            )
            outputs_decl = getattr(fn_or_path, "__aether_outputs__", ["result"])

        if node_id_kw:
            node_id = node_id_kw
        elif alias:
            node_id = alias
        else:
            node_id = builder.next_id(logic_name=logic_name)

        if node_id in builder.spec.nodes:
            raise ValueError(
                f"Node ID '{node_id}' already exists in graph '{builder.spec.graph_id}'"
            )

        builder.add_tool_node(
            node_id=node_id,
            logic=logic,
            inputs=kwargs,
            expected_input_keys=inputs_decl,
            expected_output_keys=outputs_decl,
            after=after_ids,
            condition=condition,
            tool_name=logic_name,
            tool_version=logic_version,
        )

        builder.register_logic_name(logic_name, node_id)
        builder.register_labels(labels, node_id)
        if alias:
            builder.register_alias(alias, node_id)

        builder.spec.nodes[node_id].metadata.update(
            {
                "alias": alias,
                "labels": labels,
                "display_name": name_hint or logic_name,
            }
        )

        return NodeHandle(node_id=node_id, output_keys=outputs_decl)

    _raise_immediate_control_kw_error(ctrl)
    return _execute_immediate_tool(fn_or_path, kwargs)

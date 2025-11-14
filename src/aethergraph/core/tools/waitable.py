from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from ..execution.wait_types import WaitRequested, WaitSpec
from ..graph.graph_refs import RESERVED_INJECTABLES


class DualStageTool:
    """
    Subclass and implement:
      - outputs: List[str]   # declare once
      - async def setup(self, **kwargs) -> WaitSpec | Dict[str,Any]
      - async def on_resume(self, resume: Dict[str,Any], **kwargs) -> Dict[str,Any]

    """

    outputs: list[str] = []

    async def setup(self, context, **kwargs) -> Any:
        raise NotImplementedError("DualStageTool subclass must implement setup()")

    async def on_resume(self, resume: dict[str, Any], context: Any) -> dict[str, Any]:
        raise NotImplementedError("DualStageTool subclass must implement on_resume()")


# ----- helpers -----
def _is_coro_fn(obj: Callable) -> bool:
    """Check if obj is a coroutine function or has an async __call__."""
    return inspect.iscoroutinefunction(obj) or (
        callable(obj) and inspect.iscoroutinefunction(obj.__call__)  # async __call__
    )


async def _maybe_await(fn: Callable, *args, **kwargs):
    """Call fn with args; await if it's a coroutine function."""
    if _is_coro_fn(fn):
        return await fn(*args, **kwargs)
    return fn(*args, **kwargs)


def _infer_inputs_from_method(m: Callable) -> list[str]:
    """Infer input parameter names from a method, excluding reserved names and variadic params."""
    sig = inspect.signature(m)
    inputs = []
    for name, p in sig.parameters.items():
        if name in RESERVED_INJECTABLES or name == "self":
            # reserved or self parameter
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            # variadic parameters
            continue
        # treat params with defaults as inputs too (they are optional graph inputs)
        inputs.append(name)
    return inputs


def _infer_inputs_for_waitable(cls_or_inst) -> list[str]:
    # Prefer the setup(...) signature for required inputs
    target = cls_or_inst.setup if inspect.isclass(cls_or_inst) else cls_or_inst.setup
    sig = inspect.signature(target)
    keys = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in {"node", "context", "logger"}:
            continue
        keys.append(p.name)
    return keys


def waitable_tool(cls_or_inst):
    """Wrap a DualStageTool subclass or instance into a runtime callable that:
    - on first call (no resume) -> raises WaitRequested(WaitSpec)
    - on resume -> returns outputs dict
    """

    def _make_instance():
        return cls_or_inst() if inspect.isclass(cls_or_inst) else cls_or_inst

    async def _impl(*, resume=None, context=None, **kwargs):
        tool = _make_instance()
        if hasattr(tool, "bind_node"):
            tool.bind_node(context=context)

        # FIRST CALL: request wait
        if resume is None:
            spec = await _maybe_await(tool.setup, **kwargs, context=context)
            if isinstance(spec, dict):
                # harden to a single shape; ensure channel is a STRING
                chan = spec.get("channel") or None
                spec["channel"] = chan
                spec = WaitSpec(**spec)

            raise WaitRequested(spec.to_dict())

        # RESUME CALL: process resume payload
        out = await _maybe_await(tool.on_resume, resume, context=context)
        return out

    # annotate for graph-mode build
    _impl.__aether_impl__ = _impl  # impl is this coroutine function
    _impl.__aether_inputs__ = _infer_inputs_for_waitable(cls_or_inst)  # optional helper
    _impl.__aether_outputs__ = list(getattr(cls_or_inst, "outputs", [])) or ["result"]
    _impl.__name__ = getattr(cls_or_inst, "__name__", "waitable_tool")
    _impl.__module__ = getattr(cls_or_inst, "__module__", _impl.__module__)
    return _impl

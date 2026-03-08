from __future__ import annotations

from collections.abc import Callable
import inspect
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .node_context import NodeContext

LEGACY_CONTEXT_NAMES = ("context", "ctx")


def _safe_type_hints(fn: Callable) -> dict[str, Any]:
    try:
        return get_type_hints(fn)
    except Exception:
        return getattr(fn, "__annotations__", {}) or {}


def is_node_context_annotation(annotation: Any) -> bool:
    if annotation is NodeContext:
        return True

    origin = get_origin(annotation)
    if origin not in (UnionType, Union):
        return False

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return len(args) == 1 and args[0] is NodeContext


def resolve_node_context_param(fn: Callable) -> str | None:
    sig = inspect.signature(fn)
    hints = _safe_type_hints(fn)

    typed_candidates: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            continue
        annotation = hints.get(name, param.annotation)
        if is_node_context_annotation(annotation):
            typed_candidates.append(name)

    if len(typed_candidates) > 1:
        raise TypeError(
            f"{getattr(fn, '__name__', type(fn).__name__)} declares multiple NodeContext parameters: "
            f"{typed_candidates}. Use exactly one."
        )

    if typed_candidates:
        return typed_candidates[0]

    legacy_candidates = [
        name
        for name, param in sig.parameters.items()
        if name in LEGACY_CONTEXT_NAMES
        and param.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
    ]
    if len(legacy_candidates) > 1:
        raise TypeError(
            f"{getattr(fn, '__name__', type(fn).__name__)} declares multiple legacy context aliases: "
            f"{legacy_candidates}. Use exactly one."
        )
    return legacy_candidates[0] if legacy_candidates else None


def pop_explicit_node_context(kwargs: dict[str, Any]) -> Any:
    has_context = "context" in kwargs
    has_ctx = "ctx" in kwargs
    if has_context and has_ctx:
        raise TypeError(
            "Pass only one of 'context' or 'ctx' when providing an explicit NodeContext."
        )
    if has_context:
        return kwargs.pop("context")
    if has_ctx:
        return kwargs.pop("ctx")
    return None

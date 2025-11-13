from __future__ import annotations
import asyncio
from typing import Any

__all__ = ["BindAdapter"]

class BindAdapter:
    """
    Wrap any object and make it context-bindable.
    Convention: if a wrapped method wants NodeContext, accept kwarg `_ctx`.
    """
    def __init__(self, obj: Any):
        self._obj = obj

    def bind(self, *, context):
        obj = self._obj
        ctx = context

        class Bound:
            def __getattr__(self, name: str):
                attr = getattr(obj, name)
                if callable(attr):
                    if asyncio.iscoroutinefunction(attr):
                        async def aw(*a, **k):
                            k.setdefault("_ctx", ctx)
                            return await attr(*a, **k)
                        return aw
                    else:
                        def sw(*a, **k):
                            k.setdefault("_ctx", ctx)
                            return attr(*a, **k)
                        return sw
                return attr
        return Bound()

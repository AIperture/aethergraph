from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_TOOL_EXEC_ACTIVE: ContextVar[bool] = ContextVar("_TOOL_EXEC_ACTIVE", default=False)


def is_tool_execution_active() -> bool:
    """Return True when runtime is currently executing inside a tool body."""
    return _TOOL_EXEC_ACTIVE.get()


@contextmanager
def enter_tool_execution():
    """
    Mark the current context as "inside a tool execution".
    Nested @tool proxy invocations can use this signal to reject tool-in-tool orchestration.
    """
    token = _TOOL_EXEC_ACTIVE.set(True)
    try:
        yield
    finally:
        _TOOL_EXEC_ACTIVE.reset(token)

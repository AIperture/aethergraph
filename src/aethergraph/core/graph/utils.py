import inspect
from typing import Any


# ---------- helpers for printing and debugging ----------
def _short(x: Any, maxlen: int = 42) -> str:
    """Shorten a string representation to maxlen, adding ellipsis if needed."""
    s = str(x)
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def _status_label(s: Any) -> str:
    """Return a string label for a status value.

    E.g., if s is an Enum-like object with a .name attribute, return that.
    """
    # Accept Enum-like (with .name), strings, or None
    if s is None:
        return "-"
    return getattr(s, "name", str(s))


def _logic_label(logic: Any) -> str:
    """Return a string label for a logic value.

    E.g., if logic is a function, return its module and name.
    """
    # Show a dotted path when possible; fall back to repr/str
    if isinstance(logic, str):
        return logic
    # Unwrap @tool proxies if present
    impl = getattr(logic, "__aether_impl__", logic)
    if inspect.isfunction(impl) or inspect.ismethod(impl):
        mod = getattr(impl, "__module__", None) or ""
        name = getattr(impl, "__name__", None) or "tool"
        return f"{mod}.{name}".strip(".")
    return _short(repr(logic), 80)

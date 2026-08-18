"""AetherGraph server runtime package."""

from __future__ import annotations

from typing import Any

__all__ = ["start_server", "start_server_async", "stop_server"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from . import start

    return getattr(start, name)

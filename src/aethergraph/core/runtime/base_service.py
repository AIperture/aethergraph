"""Base services are used for external context services and other common patterns.
Here we register service in the main runtime, and provide base classes for
TODO: confirm that external services runs with the main event loop locally, not the sidecar loop, such that asyncio.locks work as expected.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .node_context import NodeContext

__all__ = [
    "AsyncRWLock",
    "BaseContextService",
    "Service",
    "maybe_await",
]


class _ServiceHandle:
    """
    This class provides ergonomic access to a service instance, allowing attribute access,
    direct retrieval, and forwarding of calls if the underlying service is callable.
    Examples:
        Accessing service attributes:
        ```python
        handle = _ServiceHandle("my_service", service_instance)
        value = handle.some_attribute
        ```
        Retrieving the service instance:
        ```python
        svc = handle()
        ```
        Forwarding calls to a callable service:
        ```python
        result = handle(arg1, arg2)
        ```
    Args:
        name: The name of the service for identification and error reporting.
        bound_service: The actual service instance to be wrapped.
    Attribute Access:
        All attribute lookups are delegated to the underlying service instance.
    Calling:
        - If called with no arguments, returns the underlying service instance.
        - If called with arguments and the service is callable, forwards the call.
        - Raises TypeError if the service is not callable and called with arguments.
    Returns:
        The result of the underlying service's __call__ method, or the service instance itself
        if called with no arguments.
    """

    __slots__ = ("_name", "_svc")

    def __init__(self, name: str, bound_service: object):
        self._svc = bound_service
        self._name = name

    def __getattr__(self, attr):
        return getattr(self._svc, attr)

    def __call__(self, *args, **kwargs):
        # No-arg call => return the service instance (consistent, non-surprising)
        if not args and not kwargs:
            return self._svc

        # If the underlying service is callable, forward the call
        if callable(self._svc):
            return self._svc(*args, **kwargs)

        raise TypeError(
            f"Service '{self._name}' is not directly callable; "
            "call with no arguments to get the service instance, "
            "then invoke its methods."
        )

    def __repr__(self):
        return f"<ServiceHandle {self._name}: {self._svc!r}>"


async def maybe_await(x: Any) -> Any:
    """If x is awaitable, await it; else return it directly."""
    if asyncio.iscoroutine(x) or isinstance(x, Awaitable):
        return await x
    return x


class AsyncRWLock:
    """Simple async RW lock: many readers or one writer."""

    def __init__(self):
        self._readers = 0
        self._r_lock = asyncio.Lock()
        self._w_lock = asyncio.Lock()

    async def read(self):
        lock = self

        class _Guard:
            async def __aenter__(self):
                async with lock._r_lock:
                    lock._readers += 1
                    if lock._readers == 1:
                        await lock._w_lock.acquire()

            async def __aexit__(self, exc_type, exc, tb):
                async with lock._r_lock:
                    lock._readers -= 1
                    if lock._readers == 0:
                        lock._w_lock.release()

        return _Guard()

    async def write(self):
        return self._w_lock


class BaseContextService:
    """
        Batteries-included base for context services.
    - Lifecycle: start/close (async)
    - Binding: bind(context) returns a context-aware handle (default: self with ContextVar)
    - Concurrency: critical() async mutex, AsyncRWLock for R/W scenarios
    - Utilities: run_blocking() for CPU/IO-bound sync functions
    """

    _current_ctx: ContextVar = ContextVar("_aeg_ctx", default=None)

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._closing = False

    # ---------- lifecycle ----------
    async def start(self) -> None:
        """Async startup hook."""
        return None

    async def close(self) -> None:
        """Async shutdown hook."""
        self._closing = True
        return None

    # ---------- binding ----------
    def bind(self, *, context: NodeContext) -> BaseContextService:
        """Return a context-bound handle to this service."""
        self._current_ctx.set(context)
        return self

    def ctx(self) -> NodeContext:
        ctx = self._current_ctx.get()
        if ctx is None:
            raise RuntimeError("No context bound to this service. Call bind(context) first.")
        return ctx

    # ---------- concurrency ----------
    def critical(self):
        """Decorator for async critical section (mutex)."""

        def deco(fn: Callable[..., Any]) -> Any:
            async def wrapped(*a, **kw):
                async with self._lock:
                    return await maybe_await(fn(*a, **kw))

            return wrapped

        return deco

    async def run_blocking(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run a blocking function in a thread pool."""
        return await asyncio.to_thread(fn, *args, **kwargs)


# Alias for ergonomics
Service = BaseContextService

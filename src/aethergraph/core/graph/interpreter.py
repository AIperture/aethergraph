from __future__ import annotations
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .task_graph import TaskGraph
from .task_node import TaskNodeRuntime
from ..runtime.runtime_env import RuntimeEnv
from ..execution.retry_policy import RetryPolicy 


# Public ContextVar for current runtime environment
_INTERP_CTX: ContextVar["Interpreter | None"] = ContextVar("_INTERP_CTX", default=None)

class SimpleNS:
    """
    Lightweight attribute-access wrapper for a dict.
    Used as a 'handle' for tool nodes during graph build and also as a thin
    outputs view at runtime. Must carry node_id so the scheduler can target it.
    """
    __slots__ = ("_data", "_node_id")

    def __init__(self, d: Dict[str, Any] | None = None, *, node_id: str | None = None):
        self._data = dict(d or {})
        self._node_id = node_id  # may be None for plain dict-like use

    # ---- Introspection ----
    @property
    def node_id(self) -> str | None:
        return self._node_id

    def has_node(self) -> bool:
        return self._node_id is not None

    # ---- Dict-ish API ----
    def to_dict(self) -> Dict[str, Any]:
        # do NOT include node_id in the payload view
        return dict(self._data)

    def get(self, name, default=None):
        return self._data.get(name, default)

    def __getitem__(self, key: str):
        return self._data[key]

    def __getattr__(self, name: str):
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"No such attribute: {name}")

    def __repr__(self):
        nid = f" node_id={self._node_id}" if self._node_id else ""
        return f"SimpleNS({self._data!r}{nid})"

    # ---- Builder-time ref helpers ----
    def ref(self, *path: str) -> Dict[str, Any]:
        """
        Return a binding-ref dict usable as an input to another node during build:
            handle.ref("result")  -> {'_type':'ref','from':<node_id>,'path':['result']}
        If no path is given, refers to the entire outputs dict.
        """
        if not self._node_id:
            raise RuntimeError("Cannot create ref(): handle has no node_id (not a tool handle?)")
        return {"_type": "ref", "from": self._node_id, "path": list(path or [])}

    def on(self, key: str) -> Dict[str, Any]:
        """Alias for ref(key)."""
        return self.ref(key)

class AwaitableResult:
    """ Lightweight awaitable wrapper for a result value. """
    def __init__(self, coro: Callable[[], Any], *, node_id: str | None = None):
        self._coro = coro
        self.node_id = node_id
    
    def __await__(self):
        return self._coro().__await__()
    
@dataclass
class Interpreter:
    """ Binds a TaskGraph to a scheduler, let tools add+run nodes on the fly. """
    graph: "TaskGraph"
    env: RuntimeEnv
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    max_concurrency: int = 1 



    def __post_init__(self):
        from ..execution.forward_scheduler import ForwardScheduler
        # get logger from env's container
        from ..runtime.runtime_services import current_logger_factory
        logger = current_logger_factory().for_scheduler()
        
        self.scheduler = ForwardScheduler(
            self.graph,
            self.env,
            self.retry,
            max_concurrency=self.max_concurrency,
            logger=logger
        )

    # NEW: convenience pass-through
    def add_listener(self, cb):
        self.scheduler.add_listener(cb)

    # NEW: run the whole plan (mirrors ForwardScheduler.run)
    async def run(self):
        """ Run the entire graph to completion. """
        return await self.scheduler.run()

    def enter(self):
        """ Enter the interpreter context. """
        class _Guard:
            def __init__(_g, interp: Interpreter):
                _g.interp = interp
                _g.token = None
            def __enter__(_g):
                _g.token = _INTERP_CTX.set(_g.interp)
                return _g.interp
            def __exit__(_g, exc_type, exc_val, exc_tb):
                """ Exit the interpreter context. 
                    Args:
                        exc_type, exc_val, exc_tb: exception info if any
                """
                _INTERP_CTX.reset(_g.token)
        return _Guard(self)

    async def run_one(self, node: "TaskNodeRuntime") -> Dict[str, Any]:
        """ Run a single node by ID, return its outputs. """
        return await self.scheduler.run_one(node)


# Convenience helpers
def current_interpreter() -> Optional[Interpreter]:
    """ Get the current interpreter from context, or None if not in one. """
    return _INTERP_CTX.get()


import threading
from typing import Dict, Optional
from aethergraph.core.execution.forward_scheduler import ForwardScheduler
from contextlib import asynccontextmanager

from aethergraph.core.execution.global_scheduler import GlobalForwardScheduler

class SchedulerRegistry:
    def __init__(self):
        self._by_run: Dict[str, GlobalForwardScheduler] = {}
        self._lock = threading.RLock()

    def register(self, run_id: str, scheduler: GlobalForwardScheduler) -> None:
        with self._lock:
            self._by_run[run_id] = scheduler

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._by_run.pop(run_id, None)

    def get(self, run_id: str) -> Optional[GlobalForwardScheduler]:
        with self._lock:
            return self._by_run.get(run_id)

    def list_run_ids(self) -> Dict[str, GlobalForwardScheduler]:
        with self._lock:
            return dict(self._by_run)


@asynccontextmanager
async def registered_scheduler(registry: SchedulerRegistry, run_id: str, scheduler):
    registry.register(run_id, scheduler)
    try:
        yield
    finally:
        registry.unregister(run_id)


"""# Example usage:
async with registered_scheduler(SCHEDULERS, run_id, scheduler):
    await scheduler.run()
"""
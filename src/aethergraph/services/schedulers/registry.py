from __future__ import annotations

import asyncio
import threading
from typing import Any, Protocol


class _SchedulerControl(Protocol):
    """Minimal in-process control surface for a registered run scheduler."""

    loop: asyncio.AbstractEventLoop | None

    async def on_resume_event(
        self,
        run_id: str,
        node_id: str,
        payload: dict[str, Any],
    ) -> None: ...

    async def terminate(self) -> None: ...


class SchedulerRegistry:
    def __init__(self):
        self._by_run: dict[str, _SchedulerControl] = {}
        self._lock = threading.RLock()

    def register(self, run_id: str, scheduler: _SchedulerControl) -> None:
        """Register the active local scheduler for one run.

        Intro:
            Binds a run identity to the minimal in-process scheduler control used
            by resume delivery and cancellation.

        Examples:
            Register a newly started run:
            ```python
            registry.register("run-1", scheduler)
            ```

            Replace an intentionally restarted run binding:
            ```python
            registry.register("run-1", recovered_scheduler)
            ```

        Args:
            run_id: Exact active run identity.
            scheduler: Local scheduler control for that run.

        Returns:
            None: The registry is updated before returning.

        Notes:
            `RunRegistrationGuard` owns normal registration lifetime and removes
            the binding when execution exits.
        """
        with self._lock:
            self._by_run[run_id] = scheduler

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._by_run.pop(run_id, None)

    def get(self, run_id: str) -> _SchedulerControl | None:
        """Resolve the active local scheduler for a run.

        Intro:
            Reads one registry binding without exposing the registry's mutable
            internal dictionary.

        Examples:
            Resolve a running scheduler:
            ```python
            scheduler = registry.get("run-1")
            ```

            Detect an inactive run:
            ```python
            if registry.get("run-missing") is None:
                print("not active")
            ```

        Args:
            run_id: Exact run identity to resolve.

        Returns:
            _SchedulerControl | None: Registered scheduler, or `None` when the
            run has no active in-process scheduler.

        Notes:
            A missing binding does not consume or delete any durable continuation.
        """
        with self._lock:
            return self._by_run.get(run_id)

    def list_run_ids(self) -> dict[str, _SchedulerControl]:
        """Snapshot all active run-to-scheduler bindings.

        Intro:
            Returns a shallow copy suitable for diagnostics without granting
            mutation access to registry state.

        Examples:
            List active run identities:
            ```python
            run_ids = list(registry.list_run_ids())
            ```

            Inspect the current binding count:
            ```python
            active_count = len(registry.list_run_ids())
            ```

        Args:
            None.

        Returns:
            dict[str, _SchedulerControl]: Snapshot keyed by exact run identity.

        Notes:
            Later registry changes do not mutate the returned dictionary.
        """
        with self._lock:
            return dict(self._by_run)

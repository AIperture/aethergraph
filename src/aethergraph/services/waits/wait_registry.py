import asyncio
import threading
from typing import Any


class WaitRegistry:
    """
    In-process registry for cooperative waits.
    - register(token): binds a Future to the *current* running loop
    - resolve(token, payload): from any thread/loop, completes that Future
    - cancel(token): cancels a pending wait

    Use this only for cooperative (same-process) resumes.
    All other resumes should go via ResumeBus/Scheduler.
    """

    def __init__(self) -> None:
        # token -> (owning_loop, future)
        self._futs: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._lock = threading.RLock()
        # If a resume arrives before register()
        self._pending_payloads: dict[str, Any] = {}

    def register(self, token: str) -> asyncio.Future:
        """Create or reuse a Future on the current loop; deliver any early payload."""
        loop = asyncio.get_running_loop()
        with self._lock:
            entry = self._futs.get(token)
            if entry:
                el, fut = entry
                if fut.done() or getattr(el, "is_closed", lambda: False)():
                    fut = loop.create_future()
                    self._futs[token] = (loop, fut)
            else:
                fut = loop.create_future()
                self._futs[token] = (loop, fut)
                # deliver early resume if present
                if token in self._pending_payloads:
                    payload = self._pending_payloads.pop(token)
                    loop.call_soon(self._deliver_if_pending, fut, payload)
            return fut

    @staticmethod
    def _deliver_if_pending(future: asyncio.Future, payload: Any) -> None:
        if not future.done():
            future.set_result(payload)

    def resolve(
        self,
        token: str,
        payload: dict | None = None,
        *,
        cache_if_missing: bool = True,
    ) -> bool:
        """Schedule one response on the registered waiter's owning loop.

        Calls may originate from another thread. Completed waiters are never
        released again; cancellation before the callback runs is respected.

        Examples:
            Retain an early response before registration:
            ```python
            waits.resolve("wait-1", {"text": "Ready"})
            ```
            Deliver a committed response only to its existing waiter:
            ```python
            scheduled = waits.resolve("wait-1", answer, cache_if_missing=False)
            ```

        Args:
            token: Exact in-process wait identity.
            payload: Response delivered to its waiter.
            cache_if_missing: Retain an early response when registration has not
                occurred. False for responses to already-registered durable waits.

        Returns:
            bool: True when delivery was scheduled on an active registered waiter.

        Notes:
            A missing waiter never causes execution to be launched.
        """
        payload = payload or {}
        with self._lock:
            entry = self._futs.pop(token, None)
            if not entry:
                # resume before register: stash
                if cache_if_missing:
                    self._pending_payloads[token] = payload
                return False
            loop, fut = entry

        if fut.done() or loop.is_closed():
            return False
        loop.call_soon_threadsafe(self._deliver_if_pending, fut, payload)
        return True

    def cancel(self, token: str, exc: BaseException | None = None) -> bool:
        """Cancel from any thread; returns True if a Future was present."""
        with self._lock:
            entry = self._futs.pop(token, None)
            self._pending_payloads.pop(token, None)
        if not entry:
            return False
        loop, fut = entry
        if not fut.done():
            loop.call_soon_threadsafe(
                fut.set_exception, exc or asyncio.CancelledError(f"Wait cancelled: {token}")
            )
        return True

    # --- optional helpers ---
    def has(self, token: str) -> bool:
        with self._lock:
            return token in self._futs or token in self._pending_payloads

    def size(self) -> int:
        with self._lock:
            return len(self._futs)

    def shutdown(self) -> None:
        """Best-effort cleanup; cancels outstanding futures."""
        with self._lock:
            items = list(self._futs.items())
            self._futs.clear()
            self._pending_payloads.clear()
        for _, (loop, fut) in items:
            if not fut.done():
                loop.call_soon_threadsafe(
                    fut.set_exception, asyncio.CancelledError("Registry shutdown")
                )

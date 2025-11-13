import asyncio
import threading
from typing import Dict, Any, Tuple, Optional

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
        self._futs: Dict[str, Tuple[asyncio.AbstractEventLoop, asyncio.Future]] = {}
        self._lock = threading.RLock()
        # If a resume arrives before register()
        self._pending_payloads: Dict[str, Any] = {}

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
                    loop.call_soon(fut.set_result, payload)
            return fut

    def resolve(self, token: str, payload: Optional[dict] = None) -> bool:
        """Resolve from any thread; returns True if delivered to a registered Future."""
        payload = payload or {}
        with self._lock:
            entry = self._futs.pop(token, None)
            if not entry:
                # resume before register: stash
                self._pending_payloads[token] = payload
                return False
            loop, fut = entry

        if not fut.done():
            loop.call_soon_threadsafe(fut.set_result, payload)
        return True

    def cancel(self, token: str, exc: Optional[BaseException] = None) -> bool:
        """Cancel from any thread; returns True if a Future was present."""
        with self._lock:
            entry = self._futs.pop(token, None)
            self._pending_payloads.pop(token, None)
        if not entry:
            return False
        loop, fut = entry
        if not fut.done():
            loop.call_soon_threadsafe(
                fut.set_exception,
                exc or asyncio.CancelledError(f"Wait cancelled: {token}")
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

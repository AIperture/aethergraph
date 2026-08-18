from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock
import time


class RunBurstLimiter:
    """Apply a process-local sliding-window admission limit to run requests."""

    def __init__(
        self,
        max_events: int,
        window_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be greater than zero")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        self.max_events = max_events
        self.window_seconds = window_seconds
        self._monotonic = monotonic
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        """Reserve one request slot when the identity remains below its burst cap.

        Intro:
            Applies an atomic process-local sliding window for one normalized
            request identity.

        Examples:
            Admit the first request:
            ```python
            assert limiter.allow("org-a")
            ```

            Reject a request above the configured cap:
            ```python
            limiter.allow("org-a")
            assert not limiter.allow("org-a")
            ```

        Args:
            key: Normalized tenant or user admission identity.

        Returns:
            bool: `True` when a slot was reserved, otherwise `False`.

        Notes:
            This limiter coordinates threads in one server process. Distributed
            admission requires a shared external implementation.
        """
        now = self._monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_events:
                return False
            events.append(now)
            return True

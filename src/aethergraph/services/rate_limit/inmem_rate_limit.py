from collections import defaultdict, deque
import time


class SimpleRateLimiter:
    def __init__(self, max_events: int, window_seconds: int):
        self.max_events = max_events
        self.window = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        dq = self._events[key]
        cutoff = now - self.window

        # Drop old events
        while dq and dq[0] < cutoff:
            dq.popleft()

        if len(dq) >= self.max_events:
            return False  # Rate limit exceeded

        dq.append(now)
        return True

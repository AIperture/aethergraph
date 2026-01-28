from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any


class EventHub:
    """
    In-memory pub/sub for UI events.

    - Keys by scope_id (session_id or run_id).
    - Optionally filters by kind (e.g. "session_chat", "run_channel").
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def subscribe(self, scope_id: str) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator: yields raw EventLog-like rows with keys:
        { "id", "ts", "scope_id", "kind", "payload": {...} }
        """
        print(f"EventHub: subscribe called for scope_id={scope_id}")
        q: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers[scope_id].add(q)

        try:
            while True:
                row = await q.get()
                yield row
        finally:
            async with self._lock:
                self._subscribers[scope_id].discard(q)
                if not self._subscribers[scope_id]:
                    self._subscribers.pop(scope_id, None)

    async def broadcast(self, row: dict[str, Any]) -> None:
        scope_id = row.get("scope_id")
        if not scope_id:
            return
        async with self._lock:
            subs = list(self._subscribers.get(scope_id, []))
        for q in subs:
            # Best-effort; if queue is full we drop rather than block worker
            try:
                q.put_nowait(row)
            except asyncio.QueueFull:
                # log if you want
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"EventHub queue full for scope_id={scope_id}, dropping event")

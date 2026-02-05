from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

_SENTINEL = object()


class EventHub:
    """
    In-memory pub/sub for UI events.
    Keys by (scope_id, kind).
    """

    def __init__(self) -> None:
        self._subscribers: dict[tuple[str, str], set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._closed = False

    async def subscribe(
        self,
        *,
        scope_id: str,
        kind: str,
        max_queue: int = 256,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator that yields rows for (scope_id, kind).
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)

        async with self._lock:
            if self._closed:
                return
            self._subscribers[(scope_id, kind)].add(q)

        try:
            while True:
                item = await q.get()
                if item is _SENTINEL:
                    return
                yield item
        finally:
            async with self._lock:
                self._subscribers[(scope_id, kind)].discard(q)
                if not self._subscribers[(scope_id, kind)]:
                    self._subscribers.pop((scope_id, kind), None)

    async def broadcast(self, row: dict[str, Any]) -> None:
        scope_id = row.get("scope_id")
        kind = row.get("kind")
        if not scope_id or not kind:
            return

        async with self._lock:
            subs = list(self._subscribers.get((scope_id, kind), []))

        for q in subs:
            try:
                q.put_nowait(row)
            except asyncio.QueueFull:
                # Drop instead of blocking producer
                continue

    async def close(self) -> None:
        """
        Wake and detach all subscribers (useful on shutdown/reload).
        """
        async with self._lock:
            self._closed = True
            all_queues = []
            for qs in self._subscribers.values():
                all_queues.extend(list(qs))
            self._subscribers.clear()

        for q in all_queues:
            with suppress(Exception):
                q.put_nowait(_SENTINEL)

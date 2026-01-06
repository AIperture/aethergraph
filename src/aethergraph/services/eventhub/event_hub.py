from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable

Subscriber = Callable[[dict], Awaitable[None]]

"""
Lightweight in-memory pub/sub hub for UI events.

This sits *alongside* EventLog:

- EventLog = durable storage (what HTTP polling uses)
- EventHub = transient fan-out for live WebSocket subscribers

Later:
- You can swap this for Redis/Kafka/etc without changing the adapters.
"""


class EventHub:
    """
    Pub/sub keyed by (scope_id, kind).

    For example:
      scope_id="session:abc123", kind="session_chat"
      scope_id="run:xyz456", kind="run_channel"
    """

    def __init__(self) -> None:
        # (scope_id, kind) -> set of async callbacks
        self._subs: dict[tuple[str, str], set[Subscriber]] = defaultdict(set)

    def subscribe(self, scope_id: str, kind: str, cb: Subscriber) -> None:
        """Register a callback that should receive new rows for this (scope, kind)."""
        self._subs[(scope_id, kind)].add(cb)

    def unsubscribe(self, scope_id: str, kind: str, cb: Subscriber) -> None:
        """Remove a previously registered callback."""
        key = (scope_id, kind)
        subs = self._subs.get(key)
        if not subs:
            return
        subs.discard(cb)
        if not subs:
            # Optional: cleanup empty sets
            self._subs.pop(key, None)

    async def broadcast(self, row: dict) -> None:
        """
        Push a new EventLog row to all subscribers.

        Expected row schema:
          {
            "id": str,
            "ts": float,
            "scope_id": str,
            "kind": str,
            "payload": {...},
          }
        """
        scope_id = row.get("scope_id")
        kind = row.get("kind")
        if not scope_id or not kind:
            return

        # Snapshot to avoid mutation during iteration
        subs = list(self._subs.get((scope_id, kind), []))

        for cb in subs:
            try:
                await cb(row)
            except Exception:
                # TODO: log error; maybe drop the subscriber if repeatedly failing
                # For now, we ignore to avoid breaking others.
                continue

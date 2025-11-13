# services/eventbus/inmem.py
import asyncio, threading
from collections import defaultdict

class InMemoryEventBus:
    def __init__(self):
        self._subs = defaultdict(list)  # topic -> [handlers]
        self._lock = threading.RLock()  # TODO: check if we need thread safety

    async def publish(self, topic, event):
        # fanout without blocking publishers
        with self._lock:
            subs = list(self._subs.get(topic, []))
        for h in subs:
            asyncio.create_task(h(event))  # fire-and-forget

    async def subscribe(self, topic, handler):
        with self._lock:
            self._subs[topic].append(handler)

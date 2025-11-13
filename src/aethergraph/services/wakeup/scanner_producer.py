import asyncio
from datetime import datetime, timezone

from aethergraph.contracts.services.wakeup import WakeupQueue

# services/wakeup/scanner_producer.py
class ScannerProducer:
    def __init__(self, store, queue: WakeupQueue, logger, tick_sec=1.0, topic="default"):
        self.store = store
        self.queue = queue
        self.logger = logger
        self.tick_sec = tick_sec
        self.topic = topic
        self._task = None
        self._stopped = asyncio.Event()

    def start(self): 
        if not self._task:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stopped.set()
        if self._task:
            await self._task

    async def _loop(self):
        while not self._stopped.is_set():
            await asyncio.sleep(self.tick_sec)
            now = datetime.now(timezone.utc)
            for c in self._iter_continuations():   
                if c.poll:
                    # poll; if hit -> enqueue
                    payload = await self._try_poll(c)
                    if payload is not None:
                        await self.queue.enqueue(self.topic, {
                            "kind": "resume",
                            "run_id": c.run_id,
                            "node_id": c.node_id,
                            "token": c.token,
                            "payload": payload
                        })
                elif c.next_wakeup_at and now >= c.next_wakeup_at:
                    await self.queue.enqueue(self.topic, {
                        "kind": "resume",
                        "run_id": c.run_id,
                        "node_id": c.node_id,
                        "token": c.token,
                        "payload": {"deadline_fired": True}
                    })

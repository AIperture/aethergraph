import asyncio
import time

"""This is a template implementation of a WakeupWatcher that periodically checks for
Currenlty, we have not materialize the wakeup method in Aethergraph core.
"""


class WakeupWatcher:
    def __init__(self, cont_store, resume_bus, poll_sec: int = 10):
        self.cont_store = cont_store
        self.resume_bus = resume_bus
        self.poll_sec = poll_sec
        self._task = None
        self._stop = asyncio.Event()

    async def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._stop.set()
        if self._task:
            await self._task

    async def _loop(self):
        while not self._stop.is_set():
            now = time.time()
            due = await self.cont_store.list_due_wakeups(now)
            for cont in due:
                # Publish to bus; bus routes to the right scheduler via run_id ↦ scheduler mapping
                await self.resume_bus.post_wakeup(cont.run_id, cont.node_id)
                # update next_wakeup in the store if needed
                await self.cont_store.bump_wakeup(cont)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_sec)
            except asyncio.TimeoutError:
                import logging

                logger = logging.getLogger("aethergraph.core.runtime.wakeup_watcher")
                logger.info("WakeupWatcher polling for due continuations...")

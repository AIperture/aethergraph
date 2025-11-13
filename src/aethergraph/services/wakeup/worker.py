import asyncio

from aethergraph.contracts.services.wakeup import WakeupQueue

class WakeWorker:
    def __init__(self, queue: WakeupQueue, resume_bus, logger, topic="default"):
        self.queue = queue
        self.resume_bus = resume_bus
        self.logger = logger
        self.topic = topic

    async def run_forever(self):
        while True:
            leases = await self.queue.lease(self.topic, max_items=1, lease_s=60)
            if not leases:
                await asyncio.sleep(0.2); continue
            lease = leases[0]
            try:
                msg = lease.msg
                await self.resume_bus.enqueue_resume(
                    run_id=msg["run_id"], node_id=msg["node_id"],
                    token=msg["token"], payload=msg["payload"]
                )
                await self.queue.ack(lease)
            except Exception as e:
                self.logger.error("resume_failed", err=str(e), node=msg.get("node_id"))
                await self.queue.nack(lease, requeue_delay_s=2)

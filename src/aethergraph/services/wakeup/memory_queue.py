import time, uuid, heapq, threading
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class _Lease:
    id: str
    msg: dict
    visibility_deadline: float

class ThreadSafeWakeupQueue:
    def __init__(self):
        self._ready: List[tuple[float,int,dict]] = []
        self._inflight: Dict[str,_Lease] = {}
        self._ctr = 0
        self._lock = threading.RLock()

    async def enqueue(self, topic: str, msg: dict, delay_s: float = 0) -> str:
        with self._lock:
            self._ctr += 1
            heapq.heappush(self._ready, (time.time()+delay_s, self._ctr, msg))
            return msg.get("job_id") or str(self._ctr)

    async def lease(self, topic: str, max_items: int = 1, lease_s: int = 60) -> List[_Lease]:
        out = []
        now = time.time()
        with self._lock:
            while self._ready and len(out) < max_items:
                visible_at, _, msg = self._ready[0]
                if visible_at > now: break
                heapq.heappop(self._ready)
                lid = uuid.uuid4().hex
                lease = _Lease(lid, msg, now+lease_s)
                self._inflight[lid] = lease
                out.append(lease)
        return out

    async def extend(self, lease: _Lease, lease_s: int) -> None:
        with self._lock:
            if lease.id in self._inflight:
                self._inflight[lease.id].visibility_deadline = time.time()+lease_s

    async def ack(self, lease: _Lease) -> None:
        with self._lock:
            self._inflight.pop(lease.id, None)

    async def nack(self, lease: _Lease, requeue_delay_s: float = 5) -> None:
        with self._lock:
            l = self._inflight.pop(lease.id, None)
            if l:
                self._ctr += 1
                heapq.heappush(self._ready, (time.time()+requeue_delay_s, self._ctr, l.msg))

from typing import List, Optional
from aethergraph.contracts.services.memory import Event, HotLog
from aethergraph.contracts.services.kv import AsyncKV

def kv_hot_key(run_id: str) -> str: return f"mem:{run_id}:hot"

class KVHotLog(HotLog):
    def __init__(self, kv: AsyncKV):
        self.kv = kv

    async def append(self, run_id: str, evt: Event, *, ttl_s: int, limit: int) -> None:
        key = kv_hot_key(run_id)
        buf = list((await self.kv.get(key, default=[])) or [])
        buf.append(evt.__dict__)  # store as dict for JSON serializability
        if len(buf) > limit: buf = buf[-limit:]
        await self.kv.set(key, buf, ttl_s=ttl_s)

    async def recent(self, run_id: str, *, kinds: Optional[List[str]]=None, limit:int=50) -> List[Event]:
        buf = (await self.kv.get(kv_hot_key(run_id), default=[])) or []
        if kinds: buf = [e for e in buf if e.get("kind") in kinds]
        return [Event(**e) for e in buf[-limit:]]

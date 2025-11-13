from typing import Dict, Any, List, Optional
from aethergraph.contracts.services.memory import Event, Indices
from aethergraph.contracts.services.kv import AsyncKV

def idx_by_ref_kind(run_id:str)->str: return f"mem:{run_id}:idx2:ref_kind"
def idx_by_name(run_id:str)->str:     return f"mem:{run_id}:idx2:name"
def idx_by_topic(run_id:str)->str:    return f"mem:{run_id}:idx2:topic"

class KVIndices(Indices):
    def __init__(self, kv: AsyncKV, hot_ttl_s: int):
        self.kv = kv
        self.ttl = hot_ttl_s

    async def update(self, run_id: str, evt: Event) -> None:
        ts, eid, tool = evt.ts, evt.event_id, (evt.tool or "")
        outs = evt.outputs or []

        by_kind = (await self.kv.get(idx_by_ref_kind(run_id), {})) or {}
        by_name = (await self.kv.get(idx_by_name(run_id), {})) or {}
        by_topic= (await self.kv.get(idx_by_topic(run_id), {})) or {}

        for v in outs:
            nm = v.get("name")
            if not nm: continue
            by_name[nm] = {"ts": ts, "event_id": eid, "vtype": v.get("vtype"), "value": v.get("value")}
            if v.get("vtype") == "ref" and isinstance(v.get("value"), dict):
                kind = v["value"].get("kind"); uri = v["value"].get("uri")
                if kind and uri:
                    lst = by_kind.setdefault(kind, [])
                    lst.append({"ts": ts, "event_id": eid, "name": nm, "uri": uri, "topic": tool})
                    if len(lst) > 200: del lst[:-200]

        if tool:
            last = by_topic.get(tool, {}) or {}
            last["ts"] = ts; last["event_id"] = eid
            last["last_outputs"] = {v["name"]: v.get("value") for v in outs if v.get("name")}
            by_topic[tool] = last

        await self.kv.set(idx_by_ref_kind(run_id), by_kind, ttl_s=self.ttl)
        await self.kv.set(idx_by_name(run_id),     by_name, ttl_s=self.ttl)
        await self.kv.set(idx_by_topic(run_id),    by_topic, ttl_s=self.ttl)

    async def last_by_name(self, run_id: str, name: str) -> Optional[Dict[str, Any]]:
        by_name = await self.kv.get(idx_by_name(run_id), {}) or {}
        return by_name.get(name)

    async def latest_refs_by_kind(self, run_id: str, kind: str, *, limit:int=50) -> List[Dict[str, Any]]:
        by_kind = await self.kv.get(idx_by_ref_kind(run_id), {}) or {}
        return list(reversed((by_kind.get(kind) or [])[-limit:]))

    async def last_outputs_by_topic(self, run_id: str, topic: str) -> Optional[Dict[str, Any]]:
        by_topic = await self.kv.get(idx_by_topic(run_id), {}) or {}
        return by_topic.get(topic)

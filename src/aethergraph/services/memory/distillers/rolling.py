from aethergraph.contracts.services.memory import Distiller, HotLog, Persistence, Indices, Event
from typing import Dict, Any, List, Optional
import time

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def ar_summary_uri(run_id: str, tag: str, ts: str) -> str:
    # Save summaries under the same base "mem/<run_id>/..." tree as append_event,
    # but using a file:// URI so FSPersistence can handle it.
    safe_ts = ts.replace(":", "-")
    return f"file://mem/{run_id}/summaries/{tag}/{safe_ts}.json"

class RollingSummarizer(Distiller):
    def __init__(self, *, max_turns:int=20, min_signal:float=0.25, turn_kinds:Optional[List[str]]=None):
        self.max_turns = max_turns
        self.min_signal = min_signal
        self.turn_kinds = turn_kinds or ["user_msg","assistant_msg"]

    async def distill(self, run_id: str, *, hotlog: HotLog, persistence: Persistence, indices: Indices, **kw) -> Dict[str, Any]:
        turns = await hotlog.recent(run_id, kinds=self.turn_kinds, limit=self.max_turns*2)
        kept = [t for t in turns if (t.signal or 0.0) >= self.min_signal]
        if not kept: return {}

        lines = []
        srcs: List[str] = []
        for t in kept[-self.max_turns:]:
            role = "User" if t.kind=="user_msg" else "Assistant"
            if t.text: 
                lines.append(f"{role}: {t.text}")
                srcs.append(t.event_id)
        digest_text = "\n".join(lines)
        ts = _now_iso()
        summary = {"kind":"rolling_summary","run_id":run_id,"ts":ts,"sources":srcs,"text":digest_text}
        
        uri = ar_summary_uri(run_id,"rolling",ts)

        await persistence.save_json(uri=uri, obj=summary)

        evt = Event(
            event_id="",
            ts=ts,
            run_id=run_id,
            kind="rolling_summary",
            severity=1,
            signal=0.5,
            text=digest_text,
            metrics={"num_turns": len(kept)},
            tags=["summary"],
        )
        
        from aethergraph.services.memory.facade import stable_event_id
        evt.event_id = stable_event_id({"ts":ts,"run_id":run_id,"run_id":run_id,"kind":"rolling_summary","text":digest_text[:200]})
        await hotlog.append(run_id, evt, ttl_s=7*24*3600, limit=1000)
        await persistence.append_event(run_id, evt)
        return {"uri": uri, "sources": srcs}

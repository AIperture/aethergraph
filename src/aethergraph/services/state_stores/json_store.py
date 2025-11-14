import asyncio
import json
import os
import threading
import time

from aethergraph.contracts.services.state_stores import GraphSnapshot, GraphStateStore, StateEvent


class JsonGraphStateStore(GraphStateStore):
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self._alock = asyncio.Lock()
        self._tlock = threading.RLock()

    def _run_dir(self, run_id: str) -> str:
        d = os.path.join(self.root, run_id)
        os.makedirs(d, exist_ok=True)
        return d

    async def save_snapshot(self, snap: GraphSnapshot) -> None:
        d = self._run_dir(snap.run_id)
        ts = int(time.time())
        fn = f"snapshot_{snap.rev:08d}_{ts}.json"
        tmp = os.path.join(d, fn + ".tmp")
        dst = os.path.join(d, fn)
        with self._tlock:  # <— thread-safe region
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snap.__dict__, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)

    async def load_latest_snapshot(self, run_id: str) -> GraphSnapshot | None:
        d = self._run_dir(run_id)
        with self._tlock:
            files = [x for x in os.listdir(d) if x.startswith("snapshot_")]
            if not files:
                return None
            files.sort()
            with open(os.path.join(d, files[-1]), encoding="utf-8") as f:
                return GraphSnapshot(**json.load(f))

    async def append_event(self, ev: StateEvent) -> None:
        p = os.path.join(self._run_dir(ev.run_id), "events.jsonl")
        line = json.dumps(ev.__dict__, ensure_ascii=False) + "\n"
        with self._tlock, open(p, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    async def load_events_since(self, run_id: str, from_rev: int) -> list[StateEvent]:
        p = os.path.join(self._run_dir(run_id), "events.jsonl")
        if not os.path.exists(p):
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["rev"] > from_rev:
                    out.append(StateEvent(**rec))
        return out

    async def list_run_ids(self, graph_id: str | None = None) -> list[str]:
        # best-effort: return all directories; filter by graph_id by reading latest snapshot if needed
        return [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]

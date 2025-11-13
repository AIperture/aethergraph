from __future__ import annotations
import os, json, time, asyncio
from dataclasses import asdict
from typing import Dict
from aethergraph.contracts.services.memory import Persistence, Event

class FSPersistence(Persistence):
    def __init__(self, *, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)

    async def append_event(self, run_id: str, evt: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        rel = os.path.join("mem", run_id, "events", f"{day}.jsonl")
        path = os.path.join(self.base_dir, rel)

        def _write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(evt), ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)

    async def save_json(self, uri: str, obj: Dict[str, any]) -> None:
        assert uri.startswith("file://"), f"FSPersistence only supports file://, got {uri!r}"
        rel = uri[len("file://"):].lstrip("/\\")
        path = os.path.join(self.base_dir, rel)
        def _write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        await asyncio.to_thread(_write)

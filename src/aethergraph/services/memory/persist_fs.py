from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
import time
from typing import Any

from aethergraph.contracts.services.memory import Event, Persistence


class FSPersistence(Persistence):
    def __init__(self, *, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)

    async def append_event(self, run_id: str, evt: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        rel = os.path.join("mem", run_id, "events", f"{day}.jsonl")
        path = os.path.join(self.base_dir, rel)

        def _write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            raw = asdict(evt)
            # Drop None values to keep JSON lean, but retain empty lists/dicts and 0.
            data = {k: v for k, v in raw.items() if v is not None}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)

    async def save_json(self, uri: str, obj: dict[str, Any]) -> None:
        assert uri.startswith("file://"), f"FSPersistence only supports file://, got {uri!r}"
        rel = uri[len("file://") :].lstrip("/\\")
        path = os.path.join(self.base_dir, rel)

        def _write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

    async def load_json(self, uri: str) -> dict[str, Any]:
        """
        Inverse of save_json: load a JSON object from a file:// URI.
        """
        assert uri.startswith("file://"), f"FSPersistence only supports file://, got {uri!r}"
        rel = uri[len("file://") :].lstrip("/\\")
        path = os.path.join(self.base_dir, rel)

        def _read() -> dict[str, Any]:
            with open(path, encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

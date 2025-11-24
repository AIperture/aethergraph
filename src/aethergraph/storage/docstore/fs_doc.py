import asyncio
import json
import os
import threading
from typing import Any

from anyio import Path

from aethergraph.contracts.storage.doc_store import DocStore


class FSDocStore(DocStore):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, doc_id: str) -> Path:
        p = self.root / f"{doc_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    async def put(self, doc_id: str, doc: dict[str, Any]) -> None:
        path = self._path_for(doc_id)

        def _write():
            tmp = path.with_suffix(path.suffix + ".tmp")
            with self._lock, tmp.open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        path = self._path_for(doc_id)

        def _read():
            if not path.exists():
                return None
            with self._lock, path.open("r", encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

    async def delete(self, doc_id):
        path = self._path_for(doc_id)

        def _delete():
            if path.exists():
                with self._lock:
                    path.unlink()

        await asyncio.to_thread(_delete)

    async def list(self) -> list[str]:
        def _list():
            out = []
            for p in self.root.rglob("*.json"):
                rel = p.relative_to(self.root)
                doc_id = str(rel.with_suffix("").as_posix())
                out.append(doc_id)
            return out

        return await asyncio.to_thread(_list)

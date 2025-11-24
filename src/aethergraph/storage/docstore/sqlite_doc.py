from __future__ import annotations

import asyncio
from typing import Any

from aethergraph.contracts.storage.doc_store import DocStore

from .sqlite_doc_sync import SQLiteDocStoreSync


class SqliteDocStore(DocStore):
    """
    Async DocStore implemented on top of SQLiteDocStoreSync via asyncio.to_thread.

    Safe to use from multiple threads (sidecar + main loop) due to RLock in sync core.
    """

    def __init__(self, path: str):
        self._sync = SQLiteDocStoreSync(path)

    async def put(self, doc_id: str, doc: dict[str, Any]) -> None:
        await asyncio.to_thread(self._sync.put, doc_id, doc)

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._sync.get, doc_id)

    async def delete(self, doc_id: str) -> None:
        await asyncio.to_thread(self._sync.delete, doc_id)

    async def list(self) -> list[str]:
        return await asyncio.to_thread(self._sync.list)

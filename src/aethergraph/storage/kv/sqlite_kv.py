from __future__ import annotations

import asyncio
from typing import Any

from aethergraph.contracts.storage.async_kv import AsyncKV

from .sqlite_kv_sync import SQLiteKVSync


class SqliteKV(AsyncKV):
    """
    Async KV on top of SQLiteKVSync via asyncio.to_thread.
    Safe across threads (RLock in sync core).
    """

    def __init__(self, path: str, *, prefix: str = ""):
        self._sync = SQLiteKVSync(path, prefix=prefix)

    async def get(self, key: str, default: Any = None) -> Any:
        return await asyncio.to_thread(self._sync.get, key, default)

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        await asyncio.to_thread(self._sync.set, key, value, ttl_s)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self._sync.delete, key)

    async def mget(self, keys: list[str]) -> list[Any]:
        return await asyncio.to_thread(self._sync.mget, keys)

    async def mset(self, kv: dict[str, Any], *, ttl_s: int | None = None) -> None:
        await asyncio.to_thread(self._sync.mset, kv, ttl_s)

    async def expire(self, key: str, ttl_s: int) -> None:
        await asyncio.to_thread(self._sync.expire, key, ttl_s)

    async def purge_expired(self, limit: int = 1000) -> int:
        return await asyncio.to_thread(self._sync.purge_expired, limit)

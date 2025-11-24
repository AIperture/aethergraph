# storage/kv/layered_kv.py
from __future__ import annotations

from typing import Any

from aethergraph.contracts.storage.async_kv import AsyncKV


class LayeredKV(AsyncKV):
    """
    Read-through / write-through KV:

    - hot: typically InMemoryKV
    - cold: persistent KV (SqliteKV, RedisKV, etc.)
    """

    def __init__(self, hot: AsyncKV, cold: AsyncKV):
        self.hot = hot
        self.cold = cold

    async def get(self, key: str, default: Any = None) -> Any:
        v = await self.hot.get(key, default=None)
        if v is not None:
            return v
        v = await self.cold.get(key, default=default)
        if v is not None:
            await self.hot.set(key, v)
        return v

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        await self.cold.set(key, value, ttl_s=ttl_s)
        await self.hot.set(key, value, ttl_s=ttl_s)

    async def delete(self, key: str) -> None:
        await self.cold.delete(key)
        await self.hot.delete(key)

    async def mget(self, keys: list[str]) -> list[Any]:
        return [await self.get(k) for k in keys]

    async def mset(self, kv: dict[str, Any], *, ttl_s: int | None = None) -> None:
        for k, v in kv.items():
            await self.set(k, v, ttl_s=ttl_s)

    async def expire(self, key: str, ttl_s: int) -> None:
        await self.cold.expire(key, ttl_s)
        await self.hot.expire(key, ttl_s)

    async def purge_expired(self, limit: int = 1000) -> int:
        n_cold = await self.cold.purge_expired(limit)
        n_hot = await self.hot.purge_expired(limit)
        return n_cold + n_hot

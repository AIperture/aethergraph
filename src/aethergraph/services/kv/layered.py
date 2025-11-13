from typing import Any, Optional, List, Dict

class LayeredKV:
    """ Layered KV combining a fast ephemeral cache (e.g. in-memory) with a durable backend (e.g. SQLite, Redis).
        Gets first check the cache, then the durable store; sets write to both.
        List operations invalidate the cache to avoid staleness.
    """
    def __init__(self, cache, durable):
        self.cache = cache      # EphemeralKV
        self.durable = durable  # SQLiteKV / RedisKV

    async def get(self, key: str, default: Any = None) -> Any:
        v = await self.cache.get(key, None)
        if v is not None: return v
        v = await self.durable.get(key, default)
        if v is not None: await self.cache.set(key, v, ttl_s=5)  # short cache
        return v

    async def set(self, key: str, value: Any, *, ttl_s: Optional[int] = None) -> None:
        await self.durable.set(key, value, ttl_s=ttl_s)
        await self.cache.set(key, value, ttl_s=min(ttl_s or 5, 5))

    async def delete(self, key: str) -> None:
        await self.durable.delete(key)
        await self.cache.delete(key)

    async def list_append_unique(self, *a, **k): 
        cur = await self.durable.list_append_unique(*a, **k)
        await self.cache.delete(a[0])  # invalidate
        return cur

    async def list_pop_all(self, key: str) -> list:
        items = await self.durable.list_pop_all(key)
        await self.cache.delete(key)
        return items

    # mget/mset/expire/purge can just forward as needed

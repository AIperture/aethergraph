from __future__ import annotations

import threading
import time
from typing import Any

from aethergraph.contracts.storage.async_kv import AsyncKV


class InMemoryKV(AsyncKV):
    """
    Simple in-memory KV.

    - Process-local, not shared across processes.
    - Thread-safe via RLock (sidecar + main thread can share safely).
    - TTL managed best-effort on access / purge.
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._expires_at: dict[str, float | None] = {}
        self._lock = threading.RLock()

    async def get(self, key: str, default: Any = None) -> Any:
        now = time.time()
        with self._lock:
            if key not in self._data:
                return default
            exp = self._expires_at.get(key)
            if exp is not None and exp < now:
                # expired
                self._data.pop(key, None)
                self._expires_at.pop(key, None)
                return default
            return self._data[key]

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        with self._lock:
            self._data[key] = value
            self._expires_at[key] = time.time() + ttl_s if ttl_s is not None else None

    async def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._expires_at.pop(key, None)

    async def mget(self, keys: list[str]) -> list[Any]:
        # reuse get() so TTL is respected
        return [await self.get(k) for k in keys]

    async def mset(self, kv: dict[str, Any], *, ttl_s: int | None = None) -> None:
        for k, v in kv.items():
            await self.set(k, v, ttl_s=ttl_s)

    async def expire(self, key: str, ttl_s: int) -> None:
        with self._lock:
            if key in self._data:
                self._expires_at[key] = time.time() + ttl_s

    async def purge_expired(self, limit: int = 1000) -> int:
        now = time.time()
        removed = 0
        with self._lock:
            for k in list(self._data.keys()):
                if removed >= limit:
                    break
                exp = self._expires_at.get(k)
                if exp is not None and exp < now:
                    self._data.pop(k, None)
                    self._expires_at.pop(k, None)
                    removed += 1
        return removed

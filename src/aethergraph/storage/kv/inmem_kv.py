from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any

from aethergraph.contracts.storage.async_kv import AsyncKV


@dataclass
class KVEntry:
    value: Any
    expire_at: float | None = None


class InMemoryKV(AsyncKV):
    """
    Simple in-memory KV.

    - Process-local, not shared across processes.
    - Thread-safe via RLock (sidecar + main thread can share safely).
    - TTL managed best-effort on access / purge.
    """

    def __init__(self, *, prefix: str = ""):
        self._data: dict[str, Any] = {}
        self._expires_at: dict[str, float | None] = {}
        self._lock = threading.RLock()
        self._prefix = prefix

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

    # Helper to prefix keys
    def _k(self, k: str) -> str:
        return f"{self._prefix}{k}" if self._prefix else k

    async def list_append_unique(
        self, key: str, items: list[dict], *, id_key: str = "id", ttl_s: int | None = None
    ) -> list[dict]:
        """Append items to a list at `key`, ensuring uniqueness based on `id_key`."""
        k = self._k(key)
        with self._lock:
            cur = list(self._data.get(k, KVEntry([])).value or [])
            seen = {x.get(id_key) for x in cur if isinstance(x, dict)}
            cur.extend([x for x in items if isinstance(x, dict) and x.get(id_key) not in seen])
            self._data[k] = KVEntry(value=cur, expire_at=(time.time() + ttl_s) if ttl_s else None)
            return cur

    async def list_pop_all(self, key: str) -> list:
        """Pop and return all items from the list at `key`."""
        k = self._k(key)
        with self._lock:
            e = self._data.pop(k, None)
            return list(e.value) if e and isinstance(e.value, list) else []

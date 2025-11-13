from __future__ import annotations
import threading, time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

@dataclass
class KVEntry:
    value: Any
    expire_at: Optional[float] = None

class EphemeralKV:
    """Process-local, transient KV (not for blobs)."""
    def __init__(self, *, prefix: str = "") -> None:
        self._data: Dict[str, KVEntry] = {}
        self._lock = threading.RLock()
        self._prefix = prefix

    def _k(self, k: str) -> str:
        return f"{self._prefix}{k}" if self._prefix else k

    async def get(self, key: str, default: Any = None) -> Any:
        k = self._k(key)
        with self._lock:
            e = self._data.get(k)
            if not e: return default
            if e.expire_at and e.expire_at < time.time():
                del self._data[k]; return default
            return e.value

    async def set(self, key: str, value: Any, *, ttl_s: Optional[int] = None) -> None:
        k = self._k(key)
        with self._lock:
            self._data[k] = KVEntry(value=value, expire_at=(time.time()+ttl_s) if ttl_s else None)

    async def delete(self, key: str) -> None:
        k = self._k(key)
        with self._lock:
            self._data.pop(k, None)

    async def list_append_unique(self, key: str, items: List[dict], *, id_key: str = "id", ttl_s: Optional[int] = None) -> List[dict]:
        k = self._k(key)
        with self._lock:
            cur = list(self._data.get(k, KVEntry([])).value or [])
            seen = {x.get(id_key) for x in cur if isinstance(x, dict)}
            cur.extend([x for x in items if isinstance(x, dict) and x.get(id_key) not in seen])
            self._data[k] = KVEntry(value=cur, expire_at=(time.time()+ttl_s) if ttl_s else None)
            return cur

    async def list_pop_all(self, key: str) -> list:
        k = self._k(key)
        with self._lock:
            e = self._data.pop(k, None)
            return list(e.value) if e and isinstance(e.value, list) else []

    # Optional helpers
    async def mget(self, keys: List[str]) -> List[Any]:
        return [await self.get(k) for k in keys]

    async def mset(self, kv: Dict[str, Any], *, ttl_s: Optional[int] = None) -> None:
        for k, v in kv.items():
            await self.set(k, v, ttl_s=ttl_s)

    async def expire(self, key: str, ttl_s: int) -> None:
        k = self._k(key)
        with self._lock:
            e = self._data.get(k)
            if e:
                e.expire_at = time.time() + ttl_s

    async def purge_expired(self, limit: int = 1000) -> int:
        n = 0; now = time.time()
        with self._lock:
            for k in list(self._data.keys()):
                if n >= limit: break
                e = self._data.get(k)
                if e and e.expire_at and e.expire_at < now:
                    self._data.pop(k, None); n += 1
        return n

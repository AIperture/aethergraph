from typing import Any, Protocol

"""
Used for defining the AsyncKV protocol interface.

Typical implementations include:
- EphemeralKV: In-memory, transient key-value store.
- SQLiteKV: Durable key-value store backed by SQLite.
- LayeredKV: Combines a fast ephemeral cache with a durable backend.
- RedisKV: (future) Cloud-based key-value store using Redis.
- Factory function to create KV instances based on environment configuration.

It is used in various parts of the system for transient and durable storage needs.
- context.kv() for general KV storage.
- memory hotlog implementation with KV support 
"""


class AsyncKV(Protocol):
    async def get(self, key: str, default: Any = None) -> Any: ...
    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...

    # Optional
    async def mget(self, keys: list[str]) -> list[Any]: ...  # multiple get
    async def mset(
        self, kv: dict[str, Any], *, ttl_s: int | None = None
    ) -> None: ...  # multiple set
    async def expire(self, key: str, ttl_s: int) -> None: ...
    async def purge_expired(self, limit: int = 1000) -> int: ...  # return number purged

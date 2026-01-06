from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass
class KVEntry:
    value: Any
    expire_at: float | None = None


class EphemeralKV:
    """Process-local, transient KV (not for blobs)."""

    def __init__(self, *, prefix: str = "") -> None:
        self._data: dict[str, KVEntry] = {}
        self._lock = threading.RLock()
        self._prefix = prefix

    def _k(self, k: str) -> str:
        return f"{self._prefix}{k}" if self._prefix else k

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve the value associated with a key from the ephemeral key-value store.

        This method checks for key existence and expiration, automatically removing
        expired entries. If the key does not exist or is expired, the provided
        `default` value is returned.

        Examples:
            Basic usage to fetch a value:
            ```python
            value = await context.kv().get("session_token")
            ```

            Providing a default if the key is missing or expired:
            ```python
            user_id = await context.kv().get("user_id", default=None)
            ```

        Args:
            key: The string key to look up.
            default: The value to return if the key is not found or has expired.

        Returns:
            The stored value if present and not expired; otherwise, the `default`.
        """
        k = self._k(key)
        with self._lock:
            e = self._data.get(k)
            if not e:
                return default
            if e.expire_at and e.expire_at < time.time():
                del self._data[k]
                return default
            return e.value

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        """
        Store a value in the ephemeral key-value store with optional expiration.

        This method inserts or updates the value for a given key, optionally setting
        a time-to-live (TTL) in seconds. If a TTL is provided, the entry will expire
        and be automatically removed after the specified duration.

        Examples:
            Basic usage to store a value:
            ```python
            await context.kv().set("session_token", "abc123")
            ```

            Storing a value with a 10-minute expiration:
            ```python
            await context.kv().set("user_id", 42, ttl_s=600)
            ```

        Args:
            key: The string key under which to store the value.
            value: The value to store (any serializable object).
            ttl_s: Optional expiration time in seconds. If None, the value does not expire.

        Returns:
            None
        """
        k = self._k(key)
        with self._lock:
            self._data[k] = KVEntry(value=value, expire_at=(time.time() + ttl_s) if ttl_s else None)

    async def delete(self, key: str) -> None:
        """
        Remove a key and its associated value from the ephemeral key-value store.

        This method deletes the specified key from the store if it exists. If the key
        does not exist, the operation is a no-op and does not raise an error.

        Examples:
            Basic usage to delete a key:
            ```python
            await context.kv().delete("session_token")
            ```

            Deleting a user-specific cache entry:
            ```python
            await context.kv().delete(f"user_cache:{user_id}")
            ```

        Args:
            key: The string key to remove from the store.

        Returns:
            None
        """
        k = self._k(key)
        with self._lock:
            self._data.pop(k, None)

    async def list_append_unique(
        self, key: str, items: list[dict], *, id_key: str = "id", ttl_s: int | None = None
    ) -> list[dict]:
        """
        Append unique dictionary items to a list stored in the ephemeral key-value store.

        This method ensures that only items with unique `id_key` values are added to the list
        associated with the given key. If the key does not exist, a new list is created.
        Optionally, a time-to-live (TTL) can be set for the entry.

        Examples:
            Basic usage to append unique items:
            ```python
            await context.kv().list_append_unique("recent_users", [{"id": 1, "name": "Alice"}])
            ```

            Appending multiple items with a custom ID key and expiration:
            ```python
            await context.kv().list_append_unique(
                "tasks",
                [{"task_id": 42, "desc": "Review PR"}],
                id_key="task_id",
                ttl_s=3600
            )
            ```

        Args:
            key: The string key under which the list is stored.
            items: A list of dictionaries to append. Only items with unique `id_key` values
                (not already present in the list) will be added.
            id_key: The dictionary key used to determine uniqueness (default: `"id"`).
            ttl_s: Optional expiration time in seconds for the updated list. If None, the list does not expire.

        Returns:
            list[dict]: The updated list of dictionaries after appending unique items.

        Notes:
            - This method is used for lists of dictionaries where each dictionary has a unique identifier. For example,
                it can be used to maintain a list of recent user actions, ensuring no duplicates based on user ID.
            - Example of the stored list structure:
                ```python
                [
                    {"id": 1, "name": "Alice"},
                    {"id": 2, "name": "Bob"},
                    ...
                ]
                ```

        """
        k = self._k(key)
        with self._lock:
            cur = list(self._data.get(k, KVEntry([])).value or [])
            seen = {x.get(id_key) for x in cur if isinstance(x, dict)}
            cur.extend([x for x in items if isinstance(x, dict) and x.get(id_key) not in seen])
            self._data[k] = KVEntry(value=cur, expire_at=(time.time() + ttl_s) if ttl_s else None)
            return cur

    async def list_pop_all(self, key: str) -> list:
        """
        Atomically remove and return all items from a list stored in the ephemeral key-value store.

        This method retrieves the entire list associated with the given key and removes the key from the store.
        If the key does not exist or does not contain a list, an empty list is returned. This operation is atomic
        and ensures no items are left behind after the call.

        Examples:
            Basic usage to pop all items from a list:
            ```python
            items = await context.kv().list_pop_all("recent_events")
            ```

            Handling the case where the key may not exist:
            ```python
            logs = await context.kv().list_pop_all("logs")  # returns [] if "logs" is missing
            ```

        Args:
            key: The string key under which the list is stored.

        Returns:
            list: The list of items that were stored under the key, or an empty list if the key was not found.
        """
        k = self._k(key)
        with self._lock:
            e = self._data.pop(k, None)
            return list(e.value) if e and isinstance(e.value, list) else []

    # Optional helpers
    async def mget(self, keys: list[str]) -> list[Any]:
        """
        Retrieve multiple values from the ephemeral key-value store in a single call.

        This method fetches the values associated with each key in the provided list,
        preserving the order of the input. If a key does not exist or is expired, `None`
        is returned in its place.

        Examples:
            Basic usage to fetch several values:
            ```python
            values = await context.kv().mget(["user_id", "session_token", "profile"])
            ```

            Handling missing or expired keys:
            ```python
            results = await context.kv().mget(["foo", "bar"])
            # results might be [None, "bar_value"] if "foo" is missing or expired
            ```

        Args:
            keys: A list of string keys to retrieve from the store.

        Returns:
            list[Any]: A list of values corresponding to the input keys. If a key is not found
                or has expired, its position in the list will be `None`.
        """
        return [await self.get(k) for k in keys]

    async def mset(self, kv: dict[str, Any], *, ttl_s: int | None = None) -> None:
        """
        Set multiple key-value pairs in the ephemeral key-value store.

        This asynchronous method iterates over the provided dictionary and sets each key-value pair
        in the store, optionally applying a time-to-live (TTL) to each entry. If a TTL is specified,
        each key will expire after the given number of seconds.

        Examples:
            Basic usage to set multiple values:
            ```python
            await context.kv().mset({"foo": 1, "bar": "baz"})
            ```

            Setting multiple values with a TTL of 60 seconds:
            ```python
            await context.kv().mset({"session": "abc", "count": 42}, ttl_s=60)
            ```

        Args:
            kv: A dictionary mapping string keys to values to be stored.
            ttl_s: Optional; the time-to-live for each key in seconds. If None, keys do not expire.

        Returns:
            None
        """
        for k, v in kv.items():
            await self.set(k, v, ttl_s=ttl_s)

    async def expire(self, key: str, ttl_s: int) -> None:
        """
        Set or update the expiration time (TTL) for a key in the ephemeral key-value store.

        This method updates the expiration timestamp for an existing key, causing it to expire
        and be automatically removed after the specified number of seconds. If the key does not
        exist, this operation is a no-op.

        Examples:
            Basic usage to set a 5-minute expiration:
            ```python
            await context.kv().expire("session_token", ttl_s=300)
            ```

            Updating the TTL for a cached user profile:
            ```python
            await context.kv().expire("user_profile:42", ttl_s=60)
            ```

        Args:
            key: The string key whose expiration time should be set or updated.
            ttl_s: The time-to-live in seconds from now. After this duration, the key will expire.

        Returns:
            None
        """
        k = self._k(key)
        with self._lock:
            e = self._data.get(k)
            if e:
                e.expire_at = time.time() + ttl_s

    async def purge_expired(self, limit: int = 1000) -> int:
        """
        Remove expired key-value entries from the ephemeral store.

        This method scans the internal data store for entries whose expiration
        timestamp has passed and removes them, up to the specified limit. It is
        intended to be called periodically to keep the store clean and efficient.

        Examples:
            Purge up to 1000 expired entries:
            ```python
            removed = await context.kv().purge_expired()
            ```

            Purge a custom number of expired entries:
            ```python
            removed = await context.kv().purge_expired(limit=500)
            ```

        Args:
            limit: The maximum number of expired entries to remove in a single call.

        Returns:
            int: The number of expired entries that were removed from the store.
        """
        n = 0
        now = time.time()
        with self._lock:
            for k in list(self._data.keys()):
                if n >= limit:
                    break
                e = self._data.get(k)
                if e and e.expire_at and e.expire_at < now:
                    self._data.pop(k, None)
                    n += 1
        return n

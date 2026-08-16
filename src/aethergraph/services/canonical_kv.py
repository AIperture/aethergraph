"""Inactive canonical general key-value facade for the S9 cut."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any

from aethergraph.services.canonical_storage_scope import validate_storage_owner_scope
from aethergraph.storage.contracts import (
    FrozenJson,
    KeyValueQuery,
    KeyValueRecord,
    KeyValueStore,
    PageRequest,
    StorageBundle,
    StorageCapacityError,
    StorageConflictError,
    StorageIntegrityError,
    StorageScope,
)

_MAX_CAS_ATTEMPTS = 32
_MAX_SCAN_KEYS = 1_000


class CanonicalKeyValueFacade:
    """Project the public async KV surface onto one canonical repository."""

    def __init__(
        self,
        *,
        repository: KeyValueStore,
        owner_scope: StorageScope,
        namespace: str,
        clock: Callable[[], datetime],
    ) -> None:
        """Bind general KV operations to one scope and namespace.

        Intro:
            Captures an exact provider repository and trusted owner scope without
            selecting storage, opening files, or creating a local cache authority.

        Examples:
            Bind the runtime KV namespace:
            ```python
            kv = CanonicalKeyValueFacade(
                repository=bundle.kv,
                owner_scope=owner_scope,
                namespace="runtime.kv",
                clock=clock,
            )
            ```

            Bind a deterministic test namespace:
            ```python
            kv = CanonicalKeyValueFacade(
                repository=fake_kv,
                owner_scope=StorageScope(project_id="project-1"),
                namespace="tests.kv",
                clock=lambda: fixed_now,
            )
            ```

        Args:
            repository: Exact canonical general KV repository.
            owner_scope: Trusted provider ownership scope.
            namespace: Exact fixed logical namespace for every operation.
            clock: Timezone-aware UTC source used to translate relative TTLs.

        Returns:
            None: The inactive-until-S9 facade is ready.

        Notes:
            The bundle owns lifecycle. Keys cannot select another namespace, scope,
            provider, database, or fallback implementation.
        """
        validate_storage_owner_scope(owner_scope)
        _nonempty("namespace", namespace)
        _utc(clock(), field="clock")
        self._repository = repository
        self._owner_scope = owner_scope
        self._namespace = namespace
        self._clock = clock

    async def get(self, key: str, default: Any = None) -> Any:
        """Read one exact current value or return the caller default.

        Intro:
            Delegates TTL visibility and exact scoped identity to the provider while
            returning a detached immutable JSON-compatible value.

        Examples:
            Read an existing value:
            ```python
            value = await kv.get("settings")
            ```

            Supply a default for absence:
            ```python
            value = await kv.get("missing", {"enabled": False})
            ```

        Args:
            key: Exact key inside the bound namespace.
            default: Value returned when the provider record is absent or expired.

        Returns:
            Any: Canonical stored JSON value or the caller-provided default.

        Notes:
            No cache, prefix transformation, or alternate store is consulted.
        """
        record = await self._repository.get(
            self._owner_scope,
            self._namespace,
            _nonempty("key", key),
        )
        return record.value if record is not None else default

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        """Create or replace one value through bounded revision-CAS retry.

        Intro:
            Preserves unconditional public KV semantics while using the provider's
            current revision as the only concurrency authority.

        Examples:
            Store a durable value:
            ```python
            await kv.set("settings", {"enabled": True})
            ```

            Store a value with relative TTL:
            ```python
            await kv.set("lease", {"owner": "worker-1"}, ttl_s=30)
            ```

        Args:
            key: Exact key inside the bound namespace.
            value: Complete JSON-compatible next value.
            ttl_s: Optional positive lifetime in seconds.

        Returns:
            None: The provider committed the replacement.

        Notes:
            Exhausted conflicts propagate; there is no blind overwrite or fallback.
        """
        await self._replace(
            key=_nonempty("key", key),
            value=value,
            expires_at=self._expires_at(ttl_s),
        )

    async def delete(self, key: str) -> None:
        """Delete one value through bounded exact revision-CAS retry.

        Intro:
            Resolves and deletes only the current provider revision, preserving the
            public idempotent behavior when a key is absent or expired.

        Examples:
            Delete an existing value:
            ```python
            await kv.delete("settings")
            ```

            Delete an absent value safely:
            ```python
            await kv.delete("missing")
            ```

        Args:
            key: Exact key inside the bound namespace.

        Returns:
            None: The current value was deleted or already absent.

        Notes:
            A racing update is never silently deleted without an exact reread.
        """
        await self._delete(_nonempty("key", key))

    async def mget(self, keys: list[str]) -> list[Any]:
        """Read a bounded ordered collection of exact keys.

        Intro:
            Preserves input order and duplicate positions through independent exact
            provider reads rather than broad prefix enumeration.

        Examples:
            Read two values:
            ```python
            values = await kv.mget(["a", "b"])
            ```

            Preserve a missing position:
            ```python
            assert await kv.mget(["missing"]) == [None]
            ```

        Args:
            keys: Ordered exact keys, bounded to the provider page-size ceiling.

        Returns:
            list[Any]: Values in input order, using `None` for absence.

        Notes:
            This is not a scan and does not weaken scope authorization.
        """
        _bounded_count("keys", keys)
        return [await self.get(key) for key in keys]

    async def mset(self, kv: dict[str, Any], *, ttl_s: int | None = None) -> None:
        """Replace a bounded mapping of exact keys.

        Intro:
            Applies the public best-effort batch contract as ordered independent CAS
            writes; the canonical repository does not advertise multi-key atomicity.

        Examples:
            Store two durable values:
            ```python
            await store.mset({"a": 1, "b": 2})
            ```

            Store a short-lived batch:
            ```python
            await store.mset({"a": 1, "b": 2}, ttl_s=60)
            ```

        Args:
            kv: Exact key-to-JSON-value mapping, bounded to 1,000 entries.
            ttl_s: Optional positive lifetime shared by all committed values.

        Returns:
            None: Every ordered replacement completed.

        Notes:
            Earlier keys may be committed if a later key fails. No rollback or
            fallback is claimed by this compatibility operation.
        """
        _bounded_count("kv", kv)
        for key, value in kv.items():
            await self.set(key, value, ttl_s=ttl_s)

    async def incr(self, key: str, amount: int = 1, *, ttl_s: int | None = None) -> int:
        """Atomically increment one integer value through revision CAS.

        Intro:
            Treats absence as zero and retries only the exact key when another writer
            advances the provider revision first.

        Examples:
            Increment a counter by one:
            ```python
            value = await kv.incr("attempts")
            ```

            Add a bounded amount with TTL:
            ```python
            value = await kv.incr("rate", amount=5, ttl_s=60)
            ```

        Args:
            key: Exact key inside the bound namespace.
            amount: Integer delta; booleans are rejected.
            ttl_s: Optional positive lifetime for the new revision.

        Returns:
            int: Provider-committed counter value.

        Notes:
            Existing non-integer values fail directly as integrity errors.
        """
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise TypeError("amount must be an integer")

        def increment(current: FrozenJson | None) -> int:
            if current is None:
                return amount
            if isinstance(current, bool) or not isinstance(current, int):
                raise StorageIntegrityError("KV increment requires an integer value")
            return current + amount

        record = await self._mutate(
            key=_nonempty("key", key),
            transform=increment,
            expires_at=self._expires_at(ttl_s),
        )
        if isinstance(record.value, bool) or not isinstance(record.value, int):
            raise StorageIntegrityError("Provider returned a non-integer KV increment")
        return record.value

    async def exists(self, key: str) -> bool:
        """Report whether one exact unexpired provider value exists.

        Intro:
            Uses the canonical exact lookup so provider TTL visibility and bound
            scope semantics remain identical to `get`.

        Examples:
            Check an existing key:
            ```python
            present = await kv.exists("settings")
            ```

            Check an expired key:
            ```python
            assert await kv.exists("expired") is False
            ```

        Args:
            key: Exact key inside the bound namespace.

        Returns:
            bool: `True` only when a current unexpired provider record exists.

        Notes:
            Stored JSON `null` still counts as an existing record.
        """
        record = await self._repository.get(
            self._owner_scope,
            self._namespace,
            _nonempty("key", key),
        )
        return record is not None

    async def expire(self, key: str, ttl_s: int) -> bool:
        """Assign a new relative TTL to one exact existing value.

        Intro:
            Recommits the unchanged JSON value at the next provider revision with an
            exact UTC expiry derived from the bound clock.

        Examples:
            Expire an existing value:
            ```python
            changed = await kv.expire("lease", 30)
            ```

            Detect an absent value:
            ```python
            assert await kv.expire("missing", 30) is False
            ```

        Args:
            key: Exact key inside the bound namespace.
            ttl_s: Positive lifetime in seconds.

        Returns:
            bool: `True` when an existing value received the TTL; otherwise `False`.

        Notes:
            Expiration is a revisioned provider write, not cache-local metadata.
        """
        expires_at = self._expires_at(ttl_s)
        exact_key = _nonempty("key", key)
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await self._repository.get(
                self._owner_scope,
                self._namespace,
                exact_key,
            )
            if current is None:
                return False
            try:
                await self._repository.compare_and_set(
                    self._owner_scope,
                    self._namespace,
                    exact_key,
                    current.revision,
                    current.value,
                    expires_at,
                )
            except StorageConflictError:
                continue
            return True
        raise StorageConflictError("KV expiration exceeded bounded CAS retries")

    async def list_append_unique(
        self,
        key: str,
        items: list[dict[str, Any]],
        *,
        id_key: str = "id",
        ttl_s: int | None = None,
    ) -> list[dict[str, Any]]:
        """Atomically append mapping items unique by one identity field.

        Intro:
            Rebuilds and commits the complete list with revision CAS so concurrent
            appenders cannot overwrite one another's accepted items.

        Examples:
            Append unique artifact descriptors:
            ```python
            values = await kv.list_append_unique("artifacts", [{"id": "a"}])
            ```

            Use a custom identity field and TTL:
            ```python
            values = await kv.list_append_unique(
                "jobs", [{"name": "build"}], id_key="name", ttl_s=60
            )
            ```

        Args:
            key: Exact list key inside the bound namespace.
            items: Bounded mappings to append.
            id_key: Mapping field used to suppress duplicate identities.
            ttl_s: Optional positive lifetime for the committed list revision.

        Returns:
            list[dict[str, Any]]: Complete detached provider-committed list.

        Notes:
            Existing non-list or non-mapping content fails as storage corruption.
        """
        _bounded_count("items", items)
        _nonempty("id_key", id_key)

        def append(current: FrozenJson | None) -> list[dict[str, Any]]:
            values = _mapping_list(current)
            seen = {item.get(id_key) for item in values}
            for item in items:
                if not isinstance(item, dict):
                    raise TypeError("items must contain mappings")
                identity = item.get(id_key)
                if identity in seen:
                    continue
                values.append(dict(item))
                seen.add(identity)
            return values

        record = await self._mutate(
            key=_nonempty("key", key),
            transform=append,
            expires_at=self._expires_at(ttl_s),
        )
        return _mapping_list(record.value)

    async def list_pop_all(self, key: str) -> list[Any]:
        """Atomically remove and return the complete current list value.

        Intro:
            Reads and deletes one exact provider revision so concurrent replacement
            is retried instead of returning content that remains stored.

        Examples:
            Drain a channel inbox:
            ```python
            items = await kv.list_pop_all("inbox://ui:session")
            ```

            Drain an absent inbox:
            ```python
            assert await kv.list_pop_all("missing") == []
            ```

        Args:
            key: Exact list key inside the bound namespace.

        Returns:
            list[Any]: Detached values removed from the provider, or an empty list.

        Notes:
            Non-list persisted content is deleted but reported as an integrity error;
            no malformed inbox value remains authoritative.
        """
        exact_key = _nonempty("key", key)
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await self._repository.get(
                self._owner_scope,
                self._namespace,
                exact_key,
            )
            if current is None:
                return []
            try:
                values = _list(current.value)
            except StorageIntegrityError:
                await self._delete_revision(exact_key, current.revision)
                raise
            try:
                await self._repository.delete(
                    self._owner_scope,
                    self._namespace,
                    exact_key,
                    current.revision,
                )
            except StorageConflictError:
                continue
            return values
        raise StorageConflictError("KV list pop exceeded bounded CAS retries")

    async def scan_prefix(self, prefix: str, limit: int = 1_000) -> list[str]:
        """Return one bounded provider-cursor page of matching keys.

        Intro:
            Applies the exact prefix inside the provider query before its bound and
            returns only key identities from the first stable page.

        Examples:
            List up to 100 inbox keys:
            ```python
            keys = await kv.scan_prefix("inbox://", limit=100)
            ```

            Request the default bounded page:
            ```python
            keys = await kv.scan_prefix("cache:")
            ```

        Args:
            prefix: Non-empty exact key prefix.
            limit: Maximum keys to return, between one and 1,000.

        Returns:
            list[str]: Stable key-ordered first page.

        Notes:
            This compatibility method intentionally does not hide pagination by
            fetching subsequent pages.
        """
        page = await self._repository.scan(
            KeyValueQuery(
                scope=self._owner_scope,
                namespace=self._namespace,
                key_prefix=_nonempty("prefix", prefix),
                page=PageRequest(limit=limit),
            )
        )
        return [record.key for record in page.items]

    async def scan_keys(self, prefix: str) -> list[str]:
        """Return all matching keys only within the explicit safety ceiling.

        Intro:
            Preserves the older debug helper while failing if one provider page
            cannot represent the complete result.

        Examples:
            Enumerate a small test prefix:
            ```python
            keys = await kv.scan_keys("test:")
            ```

            Detect an oversized administrative prefix:
            ```python
            keys = await kv.scan_keys("runtime:")
            ```

        Args:
            prefix: Non-empty exact key prefix.

        Returns:
            list[str]: Complete key list when it fits the 1,000-key ceiling.

        Notes:
            A continuation cursor raises `StorageCapacityError`; the method never
            performs an unbounded page loop or silently truncates.
        """
        page = await self._repository.scan(
            KeyValueQuery(
                scope=self._owner_scope,
                namespace=self._namespace,
                key_prefix=_nonempty("prefix", prefix),
                page=PageRequest(limit=_MAX_SCAN_KEYS),
            )
        )
        if page.next_cursor is not None:
            raise StorageCapacityError(
                f"KV prefix exceeds the {_MAX_SCAN_KEYS}-key compatibility ceiling"
            )
        return [record.key for record in page.items]

    async def purge_expired(self, limit: int = 1_000) -> int:
        """Physically purge one bounded expired-value maintenance batch.

        Intro:
            Delegates exact scope, namespace, clock, and deletion ordering to the
            canonical repository instead of opening provider-private storage.

        Examples:
            Purge the default batch:
            ```python
            removed = await kv.purge_expired()
            ```

            Purge a smaller maintenance batch:
            ```python
            removed = await kv.purge_expired(limit=100)
            ```

        Args:
            limit: Maximum expired provider records to remove.

        Returns:
            int: Number of physically removed records.

        Notes:
            Normal reads already hide expired values; this is explicit bounded
            provider maintenance with no cross-namespace sweep.
        """
        return await self._repository.purge_expired(
            self._owner_scope,
            self._namespace,
            limit,
        )

    async def _replace(
        self,
        *,
        key: str,
        value: FrozenJson,
        expires_at: datetime | None,
    ) -> KeyValueRecord:
        return await self._mutate(
            key=key,
            transform=lambda _current: value,
            expires_at=expires_at,
        )

    async def _mutate(
        self,
        *,
        key: str,
        transform: Callable[[FrozenJson | None], FrozenJson],
        expires_at: datetime | None,
    ) -> KeyValueRecord:
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await self._repository.get(
                self._owner_scope,
                self._namespace,
                key,
            )
            value = transform(current.value if current is not None else None)
            try:
                return await self._repository.compare_and_set(
                    self._owner_scope,
                    self._namespace,
                    key,
                    current.revision if current is not None else 0,
                    value,
                    expires_at,
                )
            except StorageConflictError:
                continue
        raise StorageConflictError("KV mutation exceeded bounded CAS retries")

    async def _delete(self, key: str) -> bool:
        for _attempt in range(_MAX_CAS_ATTEMPTS):
            current = await self._repository.get(
                self._owner_scope,
                self._namespace,
                key,
            )
            if current is None:
                return False
            try:
                return await self._repository.delete(
                    self._owner_scope,
                    self._namespace,
                    key,
                    current.revision,
                )
            except StorageConflictError:
                continue
        raise StorageConflictError("KV deletion exceeded bounded CAS retries")

    async def _delete_revision(self, key: str, revision: int) -> None:
        try:
            await self._repository.delete(
                self._owner_scope,
                self._namespace,
                key,
                revision,
            )
        except StorageConflictError:
            return

    def _expires_at(self, ttl_s: int | None) -> datetime | None:
        if ttl_s is None:
            return None
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, int) or ttl_s < 1:
            raise ValueError("ttl_s must be a positive integer when supplied")
        now = _utc(self._clock(), field="clock")
        return now + timedelta(seconds=ttl_s)


def bind_canonical_key_value_facade(
    *,
    bundle: StorageBundle,
    owner_scope: StorageScope,
    clock: Callable[[], datetime],
    namespace: str = "runtime.kv",
) -> CanonicalKeyValueFacade:
    """Bind the general KV facade to the exact bundle repository.

    Intro:
        Constructs one inactive scoped facade without provider selection, local path
        resolution, cache layering, I/O, or lifecycle transfer.

    Examples:
        Bind the production runtime namespace:
        ```python
        kv = bind_canonical_key_value_facade(
            bundle=bundle, owner_scope=owner_scope, clock=clock
        )
        ```

        Bind an explicit test namespace:
        ```python
        kv = bind_canonical_key_value_facade(
            bundle=fake_bundle,
            owner_scope=test_scope,
            clock=lambda: fixed_now,
            namespace="tests.kv",
        )
        ```

    Args:
        bundle: One coherent already-open canonical storage bundle.
        owner_scope: Trusted provider ownership scope.
        clock: Timezone-aware UTC source used for relative TTLs.
        namespace: Exact fixed logical namespace for the facade.

    Returns:
        CanonicalKeyValueFacade: Exact provider-backed general KV projection.

    Notes:
        Binding does not activate runtime composition or own bundle close.
    """
    return CanonicalKeyValueFacade(
        repository=bundle.kv,
        owner_scope=owner_scope,
        namespace=namespace,
        clock=clock,
    )


def _mapping_list(value: FrozenJson | None) -> list[dict[str, Any]]:
    items = _list(value)
    if not all(isinstance(item, Mapping) for item in items):
        raise StorageIntegrityError("KV mapping-list operation requires mapping items")
    return [dict(item) for item in items]


def _list(value: FrozenJson | None) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise StorageIntegrityError("KV list operation requires a list value")
    return list(value)


def _bounded_count(field: str, values: Sequence[object] | Mapping[object, object]) -> None:
    if len(values) > _MAX_SCAN_KEYS:
        raise StorageCapacityError(f"{field} exceeds {_MAX_SCAN_KEYS} entries")


def _nonempty(field: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return value

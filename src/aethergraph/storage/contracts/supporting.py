"""Canonical revisioned KV and document contracts for supporting stores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from .pagination import Page, PageRequest
from .records import FrozenJson, _freeze_json, _freeze_mapping, _nonempty, _utc
from .scope import StorageScope


@dataclass(frozen=True, slots=True, kw_only=True)
class KeyValueRecord:
    """Current revisioned JSON value in one exact provider-owned namespace."""

    namespace: str
    key: str
    value: FrozenJson
    revision: int
    scope: StorageScope
    updated_at: datetime
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        _nonempty("namespace", self.namespace)
        _nonempty("key", self.key)
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        _utc("updated_at", self.updated_at)
        if self.expires_at is not None:
            _utc("expires_at", self.expires_at)
            if self.expires_at <= self.updated_at:
                raise ValueError("expires_at must be after updated_at")
        object.__setattr__(self, "value", _freeze_json(self.value))


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentRecord:
    """Current revisioned canonical JSON document and its provider metadata."""

    namespace: str
    document_id: str
    document: Mapping[str, FrozenJson]
    revision: int
    scope: StorageScope
    updated_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty("namespace", self.namespace)
        _nonempty("document_id", self.document_id)
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if isinstance(self.schema_version, bool) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        _utc("updated_at", self.updated_at)
        object.__setattr__(
            self,
            "document",
            _freeze_mapping(self.document, path="document"),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class KeyValueQuery:
    """Bounded namespace/prefix scan for administrative supporting-store use."""

    scope: StorageScope
    namespace: str
    page: PageRequest = PageRequest()
    key_prefix: str | None = None

    def __post_init__(self) -> None:
        _nonempty("namespace", self.namespace)
        if self.key_prefix is not None:
            _nonempty("key_prefix", self.key_prefix)


@dataclass(frozen=True, slots=True, kw_only=True)
class DocumentQuery:
    """Bounded namespace/prefix query for typed supporting documents."""

    scope: StorageScope
    namespace: str
    page: PageRequest = PageRequest()
    id_prefix: str | None = None
    metadata: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty("namespace", self.namespace)
        if self.id_prefix is not None:
            _nonempty("id_prefix", self.id_prefix)
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


class KeyValueStore(Protocol):
    """Revisioned scoped JSON KV store with explicit TTL and bounded scans."""

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
    ) -> KeyValueRecord | None:
        """Read one current KV record in exact canonical scope.

        Expired values behave as absent. Providers may purge them during explicit
        maintenance but do not expose stale content.

        Examples:
            Read an auth grant:
                ```python
                grant = await store.get(scope, "auth.grants", grant_id)
                ```

            Detect an absent key:
                ```python
                assert await store.get(scope, "runtime", "missing") is None
                ```

        Args:
            scope: Canonical owner scope constraining access.
            namespace: Exact provider-owned logical namespace.
            key: Exact key within the namespace.

        Returns:
            KeyValueRecord | None: Current unexpired record or `None` when absent.

        Notes:
            Namespaces replace direct SQLite prefixes and physical database selection.
        """
        ...

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
        value: FrozenJson,
        expires_at: datetime | None = None,
    ) -> KeyValueRecord:
        """Atomically create or advance one revisioned KV value.

        Revision zero requires absence. TTL is an exact UTC expiration owned by the
        provider clock supplied at bundle open.

        Examples:
            Create an invite:
                ```python
                row = await store.compare_and_set(scope, "auth.invites", key, 0, value)
                ```

            Advance a grant:
                ```python
                row = await store.compare_and_set(scope, "auth.grants", key, 1, value, expiry)
                ```

        Args:
            scope: Canonical owner scope constraining the value.
            namespace: Exact provider-owned logical namespace.
            key: Exact key within the namespace.
            expected_revision: Current revision required, or zero for creation.
            value: Complete immutable JSON-compatible next value.
            expires_at: Optional timezone-aware UTC expiration.

        Returns:
            KeyValueRecord: Newly committed value at the next revision.

        Notes:
            Stale writes raise `StorageConflictError`; there is no unconditional
            overwrite compatibility path.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
    ) -> bool:
        """Delete one KV value only at its exact current revision.

        The operation is idempotent for an absent value when revision zero is
        expected and conflict-safe for present values.

        Examples:
            Delete a consumed invite:
                ```python
                deleted = await store.delete(scope, "auth.invites", key, revision)
                ```

            Confirm absence:
                ```python
                assert await store.delete(scope, "runtime", "missing", 0) is False
                ```

        Args:
            scope: Canonical owner scope constraining deletion.
            namespace: Exact provider-owned logical namespace.
            key: Exact key within the namespace.
            expected_revision: Revision that must still be current.

        Returns:
            bool: `True` when deleted; `False` for absence at expected revision zero.

        Notes:
            A mismatched present revision raises `StorageConflictError`.
        """
        ...

    async def scan(self, query: KeyValueQuery) -> Page[KeyValueRecord]:
        """Scan a bounded stable cursor page within one namespace.

        This operation supports administrative and indexed supporting workflows. Hot
        request paths should prefer exact keys.

        Examples:
            Scan a namespace:
                ```python
                page = await store.scan(KeyValueQuery(scope=scope, namespace="auth.grants"))
                ```

            Continue a prefix scan:
                ```python
                page = await store.scan(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact canonical scope, namespace, optional prefix, and page request.

        Returns:
            Page[KeyValueRecord]: Current unexpired records and continuation cursor.

        Notes:
            Unbounded `scan_keys` and method probing are not part of this protocol.
        """
        ...

    async def purge_expired(
        self,
        scope: StorageScope,
        namespace: str,
        limit: int,
    ) -> int:
        """Physically purge a bounded number of expired values.

        Intro:
            Performs explicit maintenance inside one exact canonical scope and
            namespace after normal reads have already hidden expired values.

        Examples:
            Purge one maintenance batch:
            ```python
            removed = await store.purge_expired(scope, "runtime.kv", 100)
            ```

            Drain another bounded namespace independently:
            ```python
            removed = await store.purge_expired(scope, "auth.invites", 25)
            ```

        Args:
            scope: Canonical owner scope constraining physical deletion.
            namespace: Exact provider-owned logical namespace.
            limit: Maximum expired records to remove in this operation.

        Returns:
            int: Number of expired records physically removed.

        Notes:
            Providers use their authoritative clock. This is never an unbounded
            vacuum, cross-scope sweep, or service-owned database operation.
        """
        ...


class DocumentStore(Protocol):
    """Revisioned scoped JSON document repository with stable cursor queries."""

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        document_id: str,
    ) -> DocumentRecord | None:
        """Read one current canonical document.

        The provider performs an exact scoped identity lookup without scanning
        document identifiers.

        Examples:
            Read a registry manifest:
                ```python
                manifest = await store.get(scope, "registry", manifest_id)
                ```

            Detect an absent document:
                ```python
                assert await store.get(scope, "registry", "missing") is None
                ```

        Args:
            scope: Canonical owner scope constraining access.
            namespace: Exact provider-owned document namespace.
            document_id: Exact stable document identifier.

        Returns:
            DocumentRecord | None: Current document or `None` when absent.

        Notes:
            Physical document paths and provider row formats remain private.
        """
        ...

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        document_id: str,
        expected_revision: int,
        document: Mapping[str, FrozenJson],
        schema_version: int,
    ) -> DocumentRecord:
        """Atomically create or advance one canonical document revision.

        Schema version is stored with the document and validated by the owning
        service rather than inferred from provider-private layout.

        Examples:
            Create a manifest:
                ```python
                row = await store.compare_and_set(scope, "registry", key, 0, document, 1)
                ```

            Advance a manifest:
                ```python
                row = await store.compare_and_set(scope, "registry", key, 2, document, 1)
                ```

        Args:
            scope: Canonical owner scope constraining the document.
            namespace: Exact provider-owned document namespace.
            document_id: Exact stable document identifier.
            expected_revision: Current revision required, or zero for creation.
            document: Complete immutable JSON document value.
            schema_version: Positive owning-record schema version.

        Returns:
            DocumentRecord: Newly committed document at the next revision.

        Notes:
            Stale writes raise `StorageConflictError`; unconditional upsert is absent.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        document_id: str,
        expected_revision: int,
    ) -> bool:
        """Delete one document only at its exact current revision.

        The operation cannot remove a newer concurrent document revision.

        Examples:
            Delete a manifest:
                ```python
                deleted = await store.delete(scope, "registry", key, revision)
                ```

            Confirm absence:
                ```python
                assert await store.delete(scope, "registry", "missing", 0) is False
                ```

        Args:
            scope: Canonical owner scope constraining deletion.
            namespace: Exact provider-owned document namespace.
            document_id: Exact stable document identifier.
            expected_revision: Revision that must still be current.

        Returns:
            bool: `True` when deleted; `False` for absence at revision zero.

        Notes:
            Revision mismatch raises `StorageConflictError`.
        """
        ...

    async def query(self, query: DocumentQuery) -> Page[DocumentRecord]:
        """Query a bounded stable cursor page of canonical documents.

        Namespace, scope, identifier prefix, and promoted metadata filters apply
        before cursor pagination.

        Examples:
            List registry manifests:
                ```python
                page = await store.query(DocumentQuery(scope=scope, namespace="registry"))
                ```

            Continue filtered documents:
                ```python
                page = await store.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact canonical filters and opaque page request.

        Returns:
            Page[DocumentRecord]: Matching documents and continuation cursor.

        Notes:
            Implementations must not fetch every document and filter in Python.
        """
        ...

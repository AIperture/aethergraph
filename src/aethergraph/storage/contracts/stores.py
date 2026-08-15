"""Focused canonical protocols for high-frequency provider-owned stores."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from .pagination import Page, PageRequest
from .records import (
    ArtifactOccurrence,
    ArtifactRecord,
    ArtifactRelation,
    BlobHead,
    BlobRange,
    BlobWriteResult,
    EventDraft,
    EventRecord,
    FrozenJson,
    SearchDocument,
    SearchQuery,
    SearchResult,
    StateRecord,
)
from .scope import StorageScope


def _optional_utc(name: str, value: datetime | None) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value)):
        raise ValueError(f"{name} must be a timezone-aware UTC datetime when supplied")


class SortDirection(StrEnum):
    """Stable ordering direction requested from a cursor-paginated store."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass(frozen=True, slots=True, kw_only=True)
class EventQuery:
    """Bounded event query over exact canonical scope and promoted dimensions."""

    scope: StorageScope
    page: PageRequest = PageRequest()
    kinds: tuple[str, ...] = ()
    stage: str | None = None
    topic: str | None = None
    tags: tuple[str, ...] = ()
    occurred_at_min: datetime | None = None
    occurred_at_max: datetime | None = None
    order: SortDirection = SortDirection.DESCENDING

    def __post_init__(self) -> None:
        for name, values in (("kinds", self.kinds), ("tags", self.tags)):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be an immutable tuple")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        for name in ("stage", "topic"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when supplied")
        _optional_utc("occurred_at_min", self.occurred_at_min)
        _optional_utc("occurred_at_max", self.occurred_at_max)
        if (
            self.occurred_at_min is not None
            and self.occurred_at_max is not None
            and self.occurred_at_min > self.occurred_at_max
        ):
            raise ValueError("occurred_at_min must not be after occurred_at_max")


@dataclass(frozen=True, slots=True, kw_only=True)
class StateHistoryQuery:
    """Bounded state-history query for one exact namespace and key."""

    scope: StorageScope
    namespace: str
    key: str
    page: PageRequest = PageRequest()
    order: SortDirection = SortDirection.DESCENDING

    def __post_init__(self) -> None:
        for name in ("namespace", "key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


class EventStore(Protocol):
    """Ordered authoritative store for canonical runtime and memory events."""

    async def append(self, event: EventDraft) -> EventRecord:
        """Commit one event and assign its monotonic provider cursor.

        The event becomes authoritative before the method returns. Duplicate event
        identity follows the provider's documented idempotency semantics.

        Examples:
            Append a memory event:
                ```python
                committed = await store.append(event)
                ```

            Retain the returned cursor:
                ```python
                cursor = (await store.append(event)).cursor
                ```

        Args:
            event: Immutable canonical event content without a provider cursor.

        Returns:
            EventRecord: Committed event with its monotonic provider cursor.

        Notes:
            Search projection behavior is explicit provider configuration and never a
            fallback from the authoritative event write.
        """
        ...

    async def append_many(self, events: tuple[EventDraft, ...]) -> tuple[EventRecord, ...]:
        """Commit a bounded ordered batch of canonical events.

        Returned records preserve input order. Provider transaction semantics apply
        to the complete batch and are declared through capabilities.

        Examples:
            Append two events:
                ```python
                rows = await store.append_many((first, second))
                ```

            Append no events:
                ```python
                assert await store.append_many(()) == ()
                ```

        Args:
            events: Immutable bounded event batch in caller order.

        Returns:
            tuple[EventRecord, ...]: Committed records in the same order.

        Notes:
            Implementations must bound batch size and raise a typed configuration
            error rather than silently splitting an oversized atomic batch.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        event_id: str,
    ) -> EventRecord | None:
        """Read one exact event within canonical scope.

        The lookup is scope constrained and never searches another tenant or project
        when the requested identity is absent.

        Examples:
            Read an existing event:
                ```python
                event = await store.get(scope, "event-1")
                ```

            Handle a missing event:
                ```python
                assert await store.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner/execution scope constraining the lookup.
            event_id: Exact stable event identifier.

        Returns:
            EventRecord | None: Matching event or `None` when absent in that scope.

        Notes:
            Numeric provider cursors are not aliases for event identifiers.
        """
        ...

    async def query(self, query: EventQuery) -> Page[EventRecord]:
        """Read a stable bounded cursor page of canonical events.

        Filters apply before cursor pagination. Ordering and the next cursor remain
        stable for the logical stream represented by the query.

        Examples:
            Read recent events:
                ```python
                page = await store.query(EventQuery(scope=scope))
                ```

            Continue a page:
                ```python
                next_page = await store.query(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact filters, order, scope, and opaque page request.

        Returns:
            Page[EventRecord]: Matching records and optional continuation cursor.

        Notes:
            Unbounded production queries and offset pagination are not part of this
            protocol.
        """
        ...


class StateStore(Protocol):
    """Transactional current-state repository with optimistic revision control."""

    async def get(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
    ) -> StateRecord | None:
        """Read the current exact state record.

        The lookup addresses one namespace/key pair within canonical scope and does
        not scan general memory events.

        Examples:
            Read Agent state:
                ```python
                state = await store.get(scope, "agent", "writer")
                ```

            Detect missing state:
                ```python
                assert await store.get(scope, "graph", "missing") is None
                ```

        Args:
            scope: Canonical scope owning the state record.
            namespace: Exact service-owned state namespace.
            key: Exact state key within the namespace.

        Returns:
            StateRecord | None: Current record or `None` when no revision exists.

        Notes:
            This operation reads an indexed current row, not history convention.
        """
        ...

    async def get_many(
        self,
        scope: StorageScope,
        namespace: str,
        keys: tuple[str, ...],
    ) -> tuple[StateRecord | None, ...]:
        """Hydrate current state for multiple exact keys in one bounded call.

        Results preserve key order and include `None` placeholders for missing keys.

        Examples:
            Hydrate graph nodes:
                ```python
                rows = await store.get_many(scope, "graph", ("a", "b"))
                ```

            Hydrate no keys:
                ```python
                assert await store.get_many(scope, "graph", ()) == ()
                ```

        Args:
            scope: Canonical scope owning every requested state key.
            namespace: Exact shared namespace for the requested keys.
            keys: Immutable bounded key sequence in result order.

        Returns:
            tuple[StateRecord | None, ...]: Ordered current records or missing slots.

        Notes:
            Providers must bound key count and must not issue one connection per key.
        """
        ...

    async def compare_and_set(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
        value: FrozenJson,
        metadata: Mapping[str, FrozenJson],
    ) -> StateRecord:
        """Atomically write the next state revision when expectation matches.

        A successful transaction updates the current row and configured audit/outbox
        record together. Revision zero means the key must not yet exist.

        Examples:
            Create initial state:
                ```python
                row = await store.compare_and_set(scope, "agent", "writer", 0, {}, {})
                ```

            Advance existing state:
                ```python
                row = await store.compare_and_set(scope, "agent", "writer", 1, value, meta)
                ```

        Args:
            scope: Canonical scope owning the state record.
            namespace: Exact service-owned state namespace.
            key: Exact key within the namespace.
            expected_revision: Current revision required for the write, or zero for create.
            value: Complete JSON-compatible next state value.
            metadata: Immutable JSON-compatible audit metadata.

        Returns:
            StateRecord: Newly committed state with revision incremented by one.

        Notes:
            Stale expectations raise `StorageConflictError`; implementations do not
            fall back to read-then-write behavior.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        namespace: str,
        key: str,
        expected_revision: int,
    ) -> bool:
        """Delete current state only at one exact expected revision.

        The delete is atomic with configured audit/outbox behavior and cannot remove
        a newer concurrent revision.

        Examples:
            Delete current state:
                ```python
                deleted = await store.delete(scope, "agent", "writer", 3)
                ```

            Delete an absent key:
                ```python
                deleted = await store.delete(scope, "agent", "missing", 0)
                ```

        Args:
            scope: Canonical scope owning the state record.
            namespace: Exact service-owned namespace.
            key: Exact state key.
            expected_revision: Revision that must still be current.

        Returns:
            bool: `True` when a record was deleted; `False` when no record existed at revision zero.

        Notes:
            A present record at another revision raises `StorageConflictError`.
        """
        ...

    async def history(self, query: StateHistoryQuery) -> Page[StateRecord]:
        """Read a bounded stable page of committed state revisions.

        History is optional provider behavior declared at open time. Current-state
        reads remain independent from history storage.

        Examples:
            Read recent revisions:
                ```python
                page = await store.history(StateHistoryQuery(scope=scope, namespace="agent", key="writer"))
                ```

            Continue history:
                ```python
                page = await store.history(replace(query, page=PageRequest(cursor=cursor)))
                ```

        Args:
            query: Exact state identity, ordering, and opaque page request.

        Returns:
            Page[StateRecord]: Historical revisions and optional continuation cursor.

        Notes:
            Providers without configured history fail with a typed capability error.
        """
        ...


class BlobStore(Protocol):
    """Streaming provider-neutral storage for immutable artifact content."""

    async def put(
        self,
        scope: StorageScope,
        chunks: AsyncIterable[bytes],
        expected_hash: str | None = None,
        hash_algorithm: str = "sha256",
    ) -> BlobWriteResult:
        """Stage, hash, and atomically commit one immutable blob stream.

        Providers consume chunks incrementally and verify an optional expected hash
        before making the returned locator authoritative.

        Examples:
            Write a generated stream:
                ```python
                result = await store.put(scope, chunks())
                ```

            Verify a known digest:
                ```python
                result = await store.put(scope, chunks(), expected_hash=digest)
                ```

        Args:
            scope: Canonical owner scope for the immutable content.
            chunks: Asynchronous byte stream consumed once.
            expected_hash: Optional exact digest required before commit.
            hash_algorithm: Exact supported digest algorithm name.

        Returns:
            BlobWriteResult: Provider-neutral locator and verified content metadata.

        Notes:
            Integrity failures raise `StorageIntegrityError`; staging cleanup remains
            provider-owned and no partially committed locator is returned.
        """
        ...

    def read(
        self,
        scope: StorageScope,
        blob_locator: str,
        byte_range: BlobRange | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream complete or ranged immutable blob content.

        The iterator yields bounded byte chunks and validates canonical ownership
        before content is exposed.

        Examples:
            Read complete content:
                ```python
                async for chunk in store.read(scope, locator):
                    consume(chunk)
                ```

            Read a byte range:
                ```python
                async for chunk in store.read(scope, locator, BlobRange(start=0, end=1024)):
                    consume(chunk)
                ```

        Args:
            scope: Canonical owner scope constraining access.
            blob_locator: Opaque provider-neutral locator returned by `put`.
            byte_range: Optional half-open byte range.

        Returns:
            AsyncIterator[bytes]: Bounded chunks in content order.

        Notes:
            Callers never parse provider-private paths, buckets, or object keys.
        """
        ...

    async def head(self, scope: StorageScope, blob_locator: str) -> BlobHead | None:
        """Read immutable blob metadata without content transfer.

        The lookup is constrained by canonical owner scope and returns no physical
        provider path details.

        Examples:
            Inspect stored content:
                ```python
                head = await store.head(scope, locator)
                ```

            Detect missing content:
                ```python
                assert await store.head(scope, "blob:missing") is None
                ```

        Args:
            scope: Canonical owner scope constraining access.
            blob_locator: Exact opaque provider-neutral blob locator.

        Returns:
            BlobHead | None: Verified metadata or `None` when absent in that scope.

        Notes:
            Provider object versions may be returned for integrity, never authorization.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        blob_locator: str,
        provider_version: str | None = None,
    ) -> bool:
        """Delete immutable content under explicit retention ownership.

        An optional provider version prevents deleting a newer object at the same
        provider-private key.

        Examples:
            Delete unversioned content:
                ```python
                deleted = await store.delete(scope, locator)
                ```

            Delete one exact version:
                ```python
                deleted = await store.delete(scope, locator, provider_version=etag)
                ```

        Args:
            scope: Canonical owner scope constraining deletion.
            blob_locator: Exact opaque locator selected by retention.
            provider_version: Optional exact provider object version.

        Returns:
            bool: `True` when content was deleted; `False` when absent.

        Notes:
            Only retention/administration owners call this operation.
        """
        ...


class ArtifactRepository(Protocol):
    """Metadata, occurrence, and lineage repository for canonical artifacts."""

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        """Idempotently commit immutable artifact content metadata.

        Repeating the same artifact identity and immutable fields succeeds. Conflicting
        immutable metadata for that identity raises an integrity error.

        Examples:
            Commit new metadata:
                ```python
                stored = await repository.put(record)
                ```

            Retry an idempotent commit:
                ```python
                assert await repository.put(record) == stored
                ```

        Args:
            record: Complete immutable canonical artifact content metadata.

        Returns:
            ArtifactRecord: Authoritative stored metadata.

        Notes:
            Blob commit failure handling is coordinated by the provider, not this
            repository through distributed-transaction claims.
        """
        ...

    async def get(
        self,
        scope: StorageScope,
        artifact_id: str,
    ) -> ArtifactRecord | None:
        """Read one exact artifact record within owner scope.

        The lookup returns immutable content metadata without occurrence duplication.

        Examples:
            Read artifact metadata:
                ```python
                artifact = await repository.get(scope, "artifact-1")
                ```

            Detect missing metadata:
                ```python
                assert await repository.get(scope, "missing") is None
                ```

        Args:
            scope: Canonical owner scope constraining access.
            artifact_id: Exact stable artifact content identity.

        Returns:
            ArtifactRecord | None: Matching record or `None` when absent.

        Notes:
            Occurrences and lineage require their focused methods.
        """
        ...

    async def record_occurrence(self, occurrence: ArtifactOccurrence) -> ArtifactOccurrence:
        """Idempotently commit one artifact production or use occurrence.

        The referenced artifact must already exist. The occurrence stores execution
        context only and never copies immutable content metadata.

        Examples:
            Record production:
                ```python
                stored = await repository.record_occurrence(occurrence)
                ```

            Retry an occurrence:
                ```python
                assert await repository.record_occurrence(occurrence) == stored
                ```

        Args:
            occurrence: Complete canonical occurrence with stable identity.

        Returns:
            ArtifactOccurrence: Authoritative stored occurrence.

        Notes:
            A missing artifact raises `StorageNotFoundError`; conflicting duplicate
            identity raises `StorageIntegrityError`.
        """
        ...

    async def list_occurrences(
        self,
        scope: StorageScope,
        page: PageRequest,
        artifact_id: str | None = None,
    ) -> Page[ArtifactOccurrence]:
        """List a bounded stable page of artifact occurrences.

        Canonical scope filters apply before optional content identity and cursor
        pagination.

        Examples:
            List run occurrences:
                ```python
                page = await repository.list_occurrences(run_scope, PageRequest())
                ```

            List one artifact's uses:
                ```python
                page = await repository.list_occurrences(scope, PageRequest(), "artifact-1")
                ```

        Args:
            scope: Canonical execution/owner scope filters.
            page: Bounded opaque cursor request.
            artifact_id: Optional exact artifact identity filter.

        Returns:
            Page[ArtifactOccurrence]: Matching occurrences and continuation cursor.

        Notes:
            The protocol has no offset or unbounded list operation.
        """
        ...

    async def add_relation(self, relation: ArtifactRelation) -> ArtifactRelation:
        """Idempotently commit one typed directed artifact lineage edge.

        Both source and target artifacts must exist in authorized owner scope before
        the edge is committed.

        Examples:
            Record derivation:
                ```python
                stored = await repository.add_relation(relation)
                ```

            Retry an edge:
                ```python
                assert await repository.add_relation(relation) == stored
                ```

        Args:
            relation: Complete canonical directed lineage relation.

        Returns:
            ArtifactRelation: Authoritative stored lineage edge.

        Notes:
            Missing endpoints raise `StorageNotFoundError`; self-edges are invalid.
        """
        ...

    async def list_relations(
        self,
        scope: StorageScope,
        artifact_id: str,
        page: PageRequest,
    ) -> Page[ArtifactRelation]:
        """List a bounded stable page of lineage touching one artifact.

        The provider returns authorized incoming and outgoing edges with stable cursor
        ordering.

        Examples:
            Read initial lineage:
                ```python
                page = await repository.list_relations(scope, "artifact-1", PageRequest())
                ```

            Continue lineage:
                ```python
                page = await repository.list_relations(scope, "artifact-1", PageRequest(cursor=cursor))
                ```

        Args:
            scope: Canonical owner scope constraining every endpoint.
            artifact_id: Exact artifact identity whose lineage is requested.
            page: Bounded opaque cursor request.

        Returns:
            Page[ArtifactRelation]: Matching edges and continuation cursor.

        Notes:
            Implementations must not hydrate artifact content for this metadata query.
        """
        ...


class SearchBackend(Protocol):
    """One explicit-mode canonical search service for Memory and named corpora."""

    async def upsert(self, document: SearchDocument) -> str:
        """Index one canonical document and return the indexed cursor.

        The operation uses the exact configured consistency mode and never downgrades
        to a different search implementation after failure.

        Examples:
            Index one memory event:
                ```python
                cursor = await search.upsert(document)
                ```

            Retain freshness state:
                ```python
                indexed_cursor = await search.upsert(document)
                ```

        Args:
            document: Complete canonical searchable projection.

        Returns:
            str: Latest provider-owned indexed cursor after this write.

        Notes:
            Stable item identity makes retries idempotent within one corpus.
        """
        ...

    async def upsert_many(self, documents: tuple[SearchDocument, ...]) -> str | None:
        """Index a bounded ordered document batch.

        The returned cursor covers every committed document in the batch. An empty
        batch performs no write and returns `None`.

        Examples:
            Index a batch:
                ```python
                cursor = await search.upsert_many((first, second))
                ```

            Index no documents:
                ```python
                assert await search.upsert_many(()) is None
                ```

        Args:
            documents: Immutable bounded batch, normally from one corpus.

        Returns:
            str | None: Covering indexed cursor, or `None` for an empty batch.

        Notes:
            Providers reject unsupported cross-corpus atomic batches explicitly.
        """
        ...

    async def delete(
        self,
        scope: StorageScope,
        corpus: str,
        item_ids: tuple[str, ...],
    ) -> str | None:
        """Delete exact searchable projections and return indexed freshness.

        Deletion is scope constrained and idempotent for missing items.

        Examples:
            Delete one projection:
                ```python
                cursor = await search.delete(scope, "memory", ("event-1",))
                ```

            Delete no projections:
                ```python
                assert await search.delete(scope, "memory", ()) is None
                ```

        Args:
            scope: Canonical owner scope constraining deletion.
            corpus: Exact named search corpus.
            item_ids: Immutable bounded stable item identities.

        Returns:
            str | None: Latest indexed cursor, or `None` for an empty request.

        Notes:
            This operation does not delete authoritative memory events or artifacts.
        """
        ...

    async def query(self, query: SearchQuery) -> tuple[SearchResult, ...]:
        """Execute one bounded explicit-mode search query.

        The requested mode and freshness requirement must be supported. Results keep
        stable corpus/item identity and descending provider score order.

        Examples:
            Execute semantic search:
                ```python
                results = await search.query(query)
                ```

            Execute structural search:
                ```python
                recent = await search.query(structural_query)
                ```

        Args:
            query: Exact corpus, mode, scope, filters, bound, and freshness request.

        Returns:
            tuple[SearchResult, ...]: At most `top_k` stable ordered results.

        Notes:
            Unsupported lexical/hybrid modes raise a typed capability error and never
            fall back to semantic search.
        """
        ...

    async def indexed_cursor(self, corpus: str) -> str | None:
        """Return the latest committed search-index cursor for one corpus.

        The cursor is opaque and used only for freshness comparison/waits.

        Examples:
            Inspect current freshness:
                ```python
                cursor = await search.indexed_cursor("memory")
                ```

            Detect an empty corpus:
                ```python
                assert await search.indexed_cursor("new") is None
                ```

        Args:
            corpus: Exact named search corpus.

        Returns:
            str | None: Latest opaque indexed cursor, or `None` before first indexing.

        Notes:
            Cursors are provider-specific and not parsed by services.
        """
        ...

    async def wait_until_indexed(
        self,
        corpus: str,
        cursor: str,
        timeout_seconds: float,
    ) -> str:
        """Wait bounded time until a corpus covers the required cursor.

        The wait returns the current covering cursor and never changes indexing mode
        after a projection failure.

        Examples:
            Wait for asynchronous indexing:
                ```python
                covered = await search.wait_until_indexed("memory", cursor, 5.0)
                ```

            Perform a nonblocking freshness check:
                ```python
                covered = await search.wait_until_indexed("memory", cursor, 0.0)
                ```

        Args:
            corpus: Exact named search corpus.
            cursor: Opaque authoritative event/index cursor that must be covered.
            timeout_seconds: Non-negative maximum wait duration.

        Returns:
            str: Current indexed cursor covering the requirement.

        Notes:
            Timeout raises `StorageTimeoutError`; callers choose whether stale search
            results are acceptable before invoking this method.
        """
        ...

"""Canonical memory events, bounded hot cache, and explicit search projection."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from time import monotonic
from typing import Any, Literal

from aethergraph.storage.contracts import (
    EventDraft,
    EventQuery,
    EventRecord,
    EventSearchProjectionIntent,
    EventStore,
    FrozenJson,
    Page,
    PageRequest,
    SearchBackend,
    SearchDocument,
    SearchMode,
    SearchProjectionStatus,
    SearchQuery,
    SearchResult,
    SortDirection,
    StateHistoryQuery,
    StateRecord,
    StateStore,
    StorageScope,
)

_MEMORY_CORPUS = "memory"
_MEMORY_STATE_NAMESPACE_PREFIX = "memory.state"
_MAX_HOT_EVENTS = 10_000


def _search_intent_id(event_id: str) -> str:
    return f"memory-search:{event_id}"


@dataclass(frozen=True, slots=True)
class MemoryCommitReceipt:
    """Committed authoritative events and covering search-index freshness."""

    events: tuple[EventRecord, ...]
    event_cursor: str | None
    indexed_cursor: str | None
    created: tuple[bool, ...] = ()
    projection_status: Literal["not_requested", "indexed", "failed"] = "not_requested"
    projection_diagnostic: str | None = None


class CanonicalMemoryFacade:
    """Provider-neutral memory facade with explicit durability and search semantics."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        state_store: StateStore,
        search_backend: SearchBackend,
        scope: StorageScope,
        event_scope: StorageScope | None = None,
        hot_max_events: int = 500,
        hot_ttl_seconds: float = 900.0,
        monotonic_clock: Callable[[], float] = monotonic,
    ) -> None:
        """Compose canonical memory over one bundle's event and search stores.

        Retains exact dependencies and creates one bounded process-local cache without
        selecting a provider, adapting legacy events, or enabling fallback search.

        Examples:
            Build from one bundle:
                ```python
                memory = CanonicalMemoryFacade(
                    event_store=bundle.memory_events,
                    state_store=bundle.state,
                    search_backend=bundle.search,
                    scope=scope,
                )
                ```

            Configure a smaller hot cache:
                ```python
                memory = CanonicalMemoryFacade(
                    event_store=events,
                    state_store=state,
                    search_backend=search,
                    scope=scope,
                    hot_max_events=100,
                    hot_ttl_seconds=60.0,
                )
                ```

        Args:
            event_store: Canonical authoritative memory event stream.
            state_store: Canonical transactional memory-state repository.
            search_backend: Canonical exact-mode searchable projection.
            scope: Exact immutable memory ownership/execution scope.
            event_scope: Optional full event provenance containing every bucket dimension.
            hot_max_events: Positive maximum records retained in process.
            hot_ttl_seconds: Positive insertion-age lifetime for hot records.
            monotonic_clock: Monotonic time source used only for cache expiry.

        Returns:
            None: The provider-backed facade is ready without I/O.

        Notes:
            Durable events remain authoritative; hot cache and search are projections.
        """
        if isinstance(hot_max_events, bool) or not isinstance(hot_max_events, int):
            raise TypeError("hot_max_events must be an integer")
        if hot_max_events < 1:
            raise ValueError("hot_max_events must be positive")
        if hot_max_events > _MAX_HOT_EVENTS:
            raise ValueError(f"hot_max_events must not exceed {_MAX_HOT_EVENTS}")
        if isinstance(hot_ttl_seconds, bool) or not isinstance(hot_ttl_seconds, int | float):
            raise TypeError("hot_ttl_seconds must be numeric")
        if hot_ttl_seconds <= 0:
            raise ValueError("hot_ttl_seconds must be positive")
        self._events = event_store
        self._state = state_store
        self._search = search_backend
        self.scope = scope
        self.event_scope = event_scope or scope
        for name, value in scope.as_filter().items():
            if name != "scope_key" and getattr(self.event_scope, name) != value:
                raise ValueError(f"Memory event scope conflicts with bucket scope {name}")
        self._hot_max_events = hot_max_events
        self._hot_ttl_seconds = float(hot_ttl_seconds)
        self._monotonic = monotonic_clock
        self._hot: deque[tuple[float, EventRecord]] = deque(maxlen=hot_max_events)
        self._hot_lock = asyncio.Lock()

    async def append_event(
        self,
        *,
        event_id: str,
        occurred_at: datetime,
        kind: str,
        stage: str | None = None,
        topic: str | None = None,
        text: str | None = None,
        tags: tuple[str, ...] = (),
        payload: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        severity: int | None = None,
        signal: float | None = None,
    ) -> MemoryCommitReceipt:
        """Commit and index one normalized canonical memory event.

        Constructs only canonical event fields, then delegates to the same bulk pipeline
        used for multi-event commits.

        Examples:
            Append a user message:
                ```python
                receipt = await memory.append_event(
                    event_id="event-1",
                    occurred_at=now,
                    kind="user.message",
                    text="hello",
                )
                ```

            Append a tool result:
                ```python
                receipt = await memory.append_event(
                    event_id="event-2",
                    occurred_at=now,
                    kind="tool.result",
                    topic="search",
                    payload={"count": 3},
                )
                ```

        Args:
            event_id: Stable caller-owned idempotency identity.
            occurred_at: Timezone-aware UTC event time.
            kind: Exact normalized event family.
            stage: Optional indexed execution stage.
            topic: Optional indexed semantic topic.
            text: Optional searchable human-readable content.
            tags: Immutable unique indexed tags.
            payload: Optional JSON-compatible event-specific content.
            metrics: Optional finite numeric metrics.
            severity: Optional normalized severity from zero through 100.
            signal: Optional finite relevance signal.

        Returns:
            MemoryCommitReceipt: Authoritative event and covering search cursors.

        Notes:
            App/client/tool aliases and inline embeddings are absent by construction.
        """
        return await self.append_many(
            (
                EventDraft(
                    event_id=event_id,
                    occurred_at=occurred_at,
                    scope=self.event_scope,
                    kind=kind,
                    stage=stage,
                    topic=topic,
                    text=text,
                    tags=tags,
                    payload=dict(payload or {}),
                    metrics=dict(metrics or {}),
                    severity=severity,
                    signal=signal,
                ),
            )
        )

    async def append_many(self, events: tuple[EventDraft, ...]) -> MemoryCommitReceipt:
        """Commit, cache, and index one bounded normalized event batch.

        Authoritative event commit occurs first, hot-cache projection second, and
        searchable projection third. Any search failure remains visible to the caller.

        Examples:
            Append a batch:
                ```python
                receipt = await memory.append_many((first, second))
                ```

            Append no events:
                ```python
                receipt = await memory.append_many(())
                assert receipt.events == ()
                ```

        Args:
            events: Immutable bounded canonical events in caller order.

        Returns:
            MemoryCommitReceipt: Committed records and latest event/search cursors.

        Notes:
            Exact retry is idempotent; no projection failure selects another backend.
        """
        if not isinstance(events, tuple):
            raise TypeError("events must be an immutable tuple")
        for event in events:
            if event.scope != self.event_scope:
                raise ValueError("Memory event scope must exactly match the bound event provenance")
        if not events:
            return MemoryCommitReceipt(events=(), event_cursor=None, indexed_cursor=None)
        pending_intents = tuple(
            EventSearchProjectionIntent(
                intent_id=_search_intent_id(event.event_id),
                event_id=event.event_id,
                scope=event.scope,
                status=SearchProjectionStatus.PENDING,
                revision=1,
                attempts=0,
                updated_at=event.occurred_at,
            )
            for event in events
        )
        committed, intents, created = await self._events.append_many_with_search_intents(
            events,
            pending_intents,
        )
        inserted_at = self._monotonic()
        async with self._hot_lock:
            self._evict_expired(inserted_at)
            cached_ids = {event.event_id for _cached_at, event in self._hot}
            for event in committed:
                if event.event_id in cached_ids:
                    continue
                self._hot.append((inserted_at, event))
                cached_ids.add(event.event_id)
        projection_pairs = tuple(
            (event, intent)
            for event, intent in zip(committed, intents, strict=True)
            if intent.status is not SearchProjectionStatus.INDEXED
        )
        if not projection_pairs:
            return MemoryCommitReceipt(
                events=committed,
                event_cursor=committed[-1].cursor,
                indexed_cursor=intents[-1].indexed_cursor,
                created=created,
                projection_status="indexed",
            )
        try:
            indexed_cursor = await self._search.upsert_many(
                tuple(
                    _search_document(event, scope=self.scope) for event, _intent in projection_pairs
                )
            )
        except Exception as exc:
            diagnostic = f"{type(exc).__name__}: search projection failed"
            failed = []
            for _event, intent in projection_pairs:
                failed.append(
                    await self._events.compare_and_set_search_intent(
                        replace(
                            intent,
                            status=SearchProjectionStatus.FAILED,
                            revision=intent.revision + 1,
                            attempts=intent.attempts + 1,
                            updated_at=datetime.now(intent.updated_at.tzinfo),
                            diagnostic=diagnostic,
                        ),
                        intent.revision,
                    )
                )
            return MemoryCommitReceipt(
                events=committed,
                event_cursor=committed[-1].cursor,
                indexed_cursor=None,
                created=created,
                projection_status="failed",
                projection_diagnostic=diagnostic,
            )
        for _event, intent in projection_pairs:
            await self._events.compare_and_set_search_intent(
                replace(
                    intent,
                    status=SearchProjectionStatus.INDEXED,
                    revision=intent.revision + 1,
                    attempts=intent.attempts + 1,
                    updated_at=datetime.now(intent.updated_at.tzinfo),
                    indexed_cursor=indexed_cursor,
                    diagnostic=None,
                ),
                intent.revision,
            )
        return MemoryCommitReceipt(
            events=committed,
            event_cursor=committed[-1].cursor,
            indexed_cursor=indexed_cursor,
            created=created,
            projection_status="indexed",
        )

    async def get_search_projection_intent(
        self,
        event_id: str,
    ) -> EventSearchProjectionIntent | None:
        """Read the durable search intent for one authoritative Memory event.

        Examples:
            Inspect failed projection:
                ```python
                intent = await memory.get_search_projection_intent("event-1")
                ```

            Detect absence:
                ```python
                assert await memory.get_search_projection_intent("missing") is None
                ```

        Args:
            event_id: Stable caller-owned event identity.

        Returns:
            EventSearchProjectionIntent | None: Durable intent or `None`.

        Notes:
            The lookup is exact and never scans or selects another provider.
        """
        return await self._events.get_search_intent(
            self.event_scope,
            _search_intent_id(event_id),
        )

    async def retry_search_projection(self, event_id: str) -> MemoryCommitReceipt:
        """Retry one pending or failed search projection without re-appending its event.

        Examples:
            Retry failed work:
                ```python
                receipt = await memory.retry_search_projection("event-1")
                ```

            Retry an already indexed event:
                ```python
                assert (await memory.retry_search_projection("event-1")).projection_status == "indexed"
                ```

        Args:
            event_id: Stable authoritative event identity.

        Returns:
            MemoryCommitReceipt: Same authoritative event and current projection outcome.

        Notes:
            Exact indexed retries perform no second search upsert.
        """
        event = await self._events.get(self.event_scope, event_id)
        intent = await self.get_search_projection_intent(event_id)
        if event is None or intent is None:
            raise ValueError("Authoritative event and projection intent must both exist")
        if intent.status is SearchProjectionStatus.INDEXED:
            return MemoryCommitReceipt(
                events=(event,),
                event_cursor=event.cursor,
                indexed_cursor=intent.indexed_cursor,
                created=(False,),
                projection_status="indexed",
            )
        try:
            indexed_cursor = await self._search.upsert_many(
                (_search_document(event, scope=self.scope),)
            )
        except Exception as exc:
            diagnostic = f"{type(exc).__name__}: search projection failed"
            await self._events.compare_and_set_search_intent(
                replace(
                    intent,
                    status=SearchProjectionStatus.FAILED,
                    revision=intent.revision + 1,
                    attempts=intent.attempts + 1,
                    updated_at=datetime.now(intent.updated_at.tzinfo),
                    diagnostic=diagnostic,
                ),
                intent.revision,
            )
            return MemoryCommitReceipt(
                events=(event,),
                event_cursor=event.cursor,
                indexed_cursor=None,
                created=(False,),
                projection_status="failed",
                projection_diagnostic=diagnostic,
            )
        await self._events.compare_and_set_search_intent(
            replace(
                intent,
                status=SearchProjectionStatus.INDEXED,
                revision=intent.revision + 1,
                attempts=intent.attempts + 1,
                updated_at=datetime.now(intent.updated_at.tzinfo),
                indexed_cursor=indexed_cursor,
                diagnostic=None,
            ),
            intent.revision,
        )
        return MemoryCommitReceipt(
            events=(event,),
            event_cursor=event.cursor,
            indexed_cursor=indexed_cursor,
            created=(False,),
            projection_status="indexed",
        )

    async def durable_query(self, query: EventQuery) -> Page[EventRecord]:
        """Read one stable authoritative cursor page of canonical memory events.

        Requires exact facade scope before delegating the complete bounded query to the
        provider event store.

        Examples:
            Read recent events:
                ```python
                page = await memory.durable_query(EventQuery(scope=scope))
                ```

            Continue a page:
                ```python
                page = await memory.durable_query(next_query)
                ```

        Args:
            query: Exact canonical event filters and opaque page request.

        Returns:
            Page[EventRecord]: Authoritative matching events and continuation cursor.

        Notes:
            This method never reads hot cache after a durable miss.
        """
        if query.scope != self.scope:
            raise ValueError("Memory query scope must exactly match the bound facade scope")
        return await self._events.query(query)

    async def get_event(self, event_id: str) -> EventRecord | None:
        """Read one exact canonical memory event by stable identity.

        The lookup is constrained to the facade's immutable storage scope and never
        interprets a provider cursor as an event identifier.

        Examples:
            Read an existing event:
                ```python
                event = await memory.get_event("event-1")
                ```

            Detect an absent event:
                ```python
                assert await memory.get_event("missing") is None
                ```

        Args:
            event_id: Exact stable caller-owned event identity.

        Returns:
            EventRecord | None: Matching canonical event or `None` when absent.

        Notes:
            A miss does not consult hot cache, search projection, or another store.
        """
        return await self._events.get(self.scope, event_id)

    async def commit_state(
        self,
        *,
        key: str,
        value: FrozenJson,
        expected_revision: int,
        kind: str = "state.snapshot",
        metadata: Mapping[str, FrozenJson] | None = None,
    ) -> StateRecord:
        """Commit one exact memory-state revision through canonical CAS.

        The complete value is stored in a kind-specific namespace. The provider
        atomically commits current state, retained history, and its audit/outbox row;
        the facade does not duplicate the snapshot into the memory event stream.

        Examples:
            Create initial state:
                ```python
                stored = await memory.commit_state(
                    key="agent:writer",
                    value={"draft": 1},
                    expected_revision=0,
                )
                ```

            Advance a custom state family:
                ```python
                stored = await memory.commit_state(
                    key="checkpoint",
                    value={"step": 2},
                    expected_revision=1,
                    kind="workflow.checkpoint",
                    metadata={"source": "planner"},
                )
                ```

        Args:
            key: Exact caller-owned state key within the memory scope.
            value: Complete JSON-compatible state value.
            expected_revision: Exact current revision, or zero for initial creation.
            kind: Exact state family used to isolate the provider namespace.
            metadata: Optional JSON-compatible audit metadata.

        Returns:
            StateRecord: Newly committed canonical state record.

        Notes:
            Conflicts propagate from `StateStore`; no retry, event append, legacy
            lookup, or alternate persistence path is attempted.
        """
        return await self._state.compare_and_set(
            self.scope,
            _memory_state_namespace(kind),
            _memory_state_key(key),
            expected_revision,
            value,
            dict(metadata or {}),
        )

    async def current_state(
        self,
        *,
        key: str,
        kind: str = "state.snapshot",
    ) -> StateRecord | None:
        """Read one exact current memory-state record.

        The lookup addresses canonical scope, state family, and key directly without
        scanning memory events or consulting a legacy state-snapshot convention.

        Examples:
            Read current Agent state:
                ```python
                current = await memory.current_state(key="agent:writer")
                ```

            Read a custom state family:
                ```python
                checkpoint = await memory.current_state(
                    key="checkpoint",
                    kind="workflow.checkpoint",
                )
                ```

        Args:
            key: Exact caller-owned state key within the memory scope.
            kind: Exact state family used to isolate the provider namespace.

        Returns:
            StateRecord | None: Current canonical record or `None` when absent.

        Notes:
            A durable miss remains a miss; there is no hot-cache or EventStore fallback.
        """
        return await self._state.get(
            self.scope,
            _memory_state_namespace(kind),
            _memory_state_key(key),
        )

    async def state_history(
        self,
        *,
        key: str,
        kind: str = "state.snapshot",
        limit: int = 50,
        cursor: str | None = None,
        order: SortDirection = SortDirection.DESCENDING,
    ) -> Page[StateRecord]:
        """Read one bounded opaque-cursor page of memory-state history.

        Provider state history is the sole durable audit authority for snapshots.
        Exact namespace and key filters apply before provider pagination.

        Examples:
            Read recent revisions:
                ```python
                page = await memory.state_history(key="agent:writer", limit=20)
                ```

            Continue oldest-first history:
                ```python
                page = await memory.state_history(
                    key="agent:writer",
                    cursor=previous.next_cursor,
                    order=SortDirection.ASCENDING,
                )
                ```

        Args:
            key: Exact caller-owned state key within the memory scope.
            kind: Exact state family used to isolate the provider namespace.
            limit: Positive provider page bound.
            cursor: Optional opaque continuation cursor from the same query.
            order: Exact canonical revision-history ordering direction.

        Returns:
            Page[StateRecord]: Matching retained revisions and continuation cursor.

        Notes:
            Opaque cursors are passed through unchanged and are never parsed as local
            SQLite row identifiers.
        """
        return await self._state.history(
            StateHistoryQuery(
                scope=self.scope,
                namespace=_memory_state_namespace(kind),
                key=_memory_state_key(key),
                page=PageRequest(limit=limit, cursor=cursor),
                order=order,
            )
        )

    async def recent_hot(
        self,
        *,
        limit: int = 50,
        kinds: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
    ) -> tuple[EventRecord, ...]:
        """Read bounded newest-first records from only the process-local hot cache.

        Evicts expired insertions before applying exact kind and all-tag filters.

        Examples:
            Read recent hot events:
                ```python
                events = await memory.recent_hot(limit=20)
                ```

            Filter hot tool results:
                ```python
                events = await memory.recent_hot(kinds=("tool.result",), tags=("verified",))
                ```

        Args:
            limit: Positive maximum number of records returned.
            kinds: Optional exact event kinds.
            tags: Optional tags every returned event must contain.

        Returns:
            tuple[EventRecord, ...]: Matching nonexpired records newest first.

        Notes:
            Cache expiry or eviction never triggers a durable fallback.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        now = self._monotonic()
        async with self._hot_lock:
            self._evict_expired(now)
            rows = tuple(event for _inserted_at, event in reversed(self._hot))
        allowed = set(kinds)
        required_tags = set(tags)
        return tuple(
            event
            for event in rows
            if (not allowed or event.kind in allowed) and required_tags.issubset(event.tags)
        )[:limit]

    async def search(
        self,
        *,
        query: str,
        mode: SearchMode,
        top_k: int = 10,
        tags: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        occurred_at_min: datetime | None = None,
        occurred_at_max: datetime | None = None,
        require_indexed_cursor: str | None = None,
    ) -> tuple[SearchResult, ...]:
        """Execute one explicit-mode canonical memory search.

        Passes exact mode, scope, metadata, time bounds, result bound, and optional
        freshness cursor directly to the provider search backend.

        Examples:
            Search lexically:
                ```python
                hits = await memory.search(query="hello", mode=SearchMode.LEXICAL)
                ```

            Require a committed projection:
                ```python
                hits = await memory.search(
                    query="hello",
                    mode=SearchMode.LEXICAL,
                    require_indexed_cursor=receipt.indexed_cursor,
                )
                ```

        Args:
            query: Exact search text; structural mode may use an empty value.
            mode: Required search mode with no fallback.
            top_k: Positive provider result bound up to 1000.
            tags: Optional tags every indexed event must contain.
            metadata: Optional exact indexed metadata filters.
            occurred_at_min: Optional inclusive UTC lower time bound.
            occurred_at_max: Optional inclusive UTC upper time bound.
            require_indexed_cursor: Optional covering search cursor requirement.

        Returns:
            tuple[SearchResult, ...]: Stable provider-ranked memory hits.

        Notes:
            Results carry identities and scores; event hydration is an explicit later read.
        """
        return await self._search.query(
            SearchQuery(
                corpus=_MEMORY_CORPUS,
                mode=mode,
                scope=self.scope,
                query=query,
                top_k=top_k,
                tags=tags,
                metadata=dict(metadata or {}),
                occurred_at_min=occurred_at_min,
                occurred_at_max=occurred_at_max,
                require_indexed_cursor=require_indexed_cursor,
            )
        )

    async def indexed_cursor(self) -> str | None:
        """Return the latest committed canonical memory search cursor.

        Reads only the named memory corpus freshness state.

        Examples:
            Read current freshness:
                ```python
                cursor = await memory.indexed_cursor()
                ```

            Detect no indexed events:
                ```python
                assert await memory.indexed_cursor() is None
                ```

        Args:
            None.

        Returns:
            str | None: Opaque indexed cursor or `None` before first projection.

        Notes:
            The cursor is never parsed or compared by the facade.
        """
        return await self._search.indexed_cursor(_MEMORY_CORPUS)

    async def wait_until_indexed(self, cursor: str, timeout_seconds: float) -> str:
        """Wait a bounded interval for canonical memory search freshness.

        Delegates the opaque required cursor and timeout without changing search mode
        or selecting another projection.

        Examples:
            Wait for a projection:
                ```python
                covered = await memory.wait_until_indexed(receipt.indexed_cursor, 5.0)
                ```

            Check without waiting:
                ```python
                covered = await memory.wait_until_indexed(cursor, 0.0)
                ```

        Args:
            cursor: Opaque required memory search cursor.
            timeout_seconds: Non-negative maximum wait duration.

        Returns:
            str: Current covering search cursor.

        Notes:
            Provider timeout failure propagates directly.
        """
        return await self._search.wait_until_indexed(
            _MEMORY_CORPUS,
            cursor,
            timeout_seconds,
        )

    def _evict_expired(self, now: float) -> None:
        cutoff = now - self._hot_ttl_seconds
        while self._hot and self._hot[0][0] <= cutoff:
            self._hot.popleft()


def _search_document(event: EventRecord, *, scope: StorageScope) -> SearchDocument:
    metadata: dict[str, Any] = {
        "event_cursor": event.cursor,
        "kind": event.kind,
        "tags": list(event.tags),
    }
    if event.stage is not None:
        metadata["stage"] = event.stage
    if event.topic is not None:
        metadata["topic"] = event.topic
    return SearchDocument(
        corpus=_MEMORY_CORPUS,
        item_id=event.event_id,
        text=event.text or "",
        scope=scope,
        occurred_at=event.occurred_at,
        tags=event.tags,
        metadata=metadata,
    )


def _memory_state_namespace(kind: str) -> str:
    if not isinstance(kind, str) or not kind.strip() or kind != kind.strip():
        raise ValueError("Memory state kind must be a non-empty string")
    return f"{_MEMORY_STATE_NAMESPACE_PREFIX}.{kind}"


def _memory_state_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip() or key != key.strip():
        raise ValueError("Memory state key must be a non-empty string")
    return key

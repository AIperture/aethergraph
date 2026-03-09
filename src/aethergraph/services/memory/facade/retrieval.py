from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple
import warnings

from aethergraph.contracts.storage.search_backend import ScoredItem, SearchMode
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.memory.facade.utils import event_matches_level
from aethergraph.services.scope.scope import ScopeLevel

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event, MemoryFacadeProtocol


class EventSearchResult(NamedTuple):
    item: ScoredItem
    event: Event | None

    @property
    def score(self) -> float:
        """
        Return the numeric search score from the backing scored item.

        This property mirrors `item.score` for convenience when consuming
        `EventSearchResult` rows.

        Examples:
            Read a score value:
            ```python
            score = result.score
            ```

            Sort by score descending:
            ```python
            ranked = sorted(results, key=lambda r: r.score, reverse=True)
            ```

        Args:
            None.

        Returns:
            float: Search score associated with this result row.
        """
        return self.item.score


class RetrievalMixin:
    """Methods for retrieving events and values."""

    async def get_event(self, event_id: str) -> Event | None:
        """
        Retrieve a specific event by ID.

        The lookup first checks hotlog, then falls back to persistence when
        supported by the configured backend.

        Examples:
            Fetch a known event:
            ```python
            evt = await context.memory().get_event("evt_123")
            ```

            Handle missing events:
            ```python
            evt = await context.memory().get_event("evt_missing")
            if evt is None:
                ...
            ```

        Args:
            event_id: Unique event identifier to resolve.

        Returns:
            Event | None: The resolved event, or None when not found.
        """
        # 1) Try hotlog
        recent = await self.hotlog.query(
            self.timeline_id,
            tenant=getattr(self, "memory_tenant", None),
            kinds=None,
            limit=self.hot_limit,
        )
        for e in recent:
            if e.event_id == event_id:
                return e

        # 2) Fallback to persistence
        if hasattr(self.persistence, "get_events_by_ids"):
            events = await self.persistence.get_events_by_ids(
                self.timeline_id,
                [event_id],
                tenant=getattr(self, "memory_tenant", None),
            )
            return events[0] if events else None

        return None

    # ------ Hotlog Retrieval ------
    async def recent(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
        return_event: bool = True,
    ) -> list[Any]:
        """
        Retrieve recent events.

        This method fetches a list of recent events, optionally filtered by kinds.

        Examples:
            Return Event objects (default):
            ```python
            events = await context.memory().recent(limit=20)
            ```

            Return normalized dict payloads:
            ```python
            rows = await context.memory().recent(limit=20, return_event=False)
            ```

        Args:
            kinds: A list of event kinds to filter by. Defaults to None.
            limit: The maximum number of events to retrieve. Defaults to 50.
            level: Optional scope level to filter events by. If provided, only events associated with the specified scope level will be returned.
            return_event: If True return `Event` objects; otherwise normalized dictionaries.

        Returns:
            list[Any]: List of Event objects or normalized dictionaries.

        Notes:
            This method interacts with the underlying HotLog service to fetch events
            associated with the current timeline. The events are returned in chronological order,
            with the most recent events appearing last in the list. Memory out of the limit will be discarded
            in the HotLog layer (but persistent in the Persistence layer). Memory in persistence cannot be retrieved
            via this method.

        Scope Level Filtering:
            - level="scope" or None:   entire memory scope / timeline (current behavior).
            - level="session": only events for this session_id.
            - level="run":     only events for this run_id.
            - level="user":    only events for this user/client.
            - level="org":     only events for this org.

        """
        # 1) Pull a reasonably large window from hotlog.
        buf = await self.hotlog.query(
            self.timeline_id,
            tenant=getattr(self, "memory_tenant", None),
            kinds=kinds,
            limit=self.hot_limit,
        )
        # 2) Apply scope-level filter
        scope = getattr(self, "scope", None)

        if level and level != "scope":
            buf = [e for e in buf if event_matches_level(e, scope, level=level)]

        # 3) Take the last `limit` events (buf is already chronological)
        return self.normalize_recent_output(buf[-limit:], return_event=return_event)

    # ------ Persistence Retrieval ------
    async def recent_persisted(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
        since: str | None = None,
        until: str | None = None,
        offset: int = 0,
        return_event: bool = True,
    ) -> list[Any]:
        """
        Retrieve events from the persistence layer (full history) for this timeline.

        This is a higher-latency, deeper history path than `recent()`.

        Examples:
            Query persisted events and return Event objects:
            ```python
            events = await context.memory().recent_persisted(limit=100)
            ```

            Query persisted events and return normalized dict payloads:
            ```python
            rows = await context.memory().recent_persisted(limit=100, return_event=False)
            ```

        Args:
            kinds: Optional event kinds filter.
            tags: Optional tag filter.
            limit: Maximum rows to return.
            level: Optional scope level filter.
            since: Optional lower timestamp bound.
            until: Optional upper timestamp bound.
            offset: Offset for pagination.
            return_event: If True return Event objects; otherwise dict payloads.

        Returns:
            list[Any]: Event rows or normalized dictionaries.
        """
        if not hasattr(self.persistence, "query_events"):
            # Fallback: no view API -> just use hotlog semantics as a degraded mode
            return await self.recent(
                kinds=kinds, limit=limit, level=level, return_event=return_event
            )

        scope = getattr(self, "scope", None)

        rows = await self.persistence.query_events(
            self.timeline_id,
            tenant=getattr(self, "memory_tenant", None),
            since=since,
            until=until,
            kinds=kinds,
            tags=tags,
            session_id=self.session_id if level == "session" else None,
            run_id=self.run_id if level == "run" else None,
            limit=None,
            offset=0,
        )
        if level and level != "scope":
            rows = [e for e in rows if event_matches_level(e, scope, level=level)]
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return self.normalize_recent_output(rows, return_event=return_event)

    # ------ Indices Search ------
    async def search(
        self: MemoryFacadeProtocol,
        *,
        query: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        use_embedding: bool = True,
        level: ScopeLevel | None = None,
        time_window: str | None = None,
        mode: SearchMode | None = None,
    ) -> list[Event]:
        """
        Search events using scoped indices with hotlog fallback.

        This method uses index-backed retrieval when available, then falls back
        to lexical filtering over recent events.

        Examples:
            Semantic search with default settings:
            ```python
            events = await context.memory().search(query="deployment failure", limit=10)
            ```

            Lexical-only search for tagged events:
            ```python
            events = await context.memory().search(
                query="timeout",
                tags=["tool", "error"],
                use_embedding=False,
                level="run",
            )
            ```

        Args:
            query: Optional query string. If None, returns filtered recent events.
            kinds: Optional event kind filters.
            tags: Optional required tags.
            limit: Maximum number of events to return.
            use_embedding: If True, prefer index-backed semantic/hybrid search.
            level: Optional scope level constraint.
            time_window: Optional backend time-window hint.
            mode: Optional explicit backend search mode.

        Returns:
            list[Event]: Matching events in relevance/fallback order.
        """
        # --- 1) Try index-backed search (ScopedIndices) ------------------
        if use_embedding and getattr(self, "scoped_indices", None) is not None:
            idx: ScopedIndices = self.scoped_indices
            if idx is not None and idx.backend is not None:
                filters: dict[str, Any] = {}

                # Let the backend do kind/tag filtering via metadata
                if kinds:
                    # `kind` is scalar in events, but filters allow list semantics.
                    filters["kind"] = kinds
                if tags:
                    # tags stored as a list, and GenericVectorSearchBackend
                    # supports list↔list intersection semantics.
                    filters["tags"] = tags

                # Decide effective mode
                if mode is not None:
                    eff_mode: SearchMode = mode
                else:
                    eff_mode = "semantic" if use_embedding else "lexical"

                items = await idx.search_events(
                    query=query,
                    top_k=limit,
                    filters=filters,
                    time_window=time_window,
                    level=level,
                    mode=eff_mode,
                )

                if items:
                    results = await self.fetch_events_for_search_results(items, corpus="event")
                    events = [r.event for r in results if r.event is not None]
                    if events:
                        return events

        # --- 2) Fallback: lexical search over recent HotLog -------------
        events = await self.recent(
            kinds=kinds,
            limit=limit,
            level=level,
        )

        if tags:
            want = set(tags)
            events = [e for e in events if want.issubset(set(e.tags or []))]

        if not query:
            return events

        query_l = query.lower()
        lexical_hits = [e for e in events if (e.text or "").lower().find(query_l) >= 0]

        return lexical_hits or events

    # ------ Convenience Wrappers ------
    async def recent_events(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        overfetch: int = 5,
        level: ScopeLevel | None = None,
        use_persistence: bool = False,
        return_event: bool = True,
    ) -> list[Any]:
        """
        Convenience wrapper to fetch recent events with tag filtering.

        - By default uses HotLog (`use_persistence=False`).
        - Can optionally use persistence for deeper history.

        Examples:
            Return Event objects:
            ```python
            events = await context.memory().recent_events(kinds=["chat.turn"], limit=30)
            ```

            Return normalized dict payloads:
            ```python
            rows = await context.memory().recent_events(
                kinds=["chat.turn"], limit=30, return_event=False
            )
            ```

        Args:
            kinds: Optional event kinds filter.
            tags: Optional tag filter.
            limit: Final output size cap.
            overfetch: Over-fetch factor used before post-filtering by tags.
            level: Optional scope level filter.
            use_persistence: If True use persistence query path.
            return_event: If True return Event objects; otherwise dict payloads.

        Returns:
            list[Any]: Event rows or normalized dictionaries.
        """
        fetch_n = limit if not tags else max(limit * overfetch, 100)

        if use_persistence:
            evts = await self.recent_persisted(
                kinds=kinds,
                tags=None,  # we re-apply tags below for consistent semantics
                limit=fetch_n,
                level=level,
                return_event=True,
            )
        else:
            evts = await self.recent(
                kinds=kinds,
                limit=fetch_n,
                level=level,
                return_event=True,
            )

        if tags:
            want = set(tags)
            evts = [e for e in evts if want.issubset(set(e.tags or []))]

        return self.normalize_recent_output(evts[-limit:], return_event=return_event)

    async def recent_data(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
    ) -> list[Any]:
        """
        Deprecated helper returning recent event payloads (`data`/`text`).

        Prefer `recent(..., return_event=False)` or `recent_events(..., return_event=False)`.

        Examples:
            Legacy data-only retrieval:
            ```python
            payloads = await context.memory().recent_data(kinds=["tool_result"], limit=20)
            ```

        Args:
            kinds: A list of event kinds to filter by. Defaults to None.
            tags: A list of tags to filter events by. Defaults to None.
            level: Optional scope level filter.
            limit: The maximum number of events to retrieve. Defaults to 50.

        Returns:
            list[Any]: Data payloads (fallback to parsed text / text string).
        """
        warnings.warn(
            "recent_data() is deprecated and will be removed in a future version. "
            "Use recent(..., return_event=False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        evts = await self.recent_events(
            kinds=kinds,
            tags=tags,
            limit=limit,
            level=level,
            use_persistence=False,
            return_event=True,
        )

        out: list[Any] = []
        for e in evts:
            if e.data is not None:
                out.append(e.data)
            elif e.text:
                t = e.text.strip()
                if (t.startswith("{") and t.endswith("}")) or (
                    t.startswith("[") and t.endswith("]")
                ):
                    try:
                        out.append(json.loads(t))
                        continue
                    except Exception:
                        pass
                out.append(e.text)
        return out

    async def fetch_events_for_search_results(
        self,
        scored_items: list[ScoredItem],
        corpus: str = "event",
    ) -> list[EventSearchResult]:
        """
        Resolve scored search items into `EventSearchResult` rows.

        This method filters items by corpus, resolves events from hotlog and
        persistence, and preserves the original scoring metadata.

        Examples:
            Resolve event search results from an index query:
            ```python
            items = await context.memory().scoped_indices.search_events(query="policy")
            rows = await context.memory().fetch_events_for_search_results(items)
            ```

            Resolve only a custom corpus:
            ```python
            rows = await context.memory().fetch_events_for_search_results(
                scored_items,
                corpus="event",
            )
            ```

        Args:
            scored_items: Scored items returned by the search backend.
            corpus: Corpus name to resolve (defaults to `"event"`).

        Returns:
            list[EventSearchResult]: Scored rows with optional resolved events.
        """

        # Filter to event corpus
        event_items = [item for item in scored_items if item.corpus == corpus]
        if not event_items:
            return []

        ids = [it.item_id for it in event_items]

        # 1) Try hotlog first
        recent = await self.hotlog.query(
            self.timeline_id,
            tenant=getattr(self, "memory_tenant", None),
            kinds=None,
            limit=1,
            # limit=self.hot_limit,
        )
        by_id: dict[str, Event] = {e.event_id: e for e in recent if e.event_id in ids}

        # 2) Fallback to persistence for misses
        missing_ids = [eid for eid in ids if eid not in by_id]
        if missing_ids and hasattr(self.persistence, "get_events_by_ids"):
            persisted = await self.persistence.get_events_by_ids(
                self.timeline_id,
                missing_ids,
                tenant=getattr(self, "memory_tenant", None),
            )
            for e in persisted:
                by_id[e.event_id] = e

        # 3) Build results
        results: list[EventSearchResult] = []
        for item in event_items:
            evt = by_id.get(item.item_id)
            results.append(EventSearchResult(item=item, event=evt))
        return results

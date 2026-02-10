from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

from aethergraph.contracts.storage.search_backend import ScoredItem
from aethergraph.services.memory.facade.utils import event_matches_level
from aethergraph.services.scope.scope import ScopeLevel

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event

    from .types import MemoryFacadeInterface


class EventSearchResult(NamedTuple):
    item: ScoredItem
    event: Event | None

    @property
    def score(self) -> float:
        return self.item.score


class RetrievalMixin:
    """Methods for retrieving events and values."""

    async def get_event(self, event_id: str) -> Event | None:
        """
        Retrieve a specific event by its ID.

        This method fetches an event corresponding to the provided event ID.

        Args:
            event_id: The unique identifier of the event to retrieve.

        Returns:
            Event | None: The event object if found; otherwise, None.

        Notes:
            This method interacts with the underlying Persistence service to fetch
            the event associated with the current timeline. If no event is found
            with the given ID, it returns None.
        """
        # 1) Try hotlog
        recent = await self.hotlog.recent(
            self.timeline_id,
            kinds=None,
            limit=self.hot_limit,
        )
        for e in recent:
            if e.event_id == event_id:
                return e

        # 2) Fallback to persistence
        if hasattr(self.persistence, "get_events_by_ids"):
            events = await self.persistence.get_events_by_ids(self.timeline_id, [event_id])
            return events[0] if events else None

        return None

    # ------ Hotlog Retrieval ------
    async def recent(
        self: MemoryFacadeInterface,
        *,
        kinds: list[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
    ) -> list[Event]:
        """
        Retrieve recent events.

        This method fetches a list of recent events, optionally filtered by kinds.

        Args:
            kinds: A list of event kinds to filter by. Defaults to None.
            limit: The maximum number of events to retrieve. Defaults to 50.
            level: Optional scope level to filter events by. If provided, only events associated with the specified scope level will be returned.

        Returns:
            list[Event]: A list of recent events.

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
        buf = await self.hotlog.recent(
            self.timeline_id,
            kinds=kinds,
            limit=self.hot_limit,
        )

        # 2) Apply scope-level filter
        scope = getattr(self, "scope", None)

        if level and level != "scope":
            buf = [e for e in buf if event_matches_level(e, scope, level=level)]

        # 3) Take the last `limit` events (buf is already chronological)
        return buf[-limit:]

    # ------ Persistence Retrieval ------
    async def recent_persisted(
        self: MemoryFacadeInterface,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
        since: str | None = None,
        until: str | None = None,
        offset: int = 0,
    ) -> list[Event]:
        """
        Retrieve events from the persistence layer (full history) for this timeline.

        This is a higher-latency, deeper history path than `recent()`.
        """
        if not hasattr(self.persistence, "query_events_view"):
            # Fallback: no view API -> just use hotlog semantics as a degraded mode
            return await self.recent(kinds=kinds, limit=limit, level=level)

        scope = getattr(self, "scope", None)

        rows = await self.persistence.query_events_view(
            scope_id=self.timeline_id,
            scope=scope,
            level=level,
            since=since,
            until=until,
            kinds=kinds,
            tags=tags,
            limit=limit,
            offset=offset,
        )
        return rows

    # ------ Indices Search ------
    async def search(
        self: MemoryFacadeInterface,
        *,
        query: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        use_embedding: bool = True,
        level: ScopeLevel | None = None,
        time_window: str | None = None,
    ) -> list[Event]:
        """
        Search for events based on a query.

        This method searches for events that match a query, optionally filtered by kinds and tags.
        Note that this implementation currently performs a lexical search. Embedding-based search
        is planned for future development.

        Args:
            query: The search query string.
            kinds: A list of event kinds to filter by. Defaults to None.
            tags: A list of tags to filter events by. Defaults to None.
            limit: The maximum number of events to retrieve. Defaults to 100.
            use_embedding: Whether to use embedding-based search. Defaults to True.
            level: Optional scope level to filter events by (e.g., "session", "run", "user", "org"). If provided, the search will be constrained to events associated with the specified scope level.
            time_window: Optional time window for the search (e.g., "last_7_days"). Defaults to None.

        Returns:
            list[Event]: A list of events matching the query.

        Notes:
            This method retrieves recent events using the `recent()` method and filters them
            based on the provided tags. It performs a simple lexical search on the event text.
            Embedding-based search functionality is not yet implemented.

            Memory out of the limit will be discarded in the HotLog layer (but persistent in the Persistence layer).
            Memory in persistence cannot be retrieved via this method.

        """
        # --- 1) Try index-backed search (ScopedIndices) ------------------
        if use_embedding and getattr(self, "scoped_indices", None) is not None:
            idx = self.scoped_indices
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

                items = await idx.search_events(
                    query=query,
                    top_k=limit,
                    filters=filters,
                    time_window=time_window,
                    level=level,
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
        self: MemoryFacadeInterface,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        overfetch: int = 5,
        level: ScopeLevel | None = None,
        use_persistence: bool = False,
    ) -> list[Event]:
        """
        Convenience wrapper to fetch recent events with tag filtering.

        - By default uses HotLog (`use_persistence=False`).
        - Can optionally use persistence for deeper history.
        """
        fetch_n = limit if not tags else max(limit * overfetch, 100)

        if use_persistence:
            evts = await self.recent_persisted(
                kinds=kinds,
                tags=None,  # we re-apply tags below for consistent semantics
                limit=fetch_n,
                level=level,
            )
        else:
            evts = await self.recent(
                kinds=kinds,
                limit=fetch_n,
                level=level,
            )

        if tags:
            want = set(tags)
            evts = [e for e in evts if want.issubset(set(e.tags or []))]

        return evts[-limit:]

    async def recent_data(
        self: MemoryFacadeInterface,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level: ScopeLevel | None = None,
    ) -> list[Any]:
        """
        Retrieve recent event data.

        This method fetches the data or text of recent events, optionally filtered by kinds and tags.
        Unlike `recent()`, which returns full Event objects, this method extracts and returns only the
        data or text content of the events. This is useful for scenarios where only the event payloads are needed.

        Args:
            kinds: A list of event kinds to filter by. Defaults to None.
            tags: A list of tags to filter events by. Defaults to None.
            limit: The maximum number of events to retrieve. Defaults to 50.

        Returns:
            list[Any]: A list of event data or text.

        Notes:
            This method first retrieves recent events using the `recent()` method and then filters them
            based on the provided tags. It extracts the `data` attribute if available; otherwise, it
            attempts to parse the `text` attribute as JSON. If parsing fails, the raw text is returned.

            Memory out of the limit will be discarded in the HotLog layer (but persistent in the Persistence layer).
            Memory in persistence cannot be retrieved via this method.
        """
        evts = await self.recent_events(
            kinds=kinds,
            tags=tags,
            limit=limit,
            level=level,
            use_persistence=False,
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
        Given a list of ScoredItems from a search, fetch the corresponding Event objects.
        """

        # Filter to event corpus
        event_items = [item for item in scored_items if item.corpus == corpus]
        if not event_items:
            return []

        ids = [it.item_id for it in event_items]

        # 1) Try hotlog first
        recent = await self.hotlog.recent(
            self.timeline_id,
            kinds=None,
            limit=1,
            # limit=self.hot_limit,
        )
        by_id: dict[str, Event] = {e.event_id: e for e in recent if e.event_id in ids}

        # 2) Fallback to persistence for misses
        missing_ids = [eid for eid in ids if eid not in by_id]
        if missing_ids and hasattr(self.persistence, "get_events_by_ids"):
            persisted = await self.persistence.get_events_by_ids(self.timeline_id, missing_ids)
            for e in persisted:
                by_id[e.event_id] = e

        # 3) Build results
        results: list[EventSearchResult] = []
        for item in event_items:
            evt = by_id.get(item.item_id)
            results.append(EventSearchResult(item=item, event=evt))
        return results

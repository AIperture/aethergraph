from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from aethergraph.contracts.storage.search_backend import ScoredItem, SearchMode
from aethergraph.services.indices.scoped_indices import ScopedIndices
from aethergraph.services.memory.facade.utils import event_matches_level

from .retrieval import EventSearchResult

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event, MemoryFacadeProtocol


class ReadMixin:
    async def _call_query_backend(self, method, /, *args, **kwargs):
        try:
            return await method(*args, **kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" not in message:
                raise
            signature = inspect.signature(method)
            filtered_kwargs = {
                key: value for key, value in kwargs.items() if key in signature.parameters
            }
            return await method(*args, **filtered_kwargs)

    async def get_event(self: MemoryFacadeProtocol, event_id: str) -> Event | None:
        """Fetch a single event by id.

        Looks in durable persistence first, then falls back to the hot log.

        Args:
            event_id: Identifier of the event to fetch.

        Returns:
            Event | None: The matching event, or ``None`` if not found.
        """
        events = await self.persistence.get_events_by_ids(
            self.timeline_id,
            [event_id],
            tenant=getattr(self, "memory_tenant", None),
        )
        if events:
            return events[0]
        recent = await self.hotlog.query(
            self.timeline_id,
            tenant=getattr(self, "memory_tenant", None),
            limit=self.hot_limit,
        )
        for event in recent:
            if event.event_id == event_id:
                return event
        return None

    async def query_events(
        self: MemoryFacadeProtocol,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        level=None,
        use_persistence: bool = False,
        since: str | None = None,
        until: str | None = None,
        offset: int = 0,
        return_event: bool = True,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        order_dir: str = "desc",
    ) -> list[Any]:
        """Query events by structured filters (kinds, tags, scope, time, source).

        Reads from the hot log by default, or from durable persistence when
        ``use_persistence=True``. Results are scope-filtered by ``level``, then
        offset/limited and returned newest-first (``order_dir="desc"``) unless
        overridden.

        Args:
            kinds: Restrict to these event kinds.
            tags: Restrict to events carrying all of these tags.
            limit: Maximum number of events to return.
            level: Scope level to filter by (``"scope"``, ``"session"``, ``"run"``,
                ``"user"``, ``"org"``).
            use_persistence: Query durable storage instead of the hot log.
            since: Lower time bound (ISO timestamp).
            until: Upper time bound (ISO timestamp).
            offset: Number of leading results to skip.
            return_event: Return ``Event`` objects when ``True``, else plain dicts.
            session_id: Filter by session id (inferred from scope when ``level="session"``).
            run_id: Filter by run id (inferred from scope when ``level="run"``).
            agent_id: Filter by agent id.
            client_id: Filter by client id.
            graph_id: Filter by graph id.
            node_id: Filter by node id.
            topic: Filter by event topic.
            tool: Filter by tool topic.
            order_dir: ``"desc"`` (newest first, default) or ``"asc"``.

        Returns:
            list: Events (or dicts) matching the query.
        """
        order_dir = "asc" if str(order_dir).lower() == "asc" else "desc"
        scope = getattr(self, "scope", None)
        eff_session = (
            session_id
            if session_id is not None
            else (self.session_id if level == "session" else None)
        )
        eff_run = run_id if run_id is not None else (self.run_id if level == "run" else None)
        if use_persistence:
            rows = await self._call_query_backend(
                self.persistence.query_events,
                self.timeline_id,
                tenant=getattr(self, "memory_tenant", None),
                since=since,
                until=until,
                kinds=kinds,
                tags=tags,
                session_id=eff_session,
                run_id=eff_run,
                agent_id=agent_id,
                client_id=client_id,
                graph_id=graph_id,
                node_id=node_id,
                topic=topic,
                tool=tool,
                limit=None,
                offset=0,
                order_dir=order_dir,
            )
        else:
            rows = await self._call_query_backend(
                self.hotlog.query,
                self.timeline_id,
                tenant=getattr(self, "memory_tenant", None),
                kinds=kinds,
                tags=tags,
                since=since,
                until=until,
                session_id=eff_session,
                run_id=eff_run,
                agent_id=agent_id,
                client_id=client_id,
                graph_id=graph_id,
                node_id=node_id,
                topic=topic,
                tool=tool,
                limit=self.hot_limit,
                offset=0,
                order_dir=order_dir,
            )
        if level and level != "scope":
            rows = [event for event in rows if event_matches_level(event, scope, level=level)]
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return self.normalize_recent_output(rows, return_event=return_event)

    async def search_events(
        self: MemoryFacadeProtocol,
        *,
        query: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        use_embedding: bool = True,
        level=None,
        time_window: str | None = None,
        mode: SearchMode | None = None,
    ) -> list[Event]:
        """Search events semantically/lexically, falling back to structured query.

        When an index backend is available and ``use_embedding=True``, runs a
        semantic (or ``mode``-selected) search over indexed events and hydrates the
        matching ``Event`` objects. Otherwise falls back to a persistence-backed
        :meth:`query_events` plus a lexical substring filter on ``query``.

        Args:
            query: Free-text query. When empty, returns the most recent matches.
            kinds: Restrict to these event kinds.
            tags: Restrict to events carrying these tags.
            limit: Maximum number of results.
            use_embedding: Use the semantic index when available.
            level: Scope level to filter by.
            time_window: Optional relative window (e.g. ``"7d"``, ``"24h"``).
            mode: Optional explicit search mode (e.g. ``"semantic"``, ``"lexical"``).

        Returns:
            list[Event]: Matching events.
        """
        if use_embedding and getattr(self, "scoped_indices", None) is not None:
            idx: ScopedIndices = self.scoped_indices
            if idx is not None and idx.backend is not None:
                filters: dict[str, Any] = {}
                if kinds:
                    filters["kind"] = kinds
                if tags:
                    filters["tags"] = tags
                eff_mode: SearchMode = (
                    mode if mode is not None else ("semantic" if use_embedding else "lexical")
                )
                items = await idx.search_events(
                    query=query or "",
                    top_k=limit,
                    filters=filters,
                    time_window=time_window,
                    level=level,
                    mode=eff_mode,
                )
                if items:
                    results = await self.fetch_events_for_search_results(items, corpus="event")
                    events = [row.event for row in results if row.event is not None]
                    if events:
                        return events
        events = await self.query_events(
            kinds=kinds,
            tags=tags,
            limit=limit,
            level=level,
            use_persistence=True,
            return_event=True,
        )
        if not query:
            return events
        query_l = query.lower()
        lexical_hits = [event for event in events if (event.text or "").lower().find(query_l) >= 0]
        return lexical_hits or events

    async def fetch_events_for_search_results(
        self: MemoryFacadeProtocol,
        scored_items: list[ScoredItem],
        corpus: str = "event",
    ) -> list[EventSearchResult]:
        """Hydrate full ``Event`` objects for a list of scored search hits.

        Given ``ScoredItem`` results from the search backend, resolves the
        corresponding events from the hot log and durable persistence.

        Args:
            scored_items: Scored items returned by the search backend.
            corpus: Corpus to hydrate from (defaults to ``"event"``).

        Returns:
            list[EventSearchResult]: Pairs of scored item and resolved event
            (``event`` is ``None`` when it could not be found).
        """
        event_items = [item for item in scored_items if item.corpus == corpus]
        if not event_items:
            return []
        ids = [item.item_id for item in event_items]
        recent = await self.hotlog.query(
            self.timeline_id,
            tenant=getattr(self, "memory_tenant", None),
            limit=self.hot_limit,
        )
        by_id: dict[str, Event] = {
            event.event_id: event for event in recent if event.event_id in ids
        }
        missing_ids = [event_id for event_id in ids if event_id not in by_id]
        if missing_ids:
            persisted = await self.persistence.get_events_by_ids(
                self.timeline_id,
                missing_ids,
                tenant=getattr(self, "memory_tenant", None),
            )
            for event in persisted:
                by_id[event.event_id] = event
        return [EventSearchResult(item=item, event=by_id.get(item.item_id)) for item in event_items]

    async def get_latest_state(
        self: MemoryFacadeProtocol,
        key: str,
        *,
        tags=None,
        level=None,
        use_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> Any | None:
        """Return the most recent value stored for a state ``key``.

        Convenience wrapper over :meth:`query_events` for ``state.snapshot`` events.

        Args:
            key: State key to look up.
            tags: Extra tags to require in addition to the state tags.
            level: Scope level to filter by.
            use_persistence: Read durable storage instead of the hot log.
            kind: State event kind (defaults to ``"state.snapshot"``).

        Returns:
            Any | None: The stored value, or ``None`` if no snapshot exists.
        """
        record = await self.get_latest_state_record(
            key,
            tags=tags,
            level=level,
            use_persistence=use_persistence,
            kind=kind,
        )
        return None if record is None else record["value"]

    async def get_latest_state_record(
        self: MemoryFacadeProtocol,
        key: str,
        *,
        tags=None,
        level=None,
        use_persistence: bool = False,
        kind: str = "state.snapshot",
    ) -> dict[str, Any] | None:
        """Return the latest state value with its enclosing snapshot revision.

        The record exposes only the stored value, numeric revision metadata,
        and Event identity needed by the canonical Agent-state handle. It does
        not create another state authority or reinterpret the state payload.

        Examples:
            Read a revisioned state record:
                ```python
                record = await memory.get_latest_state_record("agent:writer")
                revision = 0 if record is None else record["revision"]
                ```

            Read durable persistence explicitly:
                ```python
                record = await memory.get_latest_state_record(
                    "agent:writer",
                    use_persistence=True,
                )
                ```

        Args:
            key: Logical state key to look up.
            tags: Extra tags required in addition to the state tags.
            level: Scope level used to filter state snapshots.
            use_persistence: Read durable persistence instead of the hot log.
            kind: State Event kind, normally `state.snapshot`.

        Returns:
            dict[str, Any] | None: Latest value, revision, metadata, and Event
            identity, or `None` when no snapshot exists.

        Notes:
            Legacy snapshots without numeric revision metadata are reported at
            revision zero and acquire revision one on their next commit.
        """

        events = await self.query_events(
            kinds=[kind],
            tags=["state", f"state:{key}", *(list(tags or []))],
            limit=1,
            level=level,
            use_persistence=use_persistence,
            return_event=True,
        )
        if not events:
            return None
        event = events[0]
        data = dict(event.data or {})
        meta = dict(data.get("meta") or {})
        raw_revision = meta.get("revision", 0)
        try:
            revision = int(raw_revision)
        except (TypeError, ValueError):
            revision = 0
        return {
            "value": data.get("value"),
            "revision": max(0, revision),
            "meta": meta,
            "event_id": str(event.event_id or ""),
        }

    async def list_state_history(
        self: MemoryFacadeProtocol,
        key: str,
        *,
        tags=None,
        limit: int = 50,
        level=None,
        kind: str = "state.snapshot",
        use_persistence: bool = False,
    ) -> list[Event]:
        """Return the history of snapshots recorded for a state ``key``.

        Args:
            key: State key to look up.
            tags: Extra tags to require in addition to the state tags.
            limit: Maximum number of snapshots to return.
            level: Scope level to filter by.
            kind: State event kind (defaults to ``"state.snapshot"``).
            use_persistence: Read durable storage instead of the hot log.

        Returns:
            list[Event]: Snapshot events, newest last.
        """
        return await self.query_events(
            kinds=[kind],
            tags=["state", f"state:{key}", *(list(tags or []))],
            limit=limit,
            level=level,
            use_persistence=use_persistence,
            return_event=True,
        )

    async def search_state(
        self: MemoryFacadeProtocol,
        query: str,
        *,
        key: str | None = None,
        tags=None,
        top_k: int = 10,
        time_window: str | None = None,
        created_at_min: float | None = None,
        created_at_max: float | None = None,
    ) -> list[EventSearchResult]:
        """Semantically search recorded state snapshots.

        Requires a configured index backend; returns an empty list otherwise.

        Args:
            query: Free-text query.
            key: Optional state key to restrict the search to.
            tags: Extra tags to require.
            top_k: Maximum number of results.
            time_window: Optional relative window (e.g. ``"7d"``).
            created_at_min: Optional lower bound as a UNIX timestamp.
            created_at_max: Optional upper bound as a UNIX timestamp.

        Returns:
            list[EventSearchResult]: Scored snapshot hits with resolved events.
        """
        scoped = getattr(self, "scoped_indices", None)
        if scoped is None or scoped.backend is None:
            return []

        filter_tags: list[str] = ["state"]
        if key:
            filter_tags.append(f"state:{key}")
        filter_tags.extend(list(tags or []))

        filters: dict[str, Any] = {
            "kind": "state.snapshot",
            "tags": filter_tags,
        }
        scored = await scoped.search_events(
            query=query,
            filters=filters,
            top_k=top_k,
            time_window=time_window,
            created_at_min=created_at_min,
            created_at_max=created_at_max,
        )
        return await self.fetch_events_for_search_results(scored, corpus="event")

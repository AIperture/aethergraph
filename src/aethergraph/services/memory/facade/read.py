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
    ) -> list[Any]:
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
            )
        if level and level != "scope":
            rows = [event for event in rows if event_matches_level(event, scope, level=level)]
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[-limit:] if not use_persistence else rows[:limit]
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
        return (events[-1].data or {}).get("value")

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

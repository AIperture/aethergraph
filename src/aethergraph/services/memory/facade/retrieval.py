from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event

    from .types import MemoryFacadeInterface


class RetrievalMixin:
    """Methods for retrieving events and values."""

    async def recent(
        self: MemoryFacadeInterface, *, kinds: list[str] | None = None, limit: int = 50
    ) -> list[Event]:
        """
        Retrieve recent events.

        This method fetches a list of recent events, optionally filtered by kinds.

        Args:
            kinds: A list of event kinds to filter by. Defaults to None.
            limit: The maximum number of events to retrieve. Defaults to 50.

        Returns:
            list[Event]: A list of recent events.

        Notes:
            This method interacts with the underlying HotLog service to fetch events
            associated with the current timeline. The events are returned in chronological order,
            with the most recent events appearing last in the list. Memory out of the limit will be discarded
            in the HotLog layer (but persistent in the Persistence layer). Memory in persistence cannot be retrieved
            via this method.
        """
        return await self.hotlog.recent(self.timeline_id, kinds=kinds, limit=limit)

    async def recent_data(
        self: MemoryFacadeInterface,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
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
        evts = await self.recent(kinds=kinds, limit=limit)
        if tags:
            want = set(tags)
            evts = [e for e in evts if want.issubset(set(e.tags or []))]

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

    async def search(
        self: MemoryFacadeInterface,
        *,
        query: str,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
        use_embedding: bool = True,
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

        Returns:
            list[Event]: A list of events matching the query.

        Notes:
            This method retrieves recent events using the `recent()` method and filters them
            based on the provided tags. It performs a simple lexical search on the event text.
            Embedding-based search functionality is not yet implemented.

            Memory out of the limit will be discarded in the HotLog layer (but persistent in the Persistence layer).
            Memory in persistence cannot be retrieved via this method.
        """
        events = await self.recent(kinds=kinds, limit=limit)
        if tags:
            want = set(tags)
            events = [e for e in events if want.issubset(set(e.tags or []))]

        query_l = query.lower()
        lexical_hits = [e for e in events if (e.text or "").lower().find(query_l) >= 0]

        if not use_embedding:
            return lexical_hits or events

        # Placeholder for future embedding search logic
        # if not (self.llm and any(e.embedding for e in events)): return lexical_hits or events
        # ... logic ...
        return lexical_hits or events

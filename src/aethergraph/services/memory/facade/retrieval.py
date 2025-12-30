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
        return await self.hotlog.recent(self.timeline_id, kinds=kinds, limit=limit)

    async def recent_data(
        self: MemoryFacadeInterface,
        *,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
    ) -> list[Any]:
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

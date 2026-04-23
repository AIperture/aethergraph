from datetime import UTC, datetime

from aethergraph.contracts.services.kv import AsyncKV
from aethergraph.contracts.services.memory import Event, HotLog, MemoryTenantFilter
from aethergraph.services.memory.storage_filters import event_matches_filters, event_time

# No specific backend is required; we use AsyncKV for storage.


def kv_hot_key(timeline_id: str) -> str:
    return f"mem:{timeline_id}:hot"


class KVHotLog(HotLog):
    def __init__(self, kv: AsyncKV):
        self.kv = kv

    async def append(self, timeline_id: str, evt: Event, *, ttl_s: int, limit: int) -> None:
        key = kv_hot_key(timeline_id)
        buf = list((await self.kv.get(key, default=[])) or [])
        buf.append(evt.__dict__)  # store as dict for JSON-ability
        if len(buf) > limit:
            buf = buf[-limit:]
        await self.kv.set(key, buf, ttl_s=ttl_s)

    async def recent(
        self,
        timeline_id: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 50,
    ) -> list[Event]:
        buf = (await self.kv.get(kv_hot_key(timeline_id), default=[])) or []
        if kinds:
            buf = [e for e in buf if e.get("kind") in kinds]
        return [Event(**e) for e in buf[-limit:]]

    async def query(
        self,
        timeline_id: str,
        *,
        tenant: MemoryTenantFilter | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_dir: str = "desc",
    ) -> list[Event]:
        order_dir = "asc" if str(order_dir).lower() == "asc" else "desc"
        buf = (await self.kv.get(kv_hot_key(timeline_id), default=[])) or []

        # TODO: optimize filtering by pushing down to storage layer if supported (e.g. Redis streams with consumer groups could handle this efficiently)
        # For we do filtering in-memory here; this is fine for memory-level hotlog with about 100 events, but would not scale for larger datasets or more complex queries
        filtered: list[tuple[int, Event]] = []
        for seq, row in enumerate(buf):
            if not event_matches_filters(
                row,
                tenant=tenant,
                kinds=kinds,
                tags=tags,
                since=since,
                until=until,
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                client_id=client_id,
                graph_id=graph_id,
                node_id=node_id,
                topic=topic,
                tool=tool,
            ):
                continue
            filtered.append((seq, Event(**row)))

        min_dt = datetime.min.replace(tzinfo=UTC)
        filtered.sort(
            key=lambda item: (event_time(item[1]) or min_dt, item[0]),
            reverse=order_dir == "desc",
        )
        if offset:
            filtered = filtered[offset:]
        if limit is not None:
            filtered = filtered[:limit]
        return [event for _, event in filtered]

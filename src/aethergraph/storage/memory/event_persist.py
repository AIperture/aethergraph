from dataclasses import asdict
import hashlib
from typing import Any

from aethergraph.contracts.services.memory import Event, MemoryTenantFilter, Persistence
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.memory.facade.utils import event_matches_level
from aethergraph.services.memory.storage_filters import (
    event_matches_filters,
    summary_matches_filters,
)
from aethergraph.services.scope.scope import Scope, ScopeLevel


class EventLogPersistence(Persistence):
    """
    Persistence built on top of generic EventLog + DocStore.

    - append_event: logs Event rows into EventLog with timeline_id partitioning.
    - save_json / load_json: store arbitrary JSON in DocStore using memdoc:// URIs.
    """

    def __init__(
        self,
        *,
        log: EventLog,
        docs: DocStore,
        uri_prefix: str = "memdoc://",
    ):
        self._log = log
        self._docs = docs
        self._prefix = uri_prefix

    def _doc_id_from_uri(self, uri: str) -> str:
        if uri.startswith(self._prefix):
            return uri[len(self._prefix) :]
        h = hashlib.sha1(uri.encode("utf-8")).hexdigest()
        return f"memdoc/{h}"

    def _uri_from_doc_id(self, doc_id: str) -> str:
        if doc_id.startswith("memdoc://"):
            return doc_id
        return f"{self._prefix}{doc_id}"

    def _event_from_row(self, row: dict[str, Any]) -> Event:
        allowed = Event.__dataclass_fields__.keys()
        payload = {k: v for k, v in row.items() if k in allowed}
        return Event(**payload)

    async def append_event(self, timeline_id: str, evt: Event) -> None:
        payload = asdict(evt)
        payload["_partition_scope_id"] = timeline_id
        payload["timeline_id"] = timeline_id
        payload.setdefault("kind", "memory")
        await self._log.append(payload)

    async def save_json(self, uri: str, obj: dict[str, Any]) -> str:
        doc_id = self._doc_id_from_uri(uri)
        await self._docs.put(doc_id, obj)
        return self._uri_from_doc_id(doc_id)

    async def load_json(self, uri: str) -> dict[str, Any]:
        doc_id = self._doc_id_from_uri(uri)
        doc = await self._docs.get(doc_id)
        if doc is None:
            raise FileNotFoundError(f"Memory JSON not found for URI: {uri}")
        return doc

    async def get_events_by_ids(
        self,
        timeline_id: str,
        event_ids: list[str],
        tenant: MemoryTenantFilter | None = None,
    ) -> list[Event]:
        if not event_ids:
            return []

        rows = await self._log.query(
            scope_id=timeline_id,
            since=None,
            until=None,
            kinds=None,
            tags=None,
            limit=None,
            offset=0,
            user_id=tenant.get("user_id") if tenant else None,
            org_id=tenant.get("org_id") if tenant else None,
        )

        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            eid = row.get("event_id")
            if eid:
                by_id[eid] = row

        result: list[Event] = []
        for eid in event_ids:
            row = by_id.get(eid)
            if row is None:
                continue
            if not event_matches_filters(row, tenant=tenant):
                continue
            result.append(self._event_from_row(row))
        return result

    async def query_events(
        self,
        timeline_id: str,
        *,
        tenant: MemoryTenantFilter | None = None,
        since: str | None = None,
        until: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_dir: str = "desc",
    ) -> list[Event]:
        order_dir = "asc" if str(order_dir).lower() == "asc" else "desc"
        rows = await self._log.query(
            scope_id=timeline_id,
            since=since,
            until=until,
            kinds=kinds,
            tags=tags,
            limit=limit,
            offset=offset,
            user_id=tenant.get("user_id") if tenant else None,
            org_id=tenant.get("org_id") if tenant else None,
            client_id=client_id,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            graph_id=graph_id,
            node_id=node_id,
            topic=topic,
            tool=tool,
            order_dir=order_dir,
        )
        events = [
            self._event_from_row(row)
            for row in rows
            if event_matches_filters(
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
            )
        ]
        return events

    async def query_events_view(
        self,
        timeline_id: str,
        *,
        scope: Scope | None = None,
        level: ScopeLevel | None = None,
        tenant: MemoryTenantFilter | None = None,
        since: str | None = None,
        until: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Event]:
        events = await self.query_events(
            timeline_id,
            tenant=tenant,
            since=since,
            until=until,
            kinds=kinds,
            tags=tags,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            limit=None,
            offset=0,
            order_dir="desc",
        )
        if level and level != "scope":
            events = [e for e in events if event_matches_level(e, scope, level=level)]
        if offset:
            events = events[offset:]
        if limit is not None:
            events = events[:limit]
        return events

    async def query_summaries(
        self,
        *,
        scope_id: str | None = None,
        timeline_id: str | None = None,
        tenant: MemoryTenantFilter | None = None,
        summary_tag: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        try:
            doc_ids = await self._docs.list()
        except TypeError:
            return []

        summaries: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            doc = await self._docs.get(doc_id)
            if not isinstance(doc, dict):
                continue
            if timeline_id is not None and doc.get("timeline_id") not in (None, timeline_id):
                continue
            if not summary_matches_filters(
                doc,
                tenant=tenant,
                scope_id=scope_id,
                summary_tag=summary_tag,
            ):
                continue
            summaries.append(doc)

        summaries.sort(key=lambda doc: str(doc.get("ts") or doc.get("created_at") or ""))
        if offset:
            summaries = summaries[offset:]
        if limit is not None:
            summaries = summaries[:limit]
        return summaries

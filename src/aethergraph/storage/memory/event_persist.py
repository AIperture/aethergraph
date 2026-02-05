from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

from aethergraph.contracts.services.memory import Event, Persistence
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog


class EventLogPersistence(Persistence):
    """
    Persistence built on top of generic EventLog + DocStore.

    - append_event: logs Event rows into EventLog with scope_id=run_id, kind="memory".
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

    # --------- helpers ---------
    def _doc_id_from_uri(self, uri: str) -> str:
        """
        Accepts:
          - memdoc://<id>  -> <id>
          - anything-else  -> hashed to a stable doc_id.
        """
        if uri.startswith(self._prefix):
            return uri[len(self._prefix) :]
        # fallback: hash to avoid weird chars
        h = hashlib.sha1(uri.encode("utf-8")).hexdigest()
        return f"memdoc/{h}"

    def _uri_from_doc_id(self, doc_id: str) -> str:
        if doc_id.startswith("memdoc://"):
            return doc_id
        return f"{self._prefix}{doc_id}"

    # --------- API ---------
    async def append_event(self, scope_id: str, evt: Event) -> None:
        payload = asdict(evt)
        payload.setdefault("scope_id", scope_id)
        payload.setdefault("kind", "memory")
        # you can add tags like ["mem"] if useful
        await self._log.append(payload)

    async def save_json(self, uri: str, obj: dict[str, Any]) -> str:
        doc_id = self._doc_id_from_uri(uri)
        # Let DocStore own where/how it writes
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
        scope_id: str,
        event_ids: list[str],
    ) -> list[Event]:
        """
        Fetch events for a given scope_id (timeline) by event_id.

        Implementation v0: use EventLog.query and filter in Python.
        For moderate timeline sizes and small event_ids lists, this is fine.
        Later, you can optimize by adding a direct get_many API on EventLog
        or indexing by (scope_id, event_id).
        """
        if not event_ids:
            return []

        # Fetch all events for the scope_id; TODO: add reasonable limits / paging
        rows = await self._log.query(
            scope_id=scope_id,
            since=None,
            until=None,
            kinds=None,
            tags=None,
            limit=None,
            offset=0,
        )

        by_id: dict[str, Event] = {}
        for row in rows:
            eid = row.get("event_id")
            if eid:
                by_id[eid] = row

        result: list[Event] = []
        for eid in event_ids:
            row = by_id.get(eid)
            if row is not None:
                result.append(Event(**row))
        return result

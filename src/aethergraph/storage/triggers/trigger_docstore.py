from __future__ import annotations

from datetime import datetime

from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.trigger_store import TriggerStore
from aethergraph.services.triggers.types import TriggerRecord

TRIGGER_DOC_PREFIX = "trigger:"


class DocTriggerStore(TriggerStore):
    """
    Simple TriggerStore implementation on top of DocStore.

    This uses DocStore.list() + in-Python filtering; it's fine for hundreds or
    a few thousand triggers.
    """

    def __init__(self, doc_store: DocStore):
        self._docs = doc_store

    async def create(self, trig: TriggerRecord) -> None:
        await self._docs.put(TRIGGER_DOC_PREFIX + trig.trigger_id, trig.to_dict())

    async def update(self, trig: TriggerRecord) -> None:
        await self._docs.put(TRIGGER_DOC_PREFIX + trig.trigger_id, trig.to_dict())

    async def get(self, trigger_id: str) -> TriggerRecord | None:
        doc = await self._docs.get(TRIGGER_DOC_PREFIX + trigger_id)
        return TriggerRecord.from_dict(doc) if doc else None

    async def delete(self, trigger_id: str) -> None:
        await self._docs.delete(TRIGGER_DOC_PREFIX + trigger_id)

    async def _iter_all(self) -> list[TriggerRecord]:
        ids = await self._docs.list()
        out: list[TriggerRecord] = []
        for doc_id in ids:
            if not doc_id.startswith(TRIGGER_DOC_PREFIX):
                continue
            doc = await self._docs.get(doc_id)
            if not doc:
                continue
            try:
                out.append(TriggerRecord.from_dict(doc))
            except Exception:
                # swallow malformed trigger docs
                continue
        return out

    async def list_active(self) -> list[TriggerRecord]:
        all_trigs = await self._iter_all()
        return [t for t in all_trigs if t.active]

    async def list_due(self, now: datetime) -> list[TriggerRecord]:
        """
        Return time-based triggers that are active and due as of 'now'.
        Event-based triggers (kind='event') are excluded; they are fired on demand.
        """
        all_trigs = await self.list_active()
        out: list[TriggerRecord] = []
        for t in all_trigs:
            if t.kind == "event":
                continue
            if t.next_fire_at is None:
                continue
            if t.next_fire_at <= now:
                out.append(t)
        return out

    async def list_by_event_key(self, event_key) -> list[TriggerRecord]:
        """
        Return event-based triggers that match the given event_key and are active.
        This is used to find triggers to fire when an event occurs.
        """
        all_trigs = await self.list_active()
        return [t for t in all_trigs if t.kind == "event" and t.event_key == event_key]

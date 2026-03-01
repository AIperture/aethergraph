from __future__ import annotations

from datetime import datetime

from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.trigger_store import TriggerStore
from aethergraph.services.triggers.types import TriggerRecord

TRIGGER_DOC_PREFIX = "trigger:"


def _matches_tenant(
    t: TriggerRecord,
    org_id: str | None,
    user_id: str | None,
    client_id: str | None,
) -> bool:
    if org_id is not None and t.org_id != org_id:
        return False

    # “User” for triggers = user_id OR client_id
    if user_id is not None:  # noqa: SIM102
        if not (t.user_id == user_id or t.client_id == user_id):
            return False

    if client_id is not None and t.client_id != client_id:  # noqa: SIM103
        return False

    return True


class DocTriggerStore(TriggerStore):
    """
    Simple TriggerStore implementation on top of DocStore.

    This uses DocStore.list() + in-Python filtering; it's fine for hundreds or
    a few thousand triggers.

    TODO: If we have a very large number of triggers and this becomes a bottleneck, we can add secondary indexes to DocStore to make the queries more efficient.
    We also need to thread org_id/user_id/client_id through the trigger engine to make this efficient for event-based triggers.
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
        """
        This should not be scope-filtered, since the TriggerEngine needs to see all active triggers regardless of tenant.
        """
        return await self.list_all(active=True)

    async def list_due(self, now: datetime) -> list[TriggerRecord]:
        """
        Return time-based triggers that are active and due as of 'now'.
        Event-based triggers (kind='event') are excluded; they are fired on demand.

        This should not be scope-filtered, since the TriggerEngine needs to see all due triggers regardless of tenant.
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

    async def list_all(
        self,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        kind: str | None = None,
        active: bool | None = None,
    ) -> list[TriggerRecord]:
        """
        General-purpose list with simple in-Python filtering.

        This is meant for API / service use (per-tenant listing), not for the
        TriggerEngine's time-based scanning.
        """
        all_trigs = await self._iter_all()
        out: list[TriggerRecord] = []

        for t in all_trigs:
            if active is not None and t.active is not active:
                continue

            if kind is not None and t.kind != kind:
                continue

            if graph_id is not None and t.graph_id != graph_id:
                continue

            if not _matches_tenant(
                t,
                org_id=org_id,
                user_id=user_id,
                client_id=client_id,
            ):
                continue

            out.append(t)

        return out

    async def list_by_event_key(
        self,
        event_key: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> list[TriggerRecord]:
        all_trigs = await self.list_active()
        out: list[TriggerRecord] = []

        for t in all_trigs:
            if t.kind != "event":
                continue
            if t.event_key != event_key:
                continue
            if not _matches_tenant(t, org_id, user_id, client_id):
                continue
            out.append(t)
        return out

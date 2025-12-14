from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from aethergraph.contracts.services.viz import VizEvent, VizKind
from aethergraph.contracts.storage.event_log import EventLog


class VizService:
    """
    Low-level service to append/query visualization events.

    - Uses EventLog as the underlying storage.
    - Does NOT know about NodeContext or Scope; that's the Facade's job.
    """

    def __init__(self, event_log: EventLog):
        self._log = event_log

    async def append(self, evt: VizEvent) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not evt.created_at:
            evt.created_at = now

        payload = asdict(evt)
        await self._log.append(
            {
                "kind": "viz",
                "scope_id": evt.run_id,
                "ts": evt.created_at,
                "data": payload,
                "tags": (evt.tags or []) + [f"track:{evt.track_id}"],
            }
        )

    async def query_run(
        self,
        run_id: str,
        *,
        kinds: list[VizKind] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Raw fetch of viz events for a given run.
        Returns raw event dicts as stored in EventLog.
        The API layer can normalize/group them for the frontend.
        """
        rows = await self._log.query(
            scope_id=run_id,
            since=since,
            until=until,
            kinds=["viz"],
            limit=limit,
            offset=offset,
        )
        # Optionally filter by viz_kind inside data
        if kinds:
            out: list[dict[str, Any]] = []
            for r in rows:
                data = r.get("data") or {}
                if data.get("viz_kind") in kinds:
                    out.append(r)
            return out
        return rows

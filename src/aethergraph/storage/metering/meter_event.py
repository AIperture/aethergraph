from dataclasses import dataclass
from datetime import datetime
from typing import Any

from aethergraph.contracts.services.metering import MeteringStore
from aethergraph.contracts.storage.event_log import EventLog

METER_TAG = "meter"  # shared tag for all metering events


@dataclass
class EventLogMeteringStore(MeteringStore):
    """
    MeteringStore backed by a generic EventLog.

    Convention:
      - kind: e.g. "meter.llm", "meter.run", "meter.artifact", "meter.event"
      - tags: always includes "meter" so queries don't mix with other app events
    """

    event_log: EventLog

    async def append(self, event: dict[str, Any]) -> None:
        # Enforce metering conventions
        kind = event.get("kind")
        if not kind or not kind.startswith("meter."):
            raise ValueError(f"Metering event kind must start with 'meter.': {kind!r}")

        tags = set(event.get("tags") or [])
        tags.add(METER_TAG)
        event["tags"] = list(tags)

        await self.event_log.append(event)

    async def query(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        # Always filter by meter tag
        return await self.event_log.query(
            scope_id=None,
            since=since,
            until=until,
            kinds=kinds,
            tags=[METER_TAG],
            limit=limit,
        )

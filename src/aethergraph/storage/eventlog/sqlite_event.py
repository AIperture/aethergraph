# storage/events/sqlite_event_log.py
from __future__ import annotations

import asyncio
from datetime import datetime

from aethergraph.contracts.storage.event_log import EventLog

from .sqlite_event_sync import SQLiteEventLogSync


class SqliteEventLog(EventLog):
    """
    Async EventLog wrapper around SQLiteEventLogSync via asyncio.to_thread.
    """

    def __init__(self, path: str):
        self._sync = SQLiteEventLogSync(path)

    async def append(self, evt: dict) -> None:
        await asyncio.to_thread(self._sync.append, evt)

    async def query(
        self,
        *,
        scope_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._sync.query,
            scope_id=scope_id,
            since=since,
            until=until,
            kinds=kinds,
            limit=limit,
            tags=tags,
        )

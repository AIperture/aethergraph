# storage/events/sqlite_event_log.py
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Literal

from aethergraph.contracts.storage.event_log import EventLog

from .sqlite_event_sync import SQLiteEventLogSync


class SqliteEventLog(EventLog):
    """
    Async EventLog wrapper around SQLiteEventLogSync via asyncio.to_thread.
    """

    def __init__(self, path: str, *, read_only: bool = False):
        self._sync = SQLiteEventLogSync(path, read_only=read_only)

    async def append(self, evt: dict) -> None:
        await asyncio.to_thread(self._sync.append, evt)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync.close)

    async def query(
        self,
        *,
        scope_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        offset: int = 0,
        user_id: str | None = None,
        org_id: str | None = None,
        client_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        after_id: int | None = None,
        before_id: int | None = None,
        order_dir: Literal["asc", "desc"] = "desc",
    ) -> list[dict]:
        return await asyncio.to_thread(
            self._sync.query,
            scope_id=scope_id,
            since=since,
            until=until,
            kinds=kinds,
            limit=limit,
            tags=tags,
            offset=offset,
            user_id=user_id,
            org_id=org_id,
            client_id=client_id,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            graph_id=graph_id,
            node_id=node_id,
            topic=topic,
            tool=tool,
            after_id=after_id,
            before_id=before_id,
            order_dir=order_dir,
        )

    async def get_many(self, scope_id: str, event_ids: list[str]) -> list[dict]:
        """Read selected events from one exact event-log scope.

        The lookup is read-only and preserves the event-log query ordering.

        Examples:
            Read two events:
                ```python
                rows = await event_log.get_many("run-1", ["event-1", "event-2"])
                ```
            Read an empty selection:
                ```python
                rows = await event_log.get_many("run-1", [])
                ```

        Args:
            scope_id: Exact persisted event scope.
            event_ids: Exact event identifiers to retain.

        Returns:
            list[dict]: Matching persisted event rows in query order.

        Notes:
            Numeric SQLite row identifiers are not accepted as event IDs.
        """
        if not event_ids:
            return []
        selected = set(event_ids)
        rows = await self.query(scope_id=scope_id, limit=None)
        return [row for row in rows if str(row.get("id") or "") in selected]

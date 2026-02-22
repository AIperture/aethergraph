from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class TriggerStore(Protocol):
    """All trig should be TriggerRecord, but we avoid importing it here to keep this protocol decoupled from the data layer."""

    async def create(self, trig: Any) -> None: ...
    async def update(self, trig: Any) -> None: ...
    async def get(self, trigger_id: str) -> Any | None: ...
    async def delete(self, trigger_id: str) -> None: ...

    async def list_active(self) -> list[Any]: ...
    async def list_due(self, now: datetime) -> list[Any]: ...
    async def list_by_event_key(
        self,
        event_key: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
    ) -> list[Any]: ...  # used for event-based triggers

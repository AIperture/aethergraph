from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol


class TriggerStore(Protocol):
    """Durable trigger definitions and atomic occurrence claims."""

    async def create(self, trig: Any) -> None: ...
    async def update(self, trig: Any) -> None: ...
    async def get(self, trigger_id: str) -> Any | None: ...
    async def delete(self, trigger_id: str) -> None: ...

    async def claim_due(
        self,
        now: datetime,
        *,
        worker_id: str,
        lease_until: datetime,
        limit: int,
        skip_missed_before: datetime | None = None,
    ) -> list[Any]: ...
    async def complete_claim(
        self, fire_id: str, *, worker_id: str, run_id: str, completed_at: datetime
    ) -> bool: ...
    async def fail_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        error: str,
        retry_at: datetime,
    ) -> bool: ...
    async def skip_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        reason: str,
        completed_at: datetime,
    ) -> bool: ...
    async def get_claim(self, fire_id: str) -> dict[str, Any] | None: ...
    async def list_all(
        self,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        kind: str | None = None,
        active: bool | None = None,
    ) -> list[Any]: ...
    async def list_by_event_key(
        self,
        event_key: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> list[Any]: ...  # used for event-based triggers

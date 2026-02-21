from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

TriggerKind = Literal["cron", "interval", "one_shot", "event"]


class TriggerService(Protocol):
    async def create_from_scope(
        self,
        *,
        scope: Any,  # we use Scope in the impl, but keep it generic here to avoid coupling
        graph_id: str,
        default_inputs: dict[str, Any],
        kind: TriggerKind,
        cron_expr: str | None = None,
        interval_seconds: int | None = None,
        run_at: datetime | None = None,
        event_key: str | None = None,
        max_overlap_runs: int | None = None,
        catch_up_missed: bool = False,
        origin: str = "schedule",
        meta: dict[str, Any] | None = None,
    ) -> Any: ...  # should return TriggerRecord, but we avoid importing it here to keep this protocol decoupled from the data layer

    async def cancel(self, trigger_id: str) -> None: ...
    async def get(self, trigger_id: str) -> Any | None: ...
    async def list_for_owner(self, org_id: str, user_id: str) -> list[Any]: ...

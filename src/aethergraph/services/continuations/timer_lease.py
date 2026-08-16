"""Provider-neutral runtime continuation timer-lease values."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TimerLeaseStatus(StrEnum):
    """Runtime state for one scheduled continuation occurrence."""

    LEASED = "leased"
    RETRY = "retry"
    DELIVERED = "delivered"
    DEAD_LETTER = "dead_letter"


@dataclass(frozen=True, slots=True)
class TimerLease:
    """Revisioned provider-neutral claim or delivery receipt."""

    fire_id: str
    continuation_id: str
    run_id: str
    node_id: str
    scheduled_for: datetime
    status: TimerLeaseStatus
    attempts: int
    revision: int
    updated_at: datetime
    worker_id: str | None = None
    lease_until: datetime | None = None
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    finished_at: datetime | None = None
    reclaimed: bool = False

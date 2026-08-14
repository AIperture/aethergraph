from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from croniter import croniter  # type: ignore[import]
from dateutil.tz import gettz  # type: ignore[import-untyped]

from aethergraph.contracts.services.trigger import TriggerKind

from .types import TriggerRecord


def _timezone(name: str | None) -> Any:
    zone_name = name or "UTC"
    zone = gettz(zone_name)
    if zone is None:
        raise ValueError(f"Unknown trigger timezone: {name}")
    return zone


def _normalize_utc(value: datetime) -> datetime:
    """Normalize one datetime to an aware UTC instant."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _validate_trigger_config(
    *,
    kind: TriggerKind,
    cron_expr: str | None,
    interval_seconds: int | None,
    run_at: datetime | None,
    event_key: str | None,
    tz: str | None,
    max_overlap_runs: int | None,
) -> None:
    """Validate one trigger definition before it reaches storage."""
    _timezone(tz)

    if max_overlap_runs is not None and max_overlap_runs < 0:
        raise ValueError("max_overlap_runs must be zero or greater")

    if kind == "cron":
        if not cron_expr:
            raise ValueError("cron_expr is required for cron triggers")
        if not croniter.is_valid(cron_expr):
            raise ValueError(f"Invalid cron expression: {cron_expr}")
        return
    if kind == "interval":
        if interval_seconds is None or interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        return
    if kind == "one_shot":
        if run_at is None:
            raise ValueError("run_at is required for one_shot triggers")
        return
    if kind == "event":
        if not event_key or not event_key.strip():
            raise ValueError("event_key is required for event triggers")
        return
    raise ValueError(f"Unsupported trigger kind: {kind}")


def _next_recurrence(trig: TriggerRecord, after: datetime) -> datetime | None:
    """Compute the next UTC recurrence while preserving trigger timezone."""
    after_utc = _normalize_utc(after)
    if trig.kind == "interval":
        if trig.interval_seconds is None or trig.interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")
        return after_utc + timedelta(seconds=trig.interval_seconds)
    if trig.kind == "cron":
        if not trig.cron_expr:
            raise ValueError("cron_expr is required for cron triggers")
        local_zone = _timezone(trig.tz)
        local_after = after_utc.astimezone(local_zone)
        return croniter(trig.cron_expr, local_after).get_next(datetime).astimezone(UTC)
    return None


def _initial_fire_at(trig: TriggerRecord, now: datetime) -> datetime | None:
    """Compute the first UTC fire time for a validated trigger."""
    now_utc = _normalize_utc(now)
    if not trig.active or trig.kind == "event":
        return None
    if trig.kind == "one_shot":
        if trig.run_at is None:
            return None
        local_zone = _timezone(trig.tz)
        run_at = (
            trig.run_at.astimezone(local_zone)
            if trig.run_at.tzinfo is not None
            else trig.run_at.replace(tzinfo=local_zone)
        )
        return run_at.astimezone(UTC)
    return _next_recurrence(trig, now_utc)


def _advance_after_claim(
    trig: TriggerRecord,
    *,
    scheduled_for: datetime,
    now: datetime,
) -> datetime | None:
    """Advance a claimed recurrence according to its catch-up policy."""
    if trig.kind not in {"cron", "interval"}:
        return None
    candidate = _next_recurrence(trig, scheduled_for)
    if trig.catch_up_missed:
        return candidate
    now_utc = _normalize_utc(now)
    while candidate is not None and candidate <= now_utc:
        candidate = _next_recurrence(trig, candidate)
    return candidate

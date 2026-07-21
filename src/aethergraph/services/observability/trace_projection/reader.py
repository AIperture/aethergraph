"""Workspace read boundary for the trace projection.

This is the only module that touches the AetherGraph observability facade. It
normalizes the facade's engine-event dicts and run records into the small typed
shapes the rest of the projection consumes, so an upstream storage change is
absorbed here by the connected-runtime presenter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from aethergraph.services.observability import ObservabilityFacade

_DISPATCH_CONTINUATION_KINDS = frozenset({"subagent_call"})


def to_epoch(value: Any) -> float:
    """Coerce an engine timestamp (float epoch or ISO string) to epoch seconds."""
    if value in (None, ""):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def to_iso(value: Any) -> str:
    """Coerce an engine timestamp to a stable ISO-8601 string (or "")."""
    if value in (None, ""):
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    return str(value)


@dataclass(frozen=True)
class EngineEvent:
    """One normalized `agent_engine.*` event."""

    event_id: str
    ts: float
    iso: str
    run_id: str
    session_id: str
    kind: str
    text: str
    agent_instance_id: str
    turn_id: str
    tags: list[str]
    data: dict[str, Any]

    @property
    def caused_by_event_id(self) -> str:
        return str(self.data.get("caused_by_event_id") or "")


@dataclass(frozen=True)
class RunInfo:
    """One normalized run record."""

    run_id: str
    session_id: str
    agent_id: str
    graph_id: str
    status: str
    started_at: str
    finished_at: str | None
    turn_id: str
    is_child: bool
    source_agent_instance_id: str
    target_agent_instance_id: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def started_epoch(self) -> float:
        return to_epoch(self.started_at)


class TraceReader:
    """Typed reads over one opened observability facade."""

    def __init__(self, facade: ObservabilityFacade) -> None:
        self._facade = facade

    async def suppressed(self) -> dict[str, set[str]]:
        return await self._facade.store.list_suppressed_scopes()

    async def runs(self) -> list[RunInfo]:
        """All non-suppressed runs, oldest first."""
        raw_runs = await self._facade.run_store.list(limit=10_000, offset=0)
        suppressed = await self.suppressed()
        result: list[RunInfo] = []
        for run in raw_runs:
            if not self._facade._run_is_visible(run):
                continue
            info = _run_info(run)
            if self._is_suppressed(info, suppressed):
                continue
            result.append(info)
        result.sort(key=lambda item: item.started_epoch)
        return result

    async def run(self, run_id: str) -> RunInfo | None:
        raw = await self._facade.run_store.get(run_id)
        if raw is None:
            return None
        if not self._facade._run_is_visible(raw):
            return None
        info = _run_info(raw)
        if self._is_suppressed(info, await self.suppressed()):
            return None
        return info

    async def events_for_runs(self, run_ids: Iterable[str]) -> list[EngineEvent]:
        """Normalized `agent_engine.*` events across the given runs, causal order.

        Runs are ordered by their earliest event; within a run the store's
        ascending order is preserved. Cross-run ordering falls back to the
        per-run bucket so equal-second events never interleave runs.
        """
        ranked: list[tuple[float, int, int, EngineEvent]] = []
        for bucket, run_id in enumerate(dict.fromkeys(run_ids)):
            rows = await self._facade.engine_event_log.query(
                tags=["agent_engine"],
                run_id=run_id,
                limit=None,
                order_dir="asc",
            )
            for order, row in enumerate(rows):
                if not _event_is_visible(row, self._facade):
                    continue
                event = _engine_event(row)
                ranked.append((event.ts, bucket, order, event))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [event for *_ignored, event in ranked]

    async def prompt_manifest(self, manifest_id: str) -> dict[str, Any] | None:
        return await self._facade.store.hydrate_prompt_manifest(manifest_id)

    @staticmethod
    def _is_suppressed(info: RunInfo, suppressed: dict[str, set[str]]) -> bool:
        session_id = info.session_id or info.run_id
        return (
            info.run_id in suppressed.get("run_id", set())
            or info.run_id in suppressed.get("trace_id", set())
            or session_id in suppressed.get("session_id", set())
        )


def _run_info(run: dict[str, Any]) -> RunInfo:
    meta = dict(run.get("meta") or {})
    original_inputs = dict(meta.get("original_inputs") or {})
    user_request = dict(original_inputs.get("user_request") or {})
    continuation = dict(original_inputs.get("continuation_payload") or {})
    continuation_kind = str(continuation.get("kind") or "")
    return RunInfo(
        run_id=str(run.get("run_id") or ""),
        session_id=str(run.get("session_id") or ""),
        agent_id=str(run.get("agent_id") or ""),
        graph_id=str(run.get("graph_id") or ""),
        status=_status_text(run.get("status")),
        started_at=to_iso(run.get("started_at")),
        finished_at=to_iso(run.get("finished_at")) or None,
        turn_id=str(user_request.get("turn_id") or ""),
        is_child=continuation_kind in _DISPATCH_CONTINUATION_KINDS,
        source_agent_instance_id=str(continuation.get("source_agent_instance_id") or ""),
        target_agent_instance_id=str(continuation.get("target_agent_instance_id") or ""),
        raw=run,
    )


def _engine_event(row: dict[str, Any]) -> EngineEvent:
    data = dict(row.get("data") or {})
    tags = [str(tag) for tag in (row.get("tags") or [])]
    turn_id = str(data.get("turn_id") or "")
    if not turn_id:
        turn_id = next(
            (tag.split(":", 1)[1] for tag in tags if tag.startswith("turn:")),
            "",
        )
    return EngineEvent(
        event_id=str(row.get("event_id") or row.get("id") or ""),
        ts=to_epoch(row.get("ts")),
        iso=to_iso(row.get("ts")),
        run_id=str(row.get("run_id") or ""),
        session_id=str(row.get("session_id") or ""),
        kind=str(row.get("kind") or ""),
        text=str(row.get("text") or ""),
        agent_instance_id=str(data.get("agent_instance_id") or ""),
        turn_id=turn_id,
        tags=tags,
        data=data,
    )


def _status_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "unknown")


def _event_is_visible(row: dict[str, Any], facade: ObservabilityFacade) -> bool:
    """Apply the runtime identity boundary before interpreting engine events."""
    identity = facade.identity
    if identity.mode not in {"cloud", "demo"}:
        return True
    if not identity.user_id:
        return False
    if str(row.get("user_id") or "") != identity.user_id:
        return False
    return not identity.org_id or str(row.get("org_id") or "") == identity.org_id


__all__ = ["EngineEvent", "RunInfo", "TraceReader", "to_epoch", "to_iso"]

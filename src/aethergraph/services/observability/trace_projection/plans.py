"""Plan and guideline snapshot timeline (one normalized row grammar).

Plans and guidelines share the `PlanSnapshot` shape: a plan contributes step
rows and a cursor; a guideline contributes guideline rows and progress events.
Change summaries are computed here so the UI never diffs snapshots.
"""

from __future__ import annotations

from typing import Any

from .models import (
    GuidelineProgressEvent,
    PlanItemRow,
    PlanSnapshot,
    PlanTimeline,
)
from .reader import EngineEvent

_GUIDELINE_STATUSES = {"open", "completed"}


def build_plan_timeline(events: list[EngineEvent]) -> PlanTimeline:
    """Ordered plan/guideline snapshots with per-step and per-plan change notes."""
    snapshots: list[PlanSnapshot] = []
    previous: dict[str, Any] | None = None
    for event in events:
        scope = event.data.get("guideline") or event.data.get("plan")
        if not isinstance(scope, dict):
            continue
        snapshot = _snapshot(event, scope, previous)
        snapshots.append(snapshot)
        previous = scope
    return PlanTimeline(snapshots=snapshots)


def _snapshot(
    event: EngineEvent,
    plan: dict[str, Any],
    previous: dict[str, Any] | None,
) -> PlanSnapshot:
    guidelines = plan.get("guidelines")
    is_guideline = plan.get("scope_kind") == "guideline" or isinstance(guidelines, list)
    scope_kind = "guideline" if is_guideline else "plan"
    cursor = None if is_guideline else _int(plan.get("cursor"))
    items = _guideline_rows(plan) if is_guideline else _step_rows(plan)
    return PlanSnapshot(
        captured_at=event.iso,
        run_id=event.run_id,
        agent_name=str(event.data.get("agent_id") or event.agent_instance_id or ""),
        scope_kind=scope_kind,
        status=str(plan.get("status") or ""),
        version=_int(plan.get("version")),
        goal=str(plan.get("goal") or ""),
        cursor=cursor,
        items=items,
        change_summary=_change_summary(plan, previous, is_guideline),
        progress_events=_progress_events(plan) if is_guideline else [],
        raw_plan=plan,
    )


def _step_rows(plan: dict[str, Any]) -> list[PlanItemRow]:
    cursor = _int(plan.get("cursor"))
    rows: list[PlanItemRow] = []
    for index, step in enumerate(plan.get("steps") or []):
        if not isinstance(step, dict):
            continue
        rows.append(
            PlanItemRow(
                item_id=str(step.get("step_id") or f"step-{index}"),
                title=str(step.get("title") or step.get("node_id") or f"Step {index + 1}"),
                detail=str(step.get("objective") or ""),
                status=str(step.get("status") or "pending"),
                is_cursor=index == cursor,
                item_kind=str(step.get("step_kind") or "node"),
                execution_record=_dict_or_none(step.get("execution_record")),
                observation=_dict_or_none(step.get("observation")),
                progress_notes=[str(note) for note in (step.get("progress_notes") or [])],
            )
        )
    return rows


def _guideline_rows(plan: dict[str, Any]) -> list[PlanItemRow]:
    completed = {str(item) for item in (plan.get("completed_guidelines") or [])}
    rows: list[PlanItemRow] = []
    for index, guideline in enumerate(plan.get("guidelines") or []):
        text = str(guideline)
        rows.append(
            PlanItemRow(
                item_id=f"guideline-{index}",
                title=text,
                detail=str(plan.get("remaining") or ""),
                status="completed" if text in completed else "open",
                is_cursor=False,
                item_kind="guideline",
                execution_record=None,
                observation=None,
                progress_notes=[],
            )
        )
    for index, question in enumerate(plan.get("open_questions") or []):
        rows.append(
            PlanItemRow(
                item_id=f"open-question-{index}",
                title=str(question),
                detail="",
                status="open",
                is_cursor=False,
                item_kind="open_question",
                execution_record=None,
                observation=None,
                progress_notes=[],
            )
        )
    return rows


def _progress_events(plan: dict[str, Any]) -> list[GuidelineProgressEvent]:
    events: list[GuidelineProgressEvent] = []
    for entry in plan.get("progress_events") or []:
        if not isinstance(entry, dict):
            continue
        events.append(
            GuidelineProgressEvent(
                event_type=str(entry.get("event_type") or ""),
                summary=str(entry.get("summary") or ""),
                tool_name=str(entry.get("tool_name") or ""),
                created_at=str(entry.get("created_at") or ""),
            )
        )
    return events


def _change_summary(
    plan: dict[str, Any],
    previous: dict[str, Any] | None,
    is_guideline: bool,
) -> list[str]:
    if previous is None:
        return []
    notes: list[str] = []
    if str(plan.get("status") or "") != str(previous.get("status") or ""):
        notes.append(f"status {previous.get('status')} → {plan.get('status')}")
    if not is_guideline and _int(plan.get("cursor")) != _int(previous.get("cursor")):
        notes.append(f"cursor {previous.get('cursor')} → {plan.get('cursor')}")
    notes.extend(_step_change_notes(plan, previous))
    notes.extend(_guideline_change_notes(plan, previous) if is_guideline else [])
    return notes


def _step_change_notes(plan: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    prev_by_id = {
        str(step.get("step_id")): step
        for step in (previous.get("steps") or [])
        if isinstance(step, dict)
    }
    notes: list[str] = []
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            continue
        prev = prev_by_id.get(str(step.get("step_id")))
        if prev and str(prev.get("status")) != str(step.get("status")):
            title = str(step.get("title") or step.get("node_id") or step.get("step_id"))
            notes.append(f"{title}: {prev.get('status')} → {step.get('status')}")
    return notes


def _guideline_change_notes(plan: dict[str, Any], previous: dict[str, Any]) -> list[str]:
    prev_done = {str(item) for item in (previous.get("completed_guidelines") or [])}
    now_done = {str(item) for item in (plan.get("completed_guidelines") or [])}
    return [f"completed: {text}" for text in sorted(now_done - prev_done)]


def _dict_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) and value else None


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_plan_timeline"]

"""Project one turn's engine events into ordered agent segments.

Consumes the normalized event stream (all runs of a turn, causal order) and the
already-linked dispatch infos, and emits `AgentSegment`s each holding an ordered
union of `Cycle` / `Dispatch` / `Interaction` / `RuntimeErrorItem`. Cycles pair
their tool by the canonical two-hop causal chain
(`tool_result → tool_call → decision`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .linking import DispatchInfo
from .models import (
    AgentSegment,
    Cycle,
    CycleAction,
    CycleContext,
    Dispatch,
    Interaction,
    ResourceLink,
    RunOutcome,
    RuntimeErrorItem,
    SegmentItem,
    ToolExecution,
)
from .reader import EngineEvent

_MAX_PREVIEW = 200


@dataclass
class _CycleBuilder:
    cycle_id: str
    step_index: int
    mode: str
    status: str
    started_at: str
    action: CycleAction
    reasoning_summary: str
    context: CycleContext
    llm_call_id: str
    ended_at: str | None = None
    tool: ToolExecution | None = None

    def build(self) -> Cycle:
        return Cycle(
            cycle_id=self.cycle_id,
            step_index=self.step_index,
            mode=self.mode,
            status=self.status,
            started_at=self.started_at,
            ended_at=self.ended_at,
            action=self.action,
            reasoning_summary=self.reasoning_summary,
            context=self.context,
            tool=self.tool,
            llm_call_id=self.llm_call_id,
        )


@dataclass
class _SegmentBuilder:
    segment_id: str
    run_id: str
    agent_instance_id: str
    agent_name: str
    entry_kind: str
    started_at: str
    status: str = "running"
    ended_at: str | None = None
    reply_preview: str = ""
    items: list[object] = field(default_factory=list)

    def build(self) -> AgentSegment:
        items: list[SegmentItem] = [
            item.build() if isinstance(item, _CycleBuilder) else item for item in self.items
        ]
        return AgentSegment(
            segment_id=self.segment_id,
            run_id=self.run_id,
            agent_instance_id=self.agent_instance_id,
            agent_name=self.agent_name,
            entry_kind=self.entry_kind,
            status=self.status,
            started_at=self.started_at,
            ended_at=self.ended_at,
            reply_preview=self.reply_preview,
            items=items,
        )


def build_segments(
    events: list[EngineEvent],
    *,
    child_run_ids: set[str],
    resumption_run_ids: set[str],
    dispatches: list[DispatchInfo],
    agent_names: dict[str, str],
) -> list[AgentSegment]:
    """Ordered agent segments for one turn."""
    dispatch_by_entered = {d.entered_event_id: d for d in dispatches}
    segments: list[_SegmentBuilder] = []
    open_segment: dict[tuple[str, str], _SegmentBuilder] = {}
    cycles_by_id: dict[str, _CycleBuilder] = {}
    cycle_of_tool_call: dict[str, str] = {}
    tool_calls: dict[str, EngineEvent] = {}
    run_segment_count: dict[str, int] = {}
    context_entry_ids_by_agent: dict[str, set[str]] = {}

    def current(agent_id: str, run_id: str) -> _SegmentBuilder | None:
        return open_segment.get((run_id, agent_id))

    def latest(run_id: str, agent_id: str) -> _SegmentBuilder | None:
        return next(
            (
                segment
                for segment in reversed(segments)
                if segment.run_id == run_id
                and (not agent_id or segment.agent_instance_id == agent_id)
            ),
            None,
        )

    for event in events:
        kind = event.kind
        agent_id = event.agent_instance_id
        run_id = event.run_id

        if kind == "agent_engine.agent_entered":
            entry_kind = _entry_kind(
                is_child=run_id in child_run_ids,
                is_resumption=run_id in resumption_run_ids,
                first_in_run=run_segment_count.get(run_id, 0) == 0,
            )
            builder = _SegmentBuilder(
                segment_id=event.event_id,
                run_id=run_id,
                agent_instance_id=agent_id,
                agent_name=agent_names.get(agent_id, agent_id),
                entry_kind=entry_kind,
                started_at=event.iso,
            )
            segments.append(builder)
            open_segment[(run_id, agent_id)] = builder
            run_segment_count[run_id] = run_segment_count.get(run_id, 0) + 1

        elif kind == "agent_engine.agent_exited":
            segment = current(agent_id, run_id)
            if segment is not None:
                segment.ended_at = event.iso
                segment.status = str(event.data.get("status") or "completed")
                segment.reply_preview = _preview(event.data.get("reply") or event.text)
                open_segment.pop((run_id, agent_id), None)

        elif kind == "agent_engine.decision":
            segment = current(agent_id, run_id)
            if segment is None:
                continue
            cycle = _cycle_builder(
                event,
                previous_context_entry_ids=context_entry_ids_by_agent.get(agent_id, set()),
            )
            context_entry_ids_by_agent[agent_id] = set(
                str(value) for value in (event.data.get("new_context_entry_ids") or [])
            )
            cycles_by_id[cycle.cycle_id] = cycle
            segment.items.append(cycle)

        elif kind == "agent_engine.tool_call":
            tool_calls[event.event_id] = event
            cycle_of_tool_call[event.event_id] = event.caused_by_event_id

        elif kind == "agent_engine.tool_result":
            call = tool_calls.get(event.caused_by_event_id)
            if call is None:
                continue
            cycle_id = cycle_of_tool_call.get(call.event_id, "")
            cycle = cycles_by_id.get(cycle_id)
            if cycle is None:
                continue
            cycle.tool = _tool_execution(call, event)
            cycle.ended_at = event.iso

        elif kind == "agent_engine.dispatch_entered":
            segment = current(str(event.data.get("source_agent_instance_id") or agent_id), run_id)
            info = dispatch_by_entered.get(event.event_id)
            if segment is not None and info is not None:
                segment.items.append(_dispatch(info))

        elif kind in ("agent_engine.interaction_waited", "agent_engine.interaction_resumed"):
            segment = current(agent_id, run_id)
            if segment is not None:
                segment.items.append(_interaction(event))

        elif kind == "agent_engine.runtime_error":
            segment = current(agent_id, run_id)
            if segment is not None:
                segment.items.append(_runtime_error(event))

        elif kind == "agent_engine.run_outcome":
            segment = current(agent_id, run_id) or latest(run_id, agent_id)
            if segment is not None:
                segment.items.append(_run_outcome(event))

    return [builder.build() for builder in segments]


def _entry_kind(*, is_child: bool, is_resumption: bool, first_in_run: bool) -> str:
    """A run's first agent window is the user turn (root) or the dispatch
    (child); every re-entry resumes after a dispatch round-trip."""
    if not first_in_run:
        return "return"
    if is_resumption:
        return "return"
    return "dispatch" if is_child else "user_turn"


def _cycle_builder(event: EngineEvent, *, previous_context_entry_ids: set[str]) -> _CycleBuilder:
    data = event.data
    selected = dict(data.get("selected_action") or {})
    args = selected.get("args")
    args_omitted = selected.get("args_omitted")
    summary = dict(data.get("dynamic_context_summary") or {})
    context_entry_ids = [str(value) for value in (data.get("new_context_entry_ids") or [])]
    added_entry_ids = [
        value for value in context_entry_ids if value not in previous_context_entry_ids
    ]
    return _CycleBuilder(
        cycle_id=event.event_id,
        step_index=_int(data.get("step_index")),
        mode=str(data.get("mode") or ""),
        status=str(data.get("status") or ""),
        started_at=event.iso,
        action=CycleAction(
            tool_name=str(selected.get("tool_name") or ""),
            args=dict(args) if isinstance(args, dict) else None,
            args_omitted=dict(args_omitted) if isinstance(args_omitted, dict) else None,
        ),
        reasoning_summary=str(data.get("reasoning_summary") or ""),
        context=CycleContext(
            manifest_id=str(data.get("prompt_manifest_id") or ""),
            new_entry_count=len(added_entry_ids),
            added_entry_ids=added_entry_ids,
            section_keys=[str(key) for key in (summary.get("section_keys") or [])],
            section_char_counts={
                str(key): _int(value)
                for key, value in dict(summary.get("section_char_counts") or {}).items()
            },
            total_chars=_int(summary.get("total_chars")),
            prompt_warning_count=_int(summary.get("prompt_warning_count")),
            has_repair_context=bool(summary.get("has_repair_context")),
        ),
        llm_call_id=str(data.get("llm_call_id") or ""),
    )


def _tool_execution(call: EngineEvent, result: EngineEvent) -> ToolExecution:
    call_data = call.data
    result_data = result.data
    links = [
        ResourceLink(
            resource_key=str(link.get("resource_key") or ""),
            relation=str(link.get("relation") or ""),
            resource_kind=str(link.get("resource_kind") or ""),
            revision=str(link.get("revision") or ""),
            content_hash=str(link.get("content_hash") or ""),
            slot_key=str(link.get("slot_key") or ""),
            status=str(link.get("status") or ""),
        )
        for link in (result_data.get("resource_links") or [])
        if isinstance(link, dict)
    ]
    return ToolExecution(
        tool_call_id=str(call_data.get("tool_call_id") or ""),
        tool_name=str(call_data.get("tool") or call_data.get("tool_name") or ""),
        status=str(result_data.get("status") or ""),
        args=dict(call_data.get("args") or {}),
        result_summary=_preview(result.text or result_data.get("summary") or ""),
        result=result_data.get("result"),
        resource_links=links,
        started_at=call.iso,
        ended_at=result.iso,
    )


def _dispatch(info: DispatchInfo) -> Dispatch:
    return Dispatch(
        dispatch_id=info.dispatch_token,
        source_agent_instance_id=info.source_agent_instance_id,
        target_agent_instance_id=info.target_agent_instance_id,
        dispatch_mode=info.dispatch_mode,
        status=info.status,
        instruction_preview=_preview(info.instruction),
        return_text_preview=_preview(info.return_text),
        child_run_id=info.child_run_id,
        required_slots=[dict(item) for item in info.required_slots],
        expected_slots=list(info.expected_slots),
        started_at=info.started_at,
        ended_at=info.ended_at,
    )


def _interaction(event: EngineEvent) -> Interaction:
    status = "waited" if event.kind.endswith("waited") else "resumed"
    return Interaction(
        interaction_id=event.event_id,
        status=status,
        prompt_preview=_preview(event.text or event.data.get("question") or ""),
        started_at=event.iso,
        ended_at=None,
    )


def _runtime_error(event: EngineEvent) -> RuntimeErrorItem:
    return RuntimeErrorItem(
        error_id=event.event_id,
        summary=_preview(event.text or event.data.get("summary") or "runtime error"),
        detail=str(event.data.get("detail") or ""),
        started_at=event.iso,
    )


def _run_outcome(event: EngineEvent) -> RunOutcome:
    return RunOutcome(
        outcome=str(event.data.get("outcome") or ""),
        code=str(event.data.get("code") or ""),
        summary=_preview(event.data.get("summary") or event.text),
        resumable=bool(event.data.get("resumable")),
        started_at=event.iso,
    )


def _preview(value: object) -> str:
    text = str(value or "")
    return text if len(text) <= _MAX_PREVIEW else text[: _MAX_PREVIEW - 1] + "…"


def _int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


__all__ = ["build_segments"]

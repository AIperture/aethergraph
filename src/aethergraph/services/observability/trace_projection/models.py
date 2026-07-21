"""Frozen DTOs for the connected-runtime v2 trace projection.

Every shape mirrors the operator UI contract in
`ag-studio/ui/src/lib/observabilityTypes.ts`. Contract changes must touch both
files in the same change. All DTOs expose ``to_dict()``
returning JSON-safe primitives so the API layer serializes them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TurnSummary:
    """One user-request boundary, routed by its root run id."""

    root_run_id: str
    turn_id: str
    session_id: str
    started_at: str
    ended_at: str | None
    status: str
    user_text_preview: str
    agent_count: int
    cycle_count: int
    tool_count: int
    dispatch_count: int
    child_run_count: int
    #: Agent that entered the turn — the session rail's human-readable title.
    entry_agent_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "user_text_preview": self.user_text_preview,
            "entry_agent_name": self.entry_agent_name,
            "agent_count": self.agent_count,
            "cycle_count": self.cycle_count,
            "tool_count": self.tool_count,
            "dispatch_count": self.dispatch_count,
            "child_run_count": self.child_run_count,
        }


@dataclass(frozen=True)
class TraceSessionGroup:
    """All turns of one session, newest first."""

    session_id: str
    latest_turn: TurnSummary
    turn_count: int
    turns: list[TurnSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "latest_turn": self.latest_turn.to_dict(),
            "turn_count": self.turn_count,
            "turns": [turn.to_dict() for turn in self.turns],
        }


@dataclass(frozen=True)
class RunNode:
    """One run in a turn's flat run tree."""

    run_id: str
    parent_run_id: str | None
    parent_dispatch_id: str | None
    dispatch_mode: str
    agent_instance_id: str
    agent_name: str
    status: str
    started_at: str
    ended_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "parent_dispatch_id": self.parent_dispatch_id,
            "dispatch_mode": self.dispatch_mode,
            "agent_instance_id": self.agent_instance_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True)
class TraceGraphNode:
    node_id: str
    node_kind: str
    target_agent_instance_id: str
    entry: bool
    agent_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "target_agent_instance_id": self.target_agent_instance_id,
            "entry": self.entry,
            "agent_name": self.agent_name,
        }


@dataclass(frozen=True)
class TraceGraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    dispatch_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "dispatch_mode": self.dispatch_mode,
        }


@dataclass(frozen=True)
class TraceGraph:
    graph_id: str
    nodes: dict[str, TraceGraphNode]
    edges: dict[str, TraceGraphEdge]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": {key: edge.to_dict() for key, edge in self.edges.items()},
        }


@dataclass(frozen=True)
class ResourceLink:
    resource_key: str
    relation: str
    resource_kind: str
    revision: str
    content_hash: str
    slot_key: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_key": self.resource_key,
            "relation": self.relation,
            "resource_kind": self.resource_kind,
            "revision": self.revision,
            "content_hash": self.content_hash,
            "slot_key": self.slot_key,
            "status": self.status,
        }


@dataclass(frozen=True)
class ResourceSlotAssignmentTrace:
    slot_version: int
    resource: dict[str, Any]
    assigned_by: str
    assigned_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_version": self.slot_version,
            "resource": dict(self.resource),
            "assigned_by": self.assigned_by,
            "assigned_at": self.assigned_at,
        }


@dataclass(frozen=True)
class ResourceSlotTrace:
    slot_key: str
    current: ResourceSlotAssignmentTrace
    history: list[ResourceSlotAssignmentTrace]

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_key": self.slot_key,
            "current": self.current.to_dict(),
            "history": [item.to_dict() for item in self.history],
        }


@dataclass(frozen=True)
class ToolExecution:
    tool_call_id: str
    tool_name: str
    status: str
    args: dict[str, Any]
    result_summary: str
    result: Any
    resource_links: list[ResourceLink]
    started_at: str
    ended_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "args": self.args,
            "result_summary": self.result_summary,
            "result": self.result,
            "resource_links": [link.to_dict() for link in self.resource_links],
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True)
class ValidationFailure:
    failure_id: str
    tool_name: str
    summary: str
    detail: str
    repair_hint: str
    failure_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "tool_name": self.tool_name,
            "summary": self.summary,
            "detail": self.detail,
            "repair_hint": self.repair_hint,
            "failure_count": self.failure_count,
        }


@dataclass(frozen=True)
class CycleAction:
    tool_name: str
    args: dict[str, Any] | None
    args_omitted: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "args_omitted": self.args_omitted,
        }


@dataclass(frozen=True)
class CycleContext:
    manifest_id: str
    new_entry_count: int
    added_entry_ids: list[str]
    section_keys: list[str]
    section_char_counts: dict[str, int]
    total_chars: int
    prompt_warning_count: int
    has_repair_context: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "new_entry_count": self.new_entry_count,
            "added_entry_ids": self.added_entry_ids,
            "section_keys": self.section_keys,
            "section_char_counts": self.section_char_counts,
            "total_chars": self.total_chars,
            "prompt_warning_count": self.prompt_warning_count,
            "has_repair_context": self.has_repair_context,
        }


@dataclass(frozen=True)
class Cycle:
    cycle_id: str
    step_index: int
    mode: str
    status: str
    started_at: str
    ended_at: str | None
    action: CycleAction
    reasoning_summary: str
    context: CycleContext
    tool: ToolExecution | None
    validation_failures: list[ValidationFailure]
    llm_call_id: str

    kind: str = "cycle"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "cycle_id": self.cycle_id,
            "step_index": self.step_index,
            "mode": self.mode,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "action": self.action.to_dict(),
            "reasoning_summary": self.reasoning_summary,
            "context": self.context.to_dict(),
            "tool": self.tool.to_dict() if self.tool else None,
            "validation_failures": [item.to_dict() for item in self.validation_failures],
            "llm_call_id": self.llm_call_id,
        }


@dataclass(frozen=True)
class Dispatch:
    dispatch_id: str
    source_agent_instance_id: str
    target_agent_instance_id: str
    dispatch_mode: str
    status: str
    instruction_preview: str
    return_text_preview: str
    child_run_id: str | None
    required_slots: list[dict[str, Any]]
    expected_slots: list[str]
    started_at: str
    ended_at: str | None

    kind: str = "dispatch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dispatch_id": self.dispatch_id,
            "source_agent_instance_id": self.source_agent_instance_id,
            "target_agent_instance_id": self.target_agent_instance_id,
            "dispatch_mode": self.dispatch_mode,
            "status": self.status,
            "instruction_preview": self.instruction_preview,
            "return_text_preview": self.return_text_preview,
            "child_run_id": self.child_run_id,
            "required_slots": [dict(item) for item in self.required_slots],
            "expected_slots": list(self.expected_slots),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True)
class Interaction:
    interaction_id: str
    status: str
    prompt_preview: str
    started_at: str
    ended_at: str | None

    kind: str = "interaction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "interaction_id": self.interaction_id,
            "status": self.status,
            "prompt_preview": self.prompt_preview,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass(frozen=True)
class RuntimeErrorItem:
    error_id: str
    summary: str
    detail: str
    started_at: str

    kind: str = "runtime_error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "error_id": self.error_id,
            "summary": self.summary,
            "detail": self.detail,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class RunOutcome:
    outcome: str
    code: str
    summary: str
    resumable: bool
    started_at: str

    kind: str = "run_outcome"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "outcome": self.outcome,
            "code": self.code,
            "summary": self.summary,
            "resumable": self.resumable,
            "started_at": self.started_at,
        }


# One ordered, discriminated union of items inside an AgentSegment.
SegmentItem = Cycle | Dispatch | Interaction | RuntimeErrorItem | RunOutcome


@dataclass(frozen=True)
class AgentSegment:
    segment_id: str
    run_id: str
    agent_instance_id: str
    agent_name: str
    entry_kind: str
    status: str
    started_at: str
    ended_at: str | None
    reply_preview: str
    items: list[SegmentItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "run_id": self.run_id,
            "agent_instance_id": self.agent_instance_id,
            "agent_name": self.agent_name,
            "entry_kind": self.entry_kind,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "reply_preview": self.reply_preview,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class TurnDetail:
    turn: TurnSummary
    runs: list[RunNode]
    graph: TraceGraph
    segments: list[AgentSegment]
    resource_slots: list[ResourceSlotTrace]

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
            "graph": self.graph.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "resource_slots": [slot.to_dict() for slot in self.resource_slots],
        }


@dataclass(frozen=True)
class GuidelineProgressEvent:
    event_type: str
    summary: str
    tool_name: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "summary": self.summary,
            "tool_name": self.tool_name,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class PlanItemRow:
    item_id: str
    title: str
    detail: str
    status: str
    is_cursor: bool
    item_kind: str
    execution_record: dict[str, Any] | None
    observation: dict[str, Any] | None
    progress_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "detail": self.detail,
            "status": self.status,
            "is_cursor": self.is_cursor,
            "item_kind": self.item_kind,
            "execution_record": self.execution_record,
            "observation": self.observation,
            "progress_notes": self.progress_notes,
        }


@dataclass(frozen=True)
class PlanSnapshot:
    captured_at: str
    run_id: str
    agent_name: str
    scope_kind: str
    status: str
    version: int
    goal: str
    cursor: int | None
    items: list[PlanItemRow]
    change_summary: list[str]
    progress_events: list[GuidelineProgressEvent]
    raw_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_at": self.captured_at,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "scope_kind": self.scope_kind,
            "status": self.status,
            "version": self.version,
            "goal": self.goal,
            "cursor": self.cursor,
            "items": [item.to_dict() for item in self.items],
            "change_summary": self.change_summary,
            "progress_events": [event.to_dict() for event in self.progress_events],
            "raw_plan": self.raw_plan,
        }


@dataclass(frozen=True)
class PlanTimeline:
    snapshots: list[PlanSnapshot]

    def to_dict(self) -> dict[str, Any]:
        return {"snapshots": [snapshot.to_dict() for snapshot in self.snapshots]}


@dataclass(frozen=True)
class ContextSection:
    key: str
    value_type: str
    char_count: int
    hash: str
    omitted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value_type": self.value_type,
            "char_count": self.char_count,
            "hash": self.hash,
            "omitted": self.omitted,
        }


@dataclass(frozen=True)
class ContextBodySection:
    key: str
    value: Any

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value}


@dataclass(frozen=True)
class ContextSnapshot:
    snapshot_id: str
    run_id: str
    agent_instance_id: str
    step_index: int
    capture_mode: str
    created_at: str
    total_chars: int
    sections: list[ContextSection]
    body_sections: list[ContextBodySection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "agent_instance_id": self.agent_instance_id,
            "step_index": self.step_index,
            "capture_mode": self.capture_mode,
            "created_at": self.created_at,
            "total_chars": self.total_chars,
            "sections": [section.to_dict() for section in self.sections],
            "body_sections": [section.to_dict() for section in self.body_sections],
        }


@dataclass
class TraceSessionPage:
    items: list[TraceSessionGroup] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }


__all__ = [
    "AgentSegment",
    "ContextBodySection",
    "ContextSection",
    "ContextSnapshot",
    "Cycle",
    "CycleAction",
    "CycleContext",
    "Dispatch",
    "GuidelineProgressEvent",
    "Interaction",
    "PlanItemRow",
    "PlanSnapshot",
    "PlanTimeline",
    "ResourceLink",
    "ResourceSlotAssignmentTrace",
    "ResourceSlotTrace",
    "RunNode",
    "RuntimeErrorItem",
    "RunOutcome",
    "SegmentItem",
    "ToolExecution",
    "TraceGraph",
    "TraceGraphEdge",
    "TraceGraphNode",
    "TraceSessionGroup",
    "TraceSessionPage",
    "TurnDetail",
    "TurnSummary",
    "ValidationFailure",
]

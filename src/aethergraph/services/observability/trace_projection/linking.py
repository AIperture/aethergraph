"""Run-level structure: turn grouping, dispatch pairing, run tree, graph.

The join model follows canonical runtime identity: every trace-bearing run with
the same `turn_id` belongs to one turn, and dispatched child runs link by their
persisted dispatch token.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .models import RunNode, TraceGraph, TraceGraphEdge, TraceGraphNode
from .reader import EngineEvent, RunInfo

_DISPATCH_ENTERED = "agent_engine.dispatch_entered"
_RETURN_INTENT = "agent_engine.return_intent"


@dataclass(frozen=True)
class TurnGroup:
    """One turn containing its root, dispatch children, and resumed run segments."""

    turn_id: str
    root: RunInfo
    children: list[RunInfo]
    resumptions: list[RunInfo]

    @property
    def members(self) -> list[RunInfo]:
        return [
            self.root,
            *sorted(
                [*self.children, *self.resumptions],
                key=lambda item: item.started_epoch,
            ),
        ]

    @property
    def run_ids(self) -> list[str]:
        return [member.run_id for member in self.members]


@dataclass(frozen=True)
class DispatchInfo:
    """One `dispatch_entered` paired with its parent-owned `return_intent`."""

    dispatch_token: str
    source_run_id: str
    source_agent_instance_id: str
    target_agent_instance_id: str
    dispatch_mode: str
    instruction: str
    status: str
    return_text: str
    required_slots: list[dict[str, object]]
    expected_slots: list[str]
    entered_event_id: str
    started_at: str
    ended_at: str | None
    entered_ts: float
    child_run_id: str | None = field(default=None)


def resolve_event_turn_ids(runs: list[RunInfo], events: list[EngineEvent]) -> list[RunInfo]:
    """Apply canonical engine-event turn identity to run-store records.

    A runtime wrapper root can be persisted before the compiled entry constructs its
    ``UserRequest``, so its run metadata can omit ``turn_id``. Engine events are
    emitted after that boundary and carry the authoritative identity. Event
    identity therefore overrides missing or stale run metadata before grouping.
    """
    event_turn_by_run: dict[str, str] = {}
    event_run_ids: set[str] = set()
    for event in events:
        event_run_ids.add(event.run_id)
        if event.turn_id:
            event_turn_by_run.setdefault(event.run_id, event.turn_id)
    initially_resolved = [
        replace(
            run,
            turn_id=event_turn_by_run.get(run.run_id, run.turn_id),
            has_engine_events=run.run_id in event_run_ids,
        )
        for run in runs
    ]

    by_run_id: dict[str, RunInfo] = {}
    turn_aliases: dict[str, str] = {}
    repaired: dict[str, RunInfo] = {}
    for run in sorted(initially_resolved, key=lambda item: item.started_epoch):
        raw_turn_id = run.turn_id
        effective_turn_id = turn_aliases.get(raw_turn_id, raw_turn_id)
        if run.is_resumption and run.resume_owner_run_id:
            owner = by_run_id.get(run.resume_owner_run_id)
            owner_turn_id = "" if owner is None else owner.turn_id
            if owner_turn_id:
                if raw_turn_id and raw_turn_id != owner_turn_id:
                    turn_aliases[raw_turn_id] = owner_turn_id
                effective_turn_id = owner_turn_id
        effective = replace(run, turn_id=effective_turn_id)
        by_run_id[effective.run_id] = effective
        repaired[effective.run_id] = effective
    return [repaired[run.run_id] for run in runs]


def group_turns(runs: list[RunInfo]) -> list[TurnGroup]:
    """Group runs into turns, newest root first.

    The earliest non-child run is the stable turn root. Later non-child runs are
    resumption segments, and dispatched children retain their separate role.
    Infrastructure runs are excluded. A legacy event-bearing run without a
    `turn_id` remains visible under its run identity.
    """
    runs_by_turn: dict[str, list[RunInfo]] = {}
    for run in runs:
        if run.is_infrastructure:
            continue
        if not run.turn_id and not run.has_engine_events:
            continue
        turn_key = run.turn_id or run.run_id
        runs_by_turn.setdefault(turn_key, []).append(run)

    groups: list[TurnGroup] = []
    for turn_key, members in runs_by_turn.items():
        ordered = sorted(members, key=lambda item: item.started_epoch)
        roots = [run for run in ordered if not run.is_child]
        if not roots:
            continue
        root = roots[0]
        groups.append(
            TurnGroup(
                turn_id=root.turn_id or turn_key,
                root=root,
                children=[run for run in ordered if run.is_child],
                resumptions=roots[1:],
            )
        )
    groups.sort(key=lambda group: group.root.started_epoch, reverse=True)
    return groups


def dispatch_infos(events: list[EngineEvent]) -> list[DispatchInfo]:
    """Pair `dispatch_entered` with `return_intent` by token in entry order."""
    returned: dict[str, EngineEvent] = {}
    for event in events:
        if event.kind == _RETURN_INTENT:
            token = str(event.data.get("dispatch_token") or "")
            if token:
                returned[token] = event

    infos: list[DispatchInfo] = []
    for event in events:
        if event.kind != _DISPATCH_ENTERED:
            continue
        data = event.data
        token = str(data.get("dispatch_token") or "")
        ret = returned.get(token)
        ret_data = ret.data if ret else {}
        infos.append(
            DispatchInfo(
                dispatch_token=token,
                source_run_id=event.run_id,
                source_agent_instance_id=str(data.get("source_agent_instance_id") or ""),
                target_agent_instance_id=str(data.get("target_agent_instance_id") or ""),
                dispatch_mode=str(data.get("dispatch_mode") or ""),
                instruction=str(data.get("instruction") or event.text or ""),
                status=str(ret_data.get("status") or data.get("status") or "dispatched"),
                return_text=str((ret.text if ret else "") or ""),
                required_slots=[
                    dict(item)
                    for item in (data.get("required_slots") or [])
                    if isinstance(item, dict)
                ],
                expected_slots=[str(item) for item in (data.get("expected_slots") or [])],
                entered_event_id=event.event_id,
                started_at=event.iso,
                ended_at=ret.iso if ret else None,
                entered_ts=event.ts,
            )
        )
    return infos


def link_children(
    dispatches: list[DispatchInfo],
    children: list[RunInfo],
) -> list[DispatchInfo]:
    """Resolve each dispatch's child run by exact persisted dispatch token."""
    child_by_token = {
        child.dispatch_token: child.run_id for child in children if child.dispatch_token
    }
    resolved: list[DispatchInfo] = []
    for dispatch in sorted(dispatches, key=lambda item: item.entered_ts):
        resolved.append(
            DispatchInfo(
                **{
                    **dispatch.__dict__,
                    "child_run_id": child_by_token.get(dispatch.dispatch_token),
                }
            )
        )
    return resolved


def build_run_tree(group: TurnGroup, dispatches: list[DispatchInfo]) -> list[RunNode]:
    """Flat run tree with child dispatch and parent-resumption linkage."""
    dispatch_by_child = {d.child_run_id: d for d in dispatches if d.child_run_id}
    nodes: list[RunNode] = [
        RunNode(
            run_id=group.root.run_id,
            parent_run_id=None,
            parent_dispatch_id=None,
            dispatch_mode="",
            agent_instance_id=group.root.agent_id,
            agent_name=group.root.agent_id,
            status=group.root.status,
            started_at=group.root.started_at,
            ended_at=group.root.finished_at,
        )
    ]
    for member in group.members[1:]:
        dispatch = dispatch_by_child.get(member.run_id)
        nodes.append(
            RunNode(
                run_id=member.run_id,
                parent_run_id=(
                    dispatch.source_run_id
                    if dispatch
                    else member.resume_owner_run_id or group.root.run_id
                ),
                parent_dispatch_id=dispatch.dispatch_token if dispatch else None,
                dispatch_mode=dispatch.dispatch_mode if dispatch else "",
                agent_instance_id=member.agent_id,
                agent_name=member.agent_id,
                status=member.status,
                started_at=member.started_at,
                ended_at=member.finished_at,
            )
        )
    return nodes


def build_graph(graph_id: str, events: list[EngineEvent]) -> TraceGraph:
    """Observed agent/dispatch graph: nodes from agents, edges from dispatches."""
    nodes: dict[str, TraceGraphNode] = {}

    def ensure(agent_id: str) -> None:
        if agent_id and agent_id not in nodes:
            nodes[agent_id] = TraceGraphNode(
                node_id=agent_id,
                node_kind="agent",
                target_agent_instance_id=agent_id,
                entry=not nodes,
                agent_name=agent_id,
            )

    edges: dict[str, TraceGraphEdge] = {}
    for event in events:
        ensure(event.agent_instance_id)
        if event.kind != _DISPATCH_ENTERED:
            continue
        data = event.data
        source = str(data.get("source_agent_instance_id") or "")
        target = str(data.get("target_agent_instance_id") or "")
        token = str(data.get("dispatch_token") or event.event_id)
        ensure(source)
        ensure(target)
        if source and target:
            edges[token] = TraceGraphEdge(
                edge_id=token,
                source_node_id=source,
                target_node_id=target,
                dispatch_mode=str(data.get("dispatch_mode") or ""),
            )
    return TraceGraph(graph_id=graph_id, nodes=nodes, edges=edges)


__all__ = [
    "DispatchInfo",
    "TurnGroup",
    "build_graph",
    "build_run_tree",
    "dispatch_infos",
    "group_turns",
    "link_children",
]

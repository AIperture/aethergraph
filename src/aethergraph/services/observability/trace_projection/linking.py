"""Run-level structure: turn grouping, dispatch pairing, run tree, graph.

The join model follows canonical runtime identity: turns group by
`turn_id`; a turn's root run has no inbound dispatch continuation; a dispatch
links to a child run by `(target_agent_instance_id, temporal order)` because the
child run never persists the parent dispatch token.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .models import RunNode, TraceGraph, TraceGraphEdge, TraceGraphNode
from .reader import EngineEvent, RunInfo

_DISPATCH_ENTERED = "agent_engine.dispatch_entered"
_DISPATCH_RETURNED = "agent_engine.dispatch_returned"


@dataclass(frozen=True)
class TurnGroup:
    """One turn: its root run and any dispatch child runs."""

    turn_id: str
    root: RunInfo
    children: list[RunInfo]

    @property
    def run_ids(self) -> list[str]:
        return [self.root.run_id, *(child.run_id for child in self.children)]


@dataclass(frozen=True)
class DispatchInfo:
    """One paired dispatch_entered(+dispatch_returned) on a source run."""

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
    for event in events:
        if event.turn_id:
            event_turn_by_run.setdefault(event.run_id, event.turn_id)
    return [replace(run, turn_id=event_turn_by_run.get(run.run_id, run.turn_id)) for run in runs]


def group_turns(runs: list[RunInfo]) -> list[TurnGroup]:
    """Group runs into turns, newest root first.

    A run with no inbound dispatch continuation is a turn root. Child runs
    attach to the root that shares their `turn_id`. Runs without a `turn_id`
    (older single-run turns) are each their own root, keyed by run id.
    """
    roots: dict[str, RunInfo] = {}
    children_by_turn: dict[str, list[RunInfo]] = {}
    for run in runs:
        turn_key = run.turn_id or run.run_id
        if run.is_child and run.turn_id:
            children_by_turn.setdefault(turn_key, []).append(run)
        elif turn_key not in roots or run.started_epoch < roots[turn_key].started_epoch:
            roots[turn_key] = run

    groups: list[TurnGroup] = []
    for turn_key, root in roots.items():
        children = sorted(
            children_by_turn.get(turn_key, []),
            key=lambda item: item.started_epoch,
        )
        groups.append(TurnGroup(turn_id=root.turn_id or turn_key, root=root, children=children))
    groups.sort(key=lambda group: group.root.started_epoch, reverse=True)
    return groups


def dispatch_infos(events: list[EngineEvent]) -> list[DispatchInfo]:
    """Pair dispatch_entered with dispatch_returned by token, in entry order."""
    returned: dict[str, EngineEvent] = {}
    for event in events:
        if event.kind == _DISPATCH_RETURNED:
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
    """Resolve each dispatch's child run by target agent + temporal order.

    The Nth dispatch targeting agent X binds to the Nth child run whose
    `agent_id` is X (both ordered by time). In-process dispatches — targets
    with no spawned child run — keep `child_run_id = None`.
    """
    children_by_agent: dict[str, list[str]] = {}
    for child in sorted(children, key=lambda item: item.started_epoch):
        children_by_agent.setdefault(child.agent_id, []).append(child.run_id)

    cursor: dict[str, int] = {}
    resolved: list[DispatchInfo] = []
    for dispatch in sorted(dispatches, key=lambda item: item.entered_ts):
        target = dispatch.target_agent_instance_id
        available = children_by_agent.get(target, [])
        index = cursor.get(target, 0)
        child_run_id = available[index] if index < len(available) else None
        if child_run_id is not None:
            cursor[target] = index + 1
        resolved.append(
            DispatchInfo(
                **{
                    **dispatch.__dict__,
                    "child_run_id": child_run_id,
                }
            )
        )
    return resolved


def build_run_tree(group: TurnGroup, dispatches: list[DispatchInfo]) -> list[RunNode]:
    """Flat run tree: root first, then children with their dispatch linkage."""
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
    for child in group.children:
        dispatch = dispatch_by_child.get(child.run_id)
        nodes.append(
            RunNode(
                run_id=child.run_id,
                parent_run_id=dispatch.source_run_id if dispatch else group.root.run_id,
                parent_dispatch_id=dispatch.dispatch_token if dispatch else None,
                dispatch_mode=dispatch.dispatch_mode if dispatch else "",
                agent_instance_id=child.agent_id,
                agent_name=child.agent_id,
                status=child.status,
                started_at=child.started_at,
                ended_at=child.finished_at,
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

"""Orchestrate runtime reads into linked turn, segment, plan, and context DTOs.

The AG HTTP layer owns identity containment and opens the facade; this service
turns one connected runtime's reader into canonical operator UI DTOs. It never
touches storage directly — only through the injected `TraceReader`.
"""

from __future__ import annotations

from .context import build_context_snapshot
from .linking import (
    TurnGroup,
    build_graph,
    build_run_tree,
    dispatch_infos,
    group_turns,
    link_children,
    resolve_event_turn_ids,
)
from .models import (
    ContextSnapshot,
    PlanTimeline,
    ResourceSlotAssignmentTrace,
    ResourceSlotTrace,
    TraceSessionGroup,
    TurnDetail,
    TurnSummary,
)
from .plans import build_plan_timeline
from .reader import EngineEvent, TraceReader
from .segments import build_segments


class TraceProjectionService:
    """Build connected-runtime trace DTOs from an identity-contained reader."""

    def __init__(self, reader: TraceReader) -> None:
        self._reader = reader

    async def list_turn_groups(self) -> list[TraceSessionGroup]:
        """List identity-visible runtime sessions with root runs presented as turns.

        The projection groups dispatch children beneath their root and orders
        sessions and turns newest first without mutating runtime storage.

        Examples:
            List visible sessions:
                ```python
                groups = await projection.list_turn_groups()
                ```

            Read the newest turn:
                ```python
                newest = (await projection.list_turn_groups())[0].latest_turn
                ```

        Args:
            None.

        Returns:
            list[TraceSessionGroup]: Canonical session groups, or an empty list.

        Notes:
            Dispatch child runs never become independent turns.
        """
        runs = await self._reader.runs()
        if not runs:
            return []
        events = await self._reader.events_for_runs([run.run_id for run in runs])
        groups = group_turns(resolve_event_turn_ids(runs, events))
        events_by_run = _bucket_by_run(events)
        sessions: dict[str, list[TurnSummary]] = {}
        for group in groups:
            turn_events = _events_for_group(group, events_by_run)
            summary = _turn_summary(group, turn_events)
            sessions.setdefault(summary.session_id, []).append(summary)

        result: list[TraceSessionGroup] = []
        for session_id, turns in sessions.items():
            turns.sort(key=lambda turn: turn.started_at, reverse=True)
            result.append(
                TraceSessionGroup(
                    session_id=session_id,
                    latest_turn=turns[0],
                    turn_count=len(turns),
                    turns=turns,
                )
            )
        result.sort(key=lambda group: group.latest_turn.started_at, reverse=True)
        return result

    async def session_detail(self, session_id: str) -> TraceSessionGroup | None:
        """Read one exact identity-visible runtime session group.

        The method searches the canonical grouped view and performs no writes.

        Examples:
            Read an existing session:
                ```python
                group = await projection.session_detail("session-1")
                ```

            Detect a missing session:
                ```python
                assert await projection.session_detail("missing") is None
                ```

        Args:
            session_id: Exact AG runtime session identity.

        Returns:
            TraceSessionGroup | None: The group, or `None` when not visible or absent.

        Notes:
            Identity containment is applied by the injected `TraceReader`.
        """
        groups = await self.list_turn_groups()
        return next((group for group in groups if group.session_id == session_id), None)

    async def turn_detail(self, root_run_id: str) -> TurnDetail | None:
        """Project one root run and its dispatch children into a canonical turn.

        The result includes the run tree, observed graph, ordered agent segments,
        cycles, tools, dispatches, outcomes, and resource slot history.

        Examples:
            Read a turn:
                ```python
                detail = await projection.turn_detail("run-1")
                ```

            Detect a child or unknown route key:
                ```python
                assert await projection.turn_detail("unknown") is None
                ```

        Args:
            root_run_id: Root run used as the HTTP routing identity for the turn.

        Returns:
            TurnDetail | None: Canonical detail, or `None` when the root is unavailable.

        Notes:
            Turn membership is resolved from canonical engine `turn_id` values.
        """
        found = await self.find_group(root_run_id)
        if found is None:
            return None
        group, events = found
        dispatches = link_children(dispatch_infos(events), group.children)
        agent_names = _agent_names(events)
        segments = build_segments(
            events,
            child_run_ids={child.run_id for child in group.children},
            dispatches=dispatches,
            agent_names=agent_names,
        )
        runs = build_run_tree(group, dispatches)
        graph = build_graph(group.root.graph_id, events)
        summary = _turn_summary(group, events)
        return TurnDetail(
            turn=summary,
            runs=runs,
            graph=graph,
            segments=segments,
            resource_slots=_resource_slots(events),
        )

    async def plan_timeline(self, root_run_id: str) -> PlanTimeline | None:
        """Project plan and guideline snapshots across every run in a turn.

        Snapshots are normalized for the operator UI and retain their raw plan.

        Examples:
            Read a plan timeline:
                ```python
                timeline = await projection.plan_timeline("run-1")
                ```

            Detect an absent turn:
                ```python
                assert await projection.plan_timeline("missing") is None
                ```

        Args:
            root_run_id: Root run identity for the selected turn.

        Returns:
            PlanTimeline | None: Ordered snapshots, or `None` for an absent turn.

        Notes:
            An existing turn without plans returns an empty timeline.
        """
        found = await self.find_group(root_run_id)
        if found is None:
            return None
        _group, events = found
        return build_plan_timeline(events)

    async def context_snapshot(self, root_run_id: str, snapshot_id: str) -> ContextSnapshot | None:
        """Hydrate one prompt manifest proven to belong to a selected turn.

        Manifest content is read only after a decision event in the contained
        turn references the requested snapshot identity.

        Examples:
            Read a captured prompt:
                ```python
                snapshot = await projection.context_snapshot("run-1", "manifest-1")
                ```

            Reject an unrelated manifest:
                ```python
                assert await projection.context_snapshot("run-1", "other") is None
                ```

        Args:
            root_run_id: Root run identity for the selected turn.
            snapshot_id: Prompt manifest identity referenced by a decision event.

        Returns:
            ContextSnapshot | None: Hydrated snapshot, or `None` when unowned or absent.

        Notes:
            Body sections are present only for `manifest` or `full` captures.
        """
        found = await self.find_group(root_run_id)
        if found is None:
            return None
        _group, events = found
        if not any(
            event.kind == "agent_engine.decision"
            and str(event.data.get("prompt_manifest_id") or "") == snapshot_id
            for event in events
        ):
            return None
        manifest = await self._reader.prompt_manifest(snapshot_id)
        if manifest is None:
            return None
        return build_context_snapshot(events, snapshot_id, manifest)

    async def find_group(self, root_run_id: str) -> tuple[TurnGroup, list[EngineEvent]] | None:
        """Resolve one root run to its exact turn group and canonical events.

        The method repairs missing run metadata from event-level `turn_id`
        evidence and excludes unrelated runs in the same session.

        Examples:
            Resolve a group:
                ```python
                found = await projection.find_group("run-1")
                ```

            Detect a child-run route key:
                ```python
                assert await projection.find_group("child-run") is None
                ```

        Args:
            root_run_id: Exact root run identity.

        Returns:
            tuple[TurnGroup, list[EngineEvent]] | None: Group and ordered events,
            or `None` when the root is not visible or does not exist.

        Notes:
            This method is also used to scope turn-level logs and LLM calls.
        """
        runs = await self._reader.runs()
        if not runs:
            return None
        root = next(
            (run for run in runs if run.run_id == root_run_id and not run.is_child),
            None,
        )
        if root is None:
            return None
        candidates = [run for run in runs if run.session_id == root.session_id]
        candidate_events = await self._reader.events_for_runs([run.run_id for run in candidates])
        resolved = resolve_event_turn_ids(candidates, candidate_events)
        group = next(
            (item for item in group_turns(resolved) if item.root.run_id == root_run_id),
            None,
        )
        if group is None:
            return None
        run_ids = set(group.run_ids)
        events = [event for event in candidate_events if event.run_id in run_ids]
        return group, events


def _bucket_by_run(events: list[EngineEvent]) -> dict[str, list[EngineEvent]]:
    buckets: dict[str, list[EngineEvent]] = {}
    for event in events:
        buckets.setdefault(event.run_id, []).append(event)
    return buckets


def _events_for_group(
    group: TurnGroup, events_by_run: dict[str, list[EngineEvent]]
) -> list[EngineEvent]:
    collected: list[EngineEvent] = []
    for run_id in group.run_ids:
        collected.extend(events_by_run.get(run_id, []))
    return collected


def _turn_summary(group: TurnGroup, events: list[EngineEvent]) -> TurnSummary:
    agents = {
        event.agent_instance_id
        for event in events
        if event.agent_instance_id and event.agent_instance_id != "entry"
    }
    cycle_count = sum(1 for event in events if event.kind == "agent_engine.decision")
    tool_count = sum(1 for event in events if event.kind == "agent_engine.tool_call")
    dispatch_count = sum(1 for event in events if event.kind == "agent_engine.dispatch_entered")
    user_text = next(
        (event.text for event in events if event.kind == "agent_engine.user_request"),
        "",
    )
    return TurnSummary(
        root_run_id=group.root.run_id,
        turn_id=group.turn_id,
        session_id=group.root.session_id or group.root.run_id,
        started_at=group.root.started_at,
        ended_at=group.root.finished_at,
        status=group.root.status,
        user_text_preview=_preview(user_text),
        agent_count=len(agents),
        cycle_count=cycle_count,
        tool_count=tool_count,
        dispatch_count=dispatch_count,
        child_run_count=len(group.children),
        entry_agent_name=group.root.agent_id,
    )


def _agent_names(events: list[EngineEvent]) -> dict[str, str]:
    names: dict[str, str] = {}
    for event in events:
        agent_id = event.agent_instance_id
        if not agent_id:
            continue
        name = str(event.data.get("agent_id") or "") or agent_id
        names.setdefault(agent_id, name)
    return names


def _resource_slots(events: list[EngineEvent]) -> list[ResourceSlotTrace]:
    """Project bounded semantic slot history from Tool result events.

    Intro:
        Builds an observability read model without persisting a second slot or
        artifact graph in the connected runtime presenter.

    Examples:
        Project an empty turn:
            ```python
            assert _resource_slots([]) == []
            ```

    Args:
        events: Canonically ordered events for one complete turn tree.

    Returns:
        list[ResourceSlotTrace]: Current assignments and the last five versions.

    Notes:
        Resource content remains host-owned and is never copied into the trace.
    """
    assignments: dict[str, list[ResourceSlotAssignmentTrace]] = {}
    for event in events:
        if event.kind != "agent_engine.tool_result":
            continue
        for value in event.data.get("slot_assignments") or []:
            if not isinstance(value, dict):
                continue
            slot_key = str(value.get("slot_key") or "")
            resource = value.get("resource")
            try:
                slot_version = int(value.get("slot_version") or 0)
            except (TypeError, ValueError):
                continue
            if not slot_key or slot_version < 1 or not isinstance(resource, dict):
                continue
            assignments.setdefault(slot_key, []).append(
                ResourceSlotAssignmentTrace(
                    slot_version=slot_version,
                    resource=dict(resource),
                    assigned_by=str(value.get("assigned_by") or ""),
                    assigned_at=str(value.get("assigned_at") or event.iso),
                )
            )
    result: list[ResourceSlotTrace] = []
    for slot_key, values in sorted(assignments.items()):
        ordered = sorted(values, key=lambda item: item.slot_version)
        result.append(
            ResourceSlotTrace(
                slot_key=slot_key,
                current=ordered[-1],
                history=ordered[-5:],
            )
        )
    return result


def _preview(value: str, limit: int = 140) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["TraceProjectionService"]

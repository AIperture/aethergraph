from __future__ import annotations

from aethergraph.services.observability.trace_projection.linking import (
    DispatchInfo,
    build_run_tree,
    dispatch_infos,
    group_turns,
    link_children,
    resolve_event_turn_ids,
)
from aethergraph.services.observability.trace_projection.reader import (
    EngineEvent,
    RunInfo,
    _run_info,
)


def _run(
    run_id: str,
    started: int,
    *,
    turn_id: str = "",
    child: bool = False,
    dispatch_token: str = "",
    resumption: bool = False,
    owner_run_id: str = "",
    infrastructure: bool = False,
) -> RunInfo:
    return RunInfo(
        run_id=run_id,
        session_id="session-1",
        agent_id="worker" if child else "planner",
        graph_id="graph",
        status="succeeded",
        started_at=f"2026-01-01T00:00:{started:02d}+00:00",
        finished_at=f"2026-01-01T00:00:{started + 1:02d}+00:00",
        turn_id=turn_id,
        is_child=child,
        dispatch_token=dispatch_token,
        is_resumption=resumption,
        resume_owner_run_id=owner_run_id,
        is_infrastructure=infrastructure,
        has_engine_events=False,
        source_agent_instance_id="planner" if child else "",
        target_agent_instance_id="worker" if child else "",
    )


def _event(run_id: str, turn_id: str, ts: int) -> EngineEvent:
    return EngineEvent(
        event_id=f"event-{run_id}",
        ts=float(ts),
        iso=str(ts),
        run_id=run_id,
        session_id="session-1",
        kind="agent_engine.agent_entered",
        text="entered",
        agent_instance_id="planner",
        turn_id=turn_id,
        tags=["agent_engine", f"turn:{turn_id}"],
        data={"turn_id": turn_id},
    )


def test_run_reader_extracts_existing_turn_and_dispatch_metadata() -> None:
    child = _run_info(
        {
            "run_id": "child",
            "session_id": "session-1",
            "agent_id": "worker",
            "graph_id": "worker-graph",
            "status": "succeeded",
            "started_at": "2026-01-01T00:00:01+00:00",
            "meta": {
                "original_inputs": {
                    "continuation_payload": {
                        "kind": "subagent_call",
                        "source_agent_instance_id": "planner",
                        "target_agent_instance_id": "worker",
                        "payload": {
                            "turn_id": "turn-1",
                            "dispatch_intent": {"dispatch_token": "dispatch-1"},
                        },
                    }
                }
            },
        }
    )
    notifier = _run_info(
        {
            "run_id": "notifier",
            "graph_id": "aethergraph_engine._internal_completion_notifier",
            "tags": ["aethergraph_engine._internal", "notifier"],
        }
    )

    assert child.turn_id == "turn-1"
    assert child.dispatch_token == "dispatch-1"
    assert child.is_child is True
    assert notifier.is_infrastructure is True


def test_async_resumptions_reuse_owner_turn_and_empty_internal_runs_are_hidden() -> None:
    runs = [
        _run("root", 1),
        _run("child-1", 2, child=True, dispatch_token="dispatch-1"),
        _run("notifier", 3, infrastructure=True),
        _run("background", 4),
        _run("resume-1", 5, resumption=True, owner_run_id="root"),
        _run("child-2", 6, child=True, dispatch_token="dispatch-2"),
        _run("resume-2", 7, resumption=True, owner_run_id="resume-1"),
    ]
    events = [
        _event("root", "turn-original", 1),
        _event("child-1", "turn-original", 2),
        _event("resume-1", "turn-resume-1", 5),
        _event("child-2", "turn-resume-1", 6),
        _event("resume-2", "turn-resume-2", 7),
    ]

    resolved = resolve_event_turn_ids(runs, events)
    groups = group_turns(resolved)

    assert [group.turn_id for group in groups] == ["turn-original"]
    group = groups[0]
    assert group.root.run_id == "root"
    assert [child.run_id for child in group.children] == ["child-1", "child-2"]
    assert [run.run_id for run in group.resumptions] == ["resume-1", "resume-2"]
    assert group.run_ids == ["root", "child-1", "resume-1", "child-2", "resume-2"]


def test_dispatch_children_link_by_token_and_resumptions_link_to_owner_run() -> None:
    group = group_turns(
        [
            _run("root", 1, turn_id="turn-1"),
            _run(
                "child-second",
                2,
                turn_id="turn-1",
                child=True,
                dispatch_token="dispatch-2",
            ),
            _run(
                "child-first",
                3,
                turn_id="turn-1",
                child=True,
                dispatch_token="dispatch-1",
            ),
            _run(
                "resume",
                4,
                turn_id="turn-1",
                resumption=True,
                owner_run_id="root",
            ),
        ]
    )[0]
    dispatches = [
        DispatchInfo(
            dispatch_token=token,
            source_run_id="root",
            source_agent_instance_id="planner",
            target_agent_instance_id="worker",
            dispatch_mode="async",
            instruction="work",
            status="completed",
            return_text="done",
            required_slots=[],
            expected_slots=[],
            entered_event_id=f"event-{token}",
            started_at="1",
            ended_at="2",
            entered_ts=float(index),
        )
        for index, token in enumerate(["dispatch-1", "dispatch-2"], start=1)
    ]

    linked = link_children(dispatches, group.children)
    tree = build_run_tree(group, linked)

    assert {item.dispatch_token: item.child_run_id for item in linked} == {
        "dispatch-1": "child-first",
        "dispatch-2": "child-second",
    }
    resume = next(node for node in tree if node.run_id == "resume")
    assert resume.parent_run_id == "root"
    assert resume.parent_dispatch_id is None


def test_dispatch_info_pairs_parent_return_intent_by_exact_token() -> None:
    def dispatch_event(
        event_id: str,
        kind: str,
        ts: int,
        text: str,
        data: dict[str, object],
    ) -> EngineEvent:
        return EngineEvent(
            event_id=event_id,
            ts=float(ts),
            iso=f"2026-01-01T00:00:0{ts}+00:00",
            run_id="root",
            session_id="session-1",
            kind=kind,
            text=text,
            agent_instance_id="planner",
            turn_id="turn-1",
            tags=["agent_engine", "turn:turn-1"],
            data=data,
        )

    [dispatch] = dispatch_infos(
        [
            dispatch_event(
                "entered",
                "agent_engine.dispatch_entered",
                1,
                "Inspect",
                {"dispatch_token": "dispatch-1", "status": "dispatched"},
            ),
            dispatch_event(
                "returned",
                "agent_engine.return_intent",
                2,
                "Inspection complete",
                {"dispatch_token": "dispatch-1", "status": "completed"},
            ),
        ]
    )

    assert dispatch.status == "completed"
    assert dispatch.return_text == "Inspection complete"
    assert dispatch.ended_at == "2026-01-01T00:00:02+00:00"

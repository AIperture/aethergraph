from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.services.planning import (
    AgentContextSnapshot,
    PlanningContext,
    PlanningContextBuilderProtocol,
    PlanningEvent,
    RoutedIntent,
    SessionState,
    ValidationResult,
)
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
from aethergraph.services.planning.plan_executor import (
    ExecutionEvent,
    ExecutionResult,
    PlanExecutor,
)
from aethergraph.services.planning.plan_types import CandidatePlan
from aethergraph.services.planning.planner import ActionPlanner


def _session_key(agent_id: str | None, session_id: str | None) -> str:
    """
    Produce a stable in-memory key for (agent_id, session_id).
    """
    return f"{agent_id or 'default_agent'}::{session_id or 'default_session'}"


@dataclass
class PlanRecord:
    """
    Minimal record of a planning attempt, for feeding back into snapshots.
    """

    plan: CandidatePlan
    validation: ValidationResult
    context: PlanningContext


@dataclass
class ExecutionRecord:
    """
    Minimal record of a plan execution, for feeding back into snapshots.

    We keep things intentionally loose (dicts) so verticals can decide
    what they want to extract later (metrics, inputs, etc.).
    """

    ok: bool
    flow_ids: list[str]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    error: str | None = None
    # Optional: tie executions back to plans
    plan_id: str | None = None


@dataclass
class PlanningService:
    """
    High-level service that:

    - builds a PlanningContext (via a builder),
    - calls ActionPlanner.plan_with_loop,
    - optionally executes the plan with PlanExecutor,
    - maintains per-(agent, session) AgentContextSnapshot in memory.

    This is the "glue" between your orchestrator and the lower-level
    planner/executor primitives.
    """

    context_builder: PlanningContextBuilderProtocol
    planner: ActionPlanner
    executor: PlanExecutor

    # In-memory session snapshots keyed by (agent_id, session_id)
    _snapshots: dict[str, AgentContextSnapshot] = field(default_factory=dict)

    # How many history items to keep per snapshot
    max_history: int = 5

    # Public API

    async def plan_and_maybe_execute(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
        agent_id: str | None = None,
        session_id: str | None = None,
        identity: RequestIdentity,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        origin: RunOrigin | None = None,
        allow_execute: bool = False,
        # If provided, this overrides ctx.user_inputs for execution
        execution_user_inputs: dict[str, Any] | None = None,
        # optional callbacks for events
        on_plan_event: Callable[[PlanningEvent], Any] | None = None,
        on_exec_event: Callable[[ExecutionEvent], Any] | None = None,
        # optional skill id hint passed to context builder
        skill_id: str | None = None,
    ) -> tuple[
        CandidatePlan | None, ValidationResult | None, ExecutionResult | None, AgentContextSnapshot
    ]:
        """
        One-shot helper:

        1) Build PlanningContext (with prior snapshot).
        2) Run planner.plan_with_loop.
        3) Update AgentContextSnapshot with plan + validation.
        4) Optionally execute the plan if validation.ok.
        5) Update snapshot with execution record.

        Returns:
            (plan, last_validation, execution_result, updated_snapshot)
        """

        key = _session_key(agent_id, session_id)
        snapshot = self._snapshots.get(key)

        # 1) Build PlanningContext
        ctx = await self._build_context(
            user_message=user_message,
            routed=routed,
            session_state=session_state,
            snapshot=snapshot,
            skill_id=skill_id,
        )

        # 2) Run planner
        plan, history = await self.planner.plan_with_loop(
            ctx,
            on_event=on_plan_event,
        )

        last_validation = history[-1] if history else None

        # 3) Update snapshot with plan
        snapshot = self._update_snapshot_with_plan(
            old_snapshot=snapshot,
            ctx=ctx,
            plan=plan,
            validation=last_validation,
        )
        self._snapshots[key] = snapshot

        exec_result: ExecutionResult | None = None

        # 4) Optionally execute on fully valid plans
        if allow_execute and plan is not None and last_validation and last_validation.ok:
            effective_user_inputs = execution_user_inputs or (ctx.user_inputs or {})
            exec_result = await self.executor.execute(
                plan,
                user_inputs=effective_user_inputs,
                on_event=on_exec_event,
                identity=identity,
                visibility=visibility,
                importance=importance,
                origin=origin,
                session_id=session_id,
                agent_id=agent_id,
            )

            # 5) Update snapshot with execution
            snapshot = self._update_snapshot_with_execution(
                old_snapshot=snapshot,
                ctx=ctx,
                plan=plan,
                exec_result=exec_result,
            )
            self._snapshots[key] = snapshot

        return plan, last_validation, exec_result, snapshot

    def get_snapshot(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> AgentContextSnapshot | None:
        """
        Retrieve the current AgentContextSnapshot for (agent_id, session_id).
        """
        key = _session_key(agent_id, session_id)
        return self._snapshots.get(key)

    def clear_snapshot(
        self,
        *,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Drop the snapshot for a given (agent, session).
        """
        key = _session_key(agent_id, session_id)
        self._snapshots.pop(key, None)

    # Internal helpers
    async def _build_context(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
        snapshot: AgentContextSnapshot | None,
        skill_id: str | None,
    ) -> PlanningContext:
        """
        Delegate to the vertical-specific PlanningContextBuilder.

        We pass in the previous AgentContextSnapshot so the builder
        can create memory_snippets, infer default inputs, etc.
        """
        # Some builders may not accept `snapshot` or `skill_id` yet.
        # To avoid forcing a breaking change, we try the extended
        # signature first, then fall back to simpler forms.
        builder = self.context_builder

        # Preferred: full featured builder (like SurrogatePlanningContextBuilder)
        if hasattr(builder, "build"):
            try:
                return await builder.build(
                    user_message=user_message,
                    routed=routed,
                    session_state=session_state,
                    snapshot=snapshot,
                    skill_id=skill_id,
                )
            except TypeError:
                # Fall back to older signature
                return await builder.build(
                    user_message=user_message,
                    routed=routed,
                    session_state=session_state,
                )

        raise RuntimeError("PlanningContextBuilderProtocol implementation has no 'build' method")

    def _update_snapshot_with_plan(
        self,
        *,
        old_snapshot: AgentContextSnapshot | None,
        ctx: PlanningContext,
        plan: CandidatePlan | None,
        validation: ValidationResult | None,
    ) -> AgentContextSnapshot:
        """
        Produce a new AgentContextSnapshot that includes the latest plan
        and a view of the current session state (user_inputs).
        """
        # Start from existing snapshot or an empty one
        snap = old_snapshot or AgentContextSnapshot(
            last_executions=[],
            last_plans=[],
            summaries=[],
            session_state_view={},
        )

        # Update session_state_view with what we know from ctx.user_inputs
        sview = dict(snap.session_state_view or {})
        for k, v in (ctx.user_inputs or {}).items():
            sview[k] = v

        last_plans = list(snap.last_plans or [])

        if plan is not None:
            # Distill a compact dict for planning history
            flow_ids = ctx.flow_ids or []
            plan_dict = plan.to_dict()
            intended_actions = [step.get("action") for step in plan_dict.get("steps", [])]

            last_plans.append(
                {
                    "flow_id": flow_ids[0] if flow_ids else None,
                    "plan": plan_dict,
                    "intended_actions": intended_actions,
                    "validation": {
                        "ok": validation.ok if validation else None,
                        "missing_user_bindings": getattr(validation, "missing_user_bindings", None),
                        "has_structural_errors": getattr(validation, "has_structural_errors", None),
                    }
                    if validation
                    else None,
                }
            )

        # Trim history
        if len(last_plans) > self.max_history:
            last_plans = last_plans[-self.max_history :]

        return AgentContextSnapshot(
            last_executions=list(snap.last_executions or []),
            last_plans=last_plans,
            summaries=list(snap.summaries or []),
            session_state_view=sview,
        )

    def _update_snapshot_with_execution(
        self,
        *,
        old_snapshot: AgentContextSnapshot | None,
        ctx: PlanningContext,
        plan: CandidatePlan,
        exec_result: ExecutionResult,
    ) -> AgentContextSnapshot:
        """
        Produce a new AgentContextSnapshot that includes the latest execution
        record, plus any updated session_state_view if you want to fold in
        outputs (optional for v1).
        """
        snap = old_snapshot or AgentContextSnapshot(
            last_executions=[],
            last_plans=[],
            summaries=[],
            session_state_view={},
        )

        last_execs = list(snap.last_executions or [])

        # For v1 we store a loose dict; verticals can decide what to put here.
        flow_ids = ctx.flow_ids or []
        exec_record = {
            "flow_id": flow_ids[0] if flow_ids else None,
            "inputs": ctx.user_inputs or {},
            "outputs": exec_result.outputs or {},
            "ok": exec_result.ok,
            "error": (exec_result.errors[-1].message if exec_result.errors else None),
        }
        last_execs.append(exec_record)

        if len(last_execs) > self.max_history:
            last_execs = last_execs[-self.max_history :]

        # For now, we keep session_state_view as-is; if desired, we could
        # also promote some outputs into the state view here.

        return AgentContextSnapshot(
            last_executions=last_execs,
            last_plans=list(snap.last_plans or []),
            summaries=list(snap.summaries or []),
            session_state_view=dict(snap.session_state_view or {}),
        )

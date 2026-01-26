from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

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
from aethergraph.services.planning.input_parser import InputParser
from aethergraph.services.planning.plan_executor import (
    ExecutionEvent,
    ExecutionResult,
    PlanExecutor,
)
from aethergraph.services.planning.plan_types import CandidatePlan
from aethergraph.services.planning.planner import ActionPlanner

PlanningTurnKind = Literal["new", "clarification", "approval"]


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
class PlanningTurnResult:
    kind: PlanningTurnKind
    plan: CandidatePlan | None
    validation: ValidationResult | None
    execution: ExecutionResult | None
    snapshot: AgentContextSnapshot

    # For UX: what should the outer agent say next?
    needs_clarification: bool = False
    clarification_keys: list[str] = field(default_factory=list)
    clarification_message: str | None = None

    needs_approval: bool = False
    approval_message: str | None = None

    # hint for the orchestrator/router
    fallback_to_router: bool = False
    fallback_reason: str | None = None


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

    # Optional: LLM-backed parser for clarification replies
    input_parser: InputParser | None = None

    # In-memory session snapshots keyed by (agent_id, session_id)
    _snapshots: dict[str, AgentContextSnapshot] = field(default_factory=dict)

    # How many history items to keep per snapshot
    max_history: int = 5

    # Public API

    async def handle_turn(
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
        allow_auto_execute: bool = False,
        skill_id: str | None = None,
        on_plan_event: Callable[[PlanningEvent], Any] | None = None,
        on_exec_event: Callable[[ExecutionEvent], Any] | None = None,
        base_snapshot: AgentContextSnapshot | None = None,
    ) -> PlanningTurnResult:
        """
        High-level, single entry point for planning-related turns.

        The router only needs to decide that this is a "planning or
        surrogate" kind of intent and then call handle_turn. This method
        decides whether the user is:

          - starting a new goal, OR
          - responding to a clarification question (missing inputs), OR
          - responding with approval / tweaks to an existing plan.

        It uses AgentContextSnapshot to infer this and then calls the
        lower-level planner/executor as needed.
        """
        key = _session_key(agent_id, session_id)

        # 1) Merge orchestrator-provided base snapshot with our stored planning snapshot.
        stored = self._snapshots.get(key)

        if stored is None and base_snapshot is not None:
            # First time we see this (agent,session): start from orchestrator snapshot
            snapshot = base_snapshot
        elif stored is not None and base_snapshot is not None:
            # Merge: keep planning state from stored, refresh memory-ish fields from base
            snapshot = stored

            # Overwrite memory-related fields from base_snapshot, keep pending_ and last_* from stored.
            snapshot.recent_chat = base_snapshot.recent_chat or []
            snapshot.summaries = base_snapshot.summaries or []
            snapshot.session_state_view = (
                base_snapshot.session_state_view or snapshot.session_state_view
            )
            # You can also merge last_plans / last_executions if memory records them differently,
            # but simplest is to treat PlanningService as the owner of last_plans/last_executions.
        else:
            snapshot = stored or base_snapshot or AgentContextSnapshot(session_id=session_id)

        self._snapshots[key] = snapshot

        # --- 1) If we know we're waiting on missing inputs, treat as clarification.
        if (
            snapshot
            and snapshot.pending_mode == "clarification"
            and snapshot.pending_missing_inputs
        ):
            return await self._handle_clarification_turn(
                user_message=user_message,
                routed=routed,
                session_state=session_state,
                snapshot=snapshot,
                agent_id=agent_id,
                session_id=session_id,
                identity=identity,
                visibility=visibility,
                importance=importance,
                origin=origin,
                allow_auto_execute=allow_auto_execute,
                skill_id=skill_id,
                on_plan_event=on_plan_event,
                on_exec_event=on_exec_event,
            )

        # --- 2) If we have a pending fully-valid plan but no execution yet,
        # we *could* treat this as an approval / modification turn.
        if snapshot and snapshot.pending_mode == "approval" and snapshot.pending_plan:
            # Simple v1 strategy: run planning again with enriched history
            # rather than special-casing "just run the old plan".
            # That way phrases like "run the same plan but skip evaluation"
            # can be handled by the regular planner using last_plans + memory.
            # So we fall through to the "new" path below.
            pass

        # --- 3) Default: treat as a new planning turn.
        plan, validation, exec_result, new_snapshot = await self.plan_and_maybe_execute(
            user_message=user_message,
            routed=routed,
            session_state=session_state,
            agent_id=agent_id,
            session_id=session_id,
            identity=identity,
            visibility=visibility,
            importance=importance,
            origin=origin,
            allow_execute=allow_auto_execute,
            execution_user_inputs=None,
            on_plan_event=on_plan_event,
            on_exec_event=on_exec_event,
            skill_id=skill_id,
        )

        # Derive clarification/approval hints from validation
        needs_clar, clar_keys, clar_msg = self._derive_clarification_from_validation(
            validation=validation,
        )

        # Update pending_* fields on snapshot accordingly
        new_snapshot = self._update_pending_state_after_planning(
            old_snapshot=new_snapshot,
            plan=plan,
            validation=validation,
            clarification_keys=clar_keys,
        )
        self._snapshots[key] = new_snapshot

        fallback_to_router, fallback_reason = self._should_fallback_to_router(
            plan=plan,
            validation=validation,
        )

        return PlanningTurnResult(
            kind="new",
            plan=plan,
            validation=validation,
            execution=exec_result,
            snapshot=new_snapshot,
            needs_clarification=needs_clar,
            clarification_keys=clar_keys,
            clarification_message=clar_msg,
            needs_approval=bool(plan and validation and validation.ok and not allow_auto_execute),
            approval_message=(
                "I have prepared a workflow plan. Do you want me to run it now?"
                if plan and validation and validation.ok and not allow_auto_execute
                else None
            ),
            fallback_to_router=fallback_to_router,
            fallback_reason=fallback_reason,
        )

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
        existing_snapshot: AgentContextSnapshot | None = None,  # NEW
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
        snapshot = existing_snapshot or self._snapshots.get(key)

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
        Produce a new AgentContextSnapshot that includes:

        - updated session_state_view from ctx.user_inputs
        - appended last_plans entry (compact)
        """
        snap = old_snapshot or AgentContextSnapshot()

        # --- 1) Update session_state_view with ctx.user_inputs
        sview = dict(snap.session_state_view or {})
        for k, v in (ctx.user_inputs or {}).items():
            sview[k] = v

        snap.session_state_view = sview

        # --- 2) Append planning history entry
        last_plans = list(snap.last_plans or [])

        if plan is not None:
            plan_dict = plan.to_dict()
            flow_ids = ctx.flow_ids or []
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

        snap.last_plans = last_plans
        return snap

    def _update_snapshot_with_execution(
        self,
        *,
        old_snapshot: AgentContextSnapshot | None,
        ctx: PlanningContext,
        plan: CandidatePlan,
        exec_result: ExecutionResult,
    ) -> AgentContextSnapshot:
        """
        Record an execution and clear pending state (we've just run the plan).
        """
        snap = old_snapshot or AgentContextSnapshot()

        last_execs = list(snap.last_executions or [])

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

        snap.last_executions = last_execs

        # For now we keep session_state_view unchanged; you could promote outputs here.

        # Clear pending state, since we just executed the plan
        snap.pending_plan = None
        snap.pending_missing_inputs = {}
        snap.pending_question = None
        snap.pending_mode = None

        return snap

    async def _handle_clarification_turn(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
        snapshot: AgentContextSnapshot,
        agent_id: str | None,
        session_id: str | None,
        identity: RequestIdentity,
        visibility: RunVisibility,
        importance: RunImportance,
        origin: RunOrigin | None,
        allow_auto_execute: bool,
        skill_id: str | None,
        on_plan_event: Callable[[PlanningEvent], Any] | None,
        on_exec_event: Callable[[ExecutionEvent], Any] | None,
    ) -> PlanningTurnResult:
        """
        Handle a turn where we expect the user to supply missing inputs
        (clarification) based on snapshot.pending_missing_inputs.
        """
        key = _session_key(agent_id, session_id)
        missing = snapshot.pending_missing_inputs or {}

        if not self.input_parser or not missing:
            # Fallback: treat as a fresh planning turn directly.
            plan, validation, exec_result, new_snapshot = await self.plan_and_maybe_execute(
                user_message=user_message,
                routed=routed,
                session_state=session_state,
                agent_id=agent_id,
                session_id=session_id,
                identity=identity,
                visibility=visibility,
                importance=importance,
                origin=origin,
                allow_execute=allow_auto_execute,
                execution_user_inputs=None,
                on_plan_event=on_plan_event,
                on_exec_event=on_exec_event,
                skill_id=skill_id,
            )

            needs_clar, clar_keys, clar_msg = self._derive_clarification_from_validation(
                validation=validation,
            )
            new_snapshot = self._update_pending_state_after_planning(
                old_snapshot=new_snapshot,
                plan=plan,
                validation=validation,
                clarification_keys=clar_keys,
            )
            self._snapshots[key] = new_snapshot

            return PlanningTurnResult(
                kind="new",
                plan=plan,
                validation=validation,
                execution=exec_result,
                snapshot=new_snapshot,
                needs_clarification=needs_clar,
                clarification_keys=clar_keys,
                clarification_message=clar_msg,
                needs_approval=bool(
                    plan and validation and validation.ok and not allow_auto_execute
                ),
                approval_message=(
                    "I have prepared a workflow plan. Do you want me to run it now?"
                    if plan and validation and validation.ok and not allow_auto_execute
                    else None
                ),
            )

        missing_keys = list(missing.keys())

        # 1) Use InputParser to extract values from the user_message.
        parsed = await self.input_parser.parse_message_for_fields(
            message=user_message,
            missing_keys=missing_keys,
            # If you have a skill registry, you could look it up here via snapshot.active_skill_id
            skill=None,
        )

        # 2) Merge parsed values into snapshot.pending_user_inputs & session_state_view
        pending_user_inputs = dict(snapshot.pending_user_inputs or {})
        pending_user_inputs.update(parsed.values)

        session_state_view = dict(snapshot.session_state_view or {})
        session_state_view.update(parsed.values)

        # 3) Compute which keys remain missing
        still_missing = {k: missing[k] for k in missing_keys if k in parsed.missing_keys}

        # 4) Update snapshot's pending state
        snapshot.pending_user_inputs = pending_user_inputs
        snapshot.session_state_view = session_state_view
        snapshot.pending_missing_inputs = still_missing

        # Persist the updated snapshot *for both* success and error paths
        self._snapshots[key] = snapshot

        # If there are parsing errors or still-missing fields, we do NOT replan yet.
        if parsed.errors or still_missing:
            clarification_message = "\n".join(parsed.errors or []) or None
            self._snapshots[key] = snapshot
            return PlanningTurnResult(
                kind="clarification",
                plan=None,
                validation=None,
                execution=None,
                snapshot=snapshot,
                needs_clarification=True,
                clarification_keys=list(still_missing.keys()),
                clarification_message=clarification_message,
            )

        # 5) All required inputs resolved → treat as a "new" planning turn,
        # but with enriched user_inputs via the context builder (through snapshot).
        # The builder can read snapshot.session_state_view and snapshot.pending_user_inputs.
        plan, validation, exec_result, new_snapshot = await self.plan_and_maybe_execute(
            user_message=user_message,
            routed=routed,
            session_state=session_state,
            agent_id=agent_id,
            session_id=session_id,
            identity=identity,
            visibility=visibility,
            importance=importance,
            origin=origin,
            allow_execute=allow_auto_execute,
            execution_user_inputs=None,  # builder will incorporate the updated session_state_view
            on_plan_event=on_plan_event,
            on_exec_event=on_exec_event,
            skill_id=skill_id,
            existing_snapshot=snapshot,
        )

        needs_clar, clar_keys, clar_msg = self._derive_clarification_from_validation(
            validation=validation,
        )
        new_snapshot = self._update_pending_state_after_planning(
            old_snapshot=new_snapshot,
            plan=plan,
            validation=validation,
            clarification_keys=clar_keys,
        )
        self._snapshots[key] = new_snapshot

        return PlanningTurnResult(
            kind="clarification",
            plan=plan,
            validation=validation,
            execution=exec_result,
            snapshot=new_snapshot,
            needs_clarification=needs_clar,
            clarification_keys=clar_keys,
            clarification_message=clar_msg,
            needs_approval=bool(plan and validation and validation.ok and not allow_auto_execute),
            approval_message=(
                "I have updated the workflow plan. Do you want me to run it now?"
                if plan and validation and validation.ok and not allow_auto_execute
                else None
            ),
        )

    @staticmethod
    def _derive_clarification_from_validation(
        validation: ValidationResult | None,
    ) -> tuple[bool, list[str], str | None]:
        if not validation:
            return False, [], None

        missing = getattr(validation, "missing_user_bindings", {}) or {}
        if not missing:
            return False, [], None

        keys = sorted(missing.keys())
        msg_lines = [
            "I have prepared a workflow, but I still need the following inputs:",
            *[f"- {k}" for k in keys],
            "Please provide values for each of them.",
        ]
        return True, keys, "\n".join(msg_lines)

    def _update_pending_state_after_planning(
        self,
        *,
        old_snapshot: AgentContextSnapshot,
        plan: CandidatePlan | None,
        validation: ValidationResult | None,
        clarification_keys: list[str],
    ) -> AgentContextSnapshot:
        snap = old_snapshot

        # Reset pending state by default
        snap.pending_plan = None
        snap.pending_user_inputs = dict(snap.pending_user_inputs or {})
        snap.pending_missing_inputs = {}
        snap.pending_question = None
        snap.pending_mode = None

        if plan is not None:
            snap.pending_plan = plan.to_dict()

        if validation and clarification_keys:
            # We are in a clarification-needed state
            missing_bindings = getattr(validation, "missing_user_bindings", {}) or {}
            snap.pending_missing_inputs = {
                k: missing_bindings.get(k, []) for k in clarification_keys
            }
            snap.pending_mode = "clarification"
            snap.pending_question = (
                "I need the following inputs before I can run this workflow:\n"
                + "\n".join(f"- {k}" for k in clarification_keys)
            )
        elif validation and validation.ok and plan is not None:
            # Plan is valid but we may want approval before execution.
            snap.pending_mode = "approval"
            snap.pending_question = "I have a valid workflow plan. Should I run it?"

        return snap

    @staticmethod
    def _should_fallback_to_router(
        plan: CandidatePlan | None,
        validation: ValidationResult | None,
    ) -> tuple[bool, str | None]:
        """
        Decide whether this turn is a bad fit for planning and should be
        handed back to the general agent/router.

        Heuristic:
          - No plan at all -> fallback
          - No validation -> fallback
          - Validation has structural errors -> fallback
          - Invalid and no missing_user_bindings -> fallback
            (i.e. it's not just "please give me dataset_path")
        """
        if plan is None:
            return True, "no_plan_produced"

        if validation is None:
            return True, "no_validation"

        has_structural = getattr(validation, "has_structural_errors", False)
        missing = getattr(validation, "missing_user_bindings", {}) or {}

        if has_structural:
            return True, "structural_errors_in_plan"

        if not validation.ok and not missing:
            # Not executable, and not just waiting on user inputs = planner is confused
            return True, "invalid_plan_without_missing_inputs"

        return False, None

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import TYPE_CHECKING, Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunVisibility,
)

if TYPE_CHECKING:
    from aethergraph.core.runtime.run_manager import RunManager

from .action_catalog import ActionCatalog
from .input_parser import InputParser, ParsedInputs
from .plan_executor import (
    ExecutionResult,
    PlanExecutor,
)
from .plan_types import (
    CandidatePlan,
    ExecutionEventCallback,
    PlanningContext,
    PlanningEvent,
    PlanningEventCallback,
    SkillSpec,
    ValidationResult,
)
from .planner import ActionPlanner


@dataclass
class PlanResult:
    plan: CandidatePlan | None
    validation: ValidationResult | None
    events: list[PlanningEvent] = field(default_factory=list)


@dataclass
class PlannerService:
    """
    High-level planning facade exposed as `context.planner()`.

    Internally composes:
      - ActionCatalog
      - ActionPlanner
      - PlanExecutor
      - InputParser
    """

    catalog: ActionCatalog
    llm: LLMClientProtocol
    validator: Any  # FlowValidator
    run_manager: RunManager | None = None  # for executing plans

    # Lazily initialized components
    _planner: ActionPlanner | None = None
    _executor: PlanExecutor | None = None
    _input_parser: InputParser | None = None

    @property
    def planner_core(self) -> ActionPlanner:
        if self._planner is None:
            self._planner = ActionPlanner(
                catalog=self.catalog,
                validator=self.validator,
                llm=self.llm,
            )
        return self._planner

    @property
    def executor(self) -> PlanExecutor:
        if self._executor is None:
            self._executor = PlanExecutor(catalog=self.catalog, run_manager=self.run_manager)
        return self._executor

    @property
    def input_parser(self) -> InputParser:
        if self._input_parser is None:
            self._input_parser = InputParser(llm=self.llm)
        return self._input_parser

    # ------------- Planning API -------------
    async def plan(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,  # advanced use with IO slots
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = True,
        preferred_external_keys: list[str] | None = None,
        skill: SkillSpec | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        on_event: PlanningEventCallback | None = None,
    ) -> PlanResult:
        """
        Ergonomic entrypoint: build a PlanningContext and delegate to plan_with_context().
        """
        ctx = PlanningContext(
            goal=goal,
            user_inputs=user_inputs or {},
            external_slots=external_slots or {},
            memory_snippets=list(memory_snippets or []),
            artifact_snippets=list(artifact_snippets or []),
            flow_ids=list(flow_ids) if flow_ids is not None else None,
            instruction=instruction,
            skill=skill,
            allow_partial_plans=allow_partial,
            preferred_external_keys=list(preferred_external_keys or []),
        )
        return await self.plan_with_context(ctx, on_event=on_event)

    async def plan_with_context(
        self,
        ctx: PlanningContext,
        *,
        on_event: PlanningEventCallback | None = None,
    ) -> PlanResult:
        """
        Plan towards the goal in the given context.
        Emits PlanningEvents via on_event callback.
        """
        events: list[PlanningEvent] = []

        async def _capture(ev: PlanningEvent) -> None:
            # Always record locally
            events.append(ev)

            # Forward to caller if provided
            if on_event:
                result = on_event(ev)
                if inspect.isawaitable(result):
                    await result

        plan, history = await self.planner_core.plan_with_loop(
            ctx,
            on_event=_capture,
        )

        validation: ValidationResult | None
        validation = history[-1] if history else None

        return PlanResult(plan=plan, validation=validation, events=events)

    # ------------- input parsing API -------------
    async def parse_inputs(
        self,
        *,
        message: str,
        missing_keys: list[str],
        skill: SkillSpec | None = None,
        instruction: str | None = None,  # currently unused, keep for future use
    ) -> ParsedInputs:
        """
        Parse user message into structured inputs for the given skill.
        """
        # For now we ignore `instruction` and rely on skill/meta,
        # We can thread it into the system prompt later.
        return await self.input_parser.parse_message_for_fields(
            message=message,
            missing_keys=missing_keys,
            skill=skill,
        )

    async def execute_plan(
        self,
        plan: CandidatePlan,
        *,
        user_inputs: dict[str, Any] | None = None,
        on_event: ExecutionEventCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> ExecutionResult:
        return await self.executor.execute(
            plan,
            user_inputs=user_inputs,
            on_event=on_event,
            identity=identity,
            visibility=visibility,
            importance=importance,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            tags=tags,
            origin=origin,
        )

    async def plan_and_execute(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = False,
        preferred_external_keys: list[str] | None = None,
        skill: SkillSpec | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        planning_events_cb: PlanningEventCallback | None = None,
        execution_events_cb: ExecutionEventCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility | None = RunVisibility.normal,
        importance: RunImportance | None = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> tuple[PlanResult, ExecutionResult | None]:
        """
        Convenience: plan first, then execute if plan is acceptable.
        """
        plan_result = await self.plan(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            skill=skill,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            on_event=planning_events_cb,
        )

        plan = plan_result.plan
        validation = plan_result.validation

        if plan is None or validation is None:
            return plan_result, None

        accept_partial = allow_partial and validation.is_partial_ok()
        if not validation.ok and not accept_partial:
            # Invalid plan and partial not allowed → do not execute
            return plan_result, None

        exec_result = await self.execute_plan(
            plan,
            user_inputs=user_inputs,
            on_event=execution_events_cb,
            identity=identity,
            visibility=visibility,
            importance=importance,
            session_id=session_id,
            agent_id=agent_id,
            app_id=app_id,
            tags=tags,
            origin=origin,
        )

        return plan_result, exec_result

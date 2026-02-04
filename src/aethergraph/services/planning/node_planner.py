from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility

from .plan_executor import BackgroundExecutionHandle, ExecutionEvent, ExecutionResult
from .plan_types import CandidatePlan, PlanningContext, PlanningEvent
from .planner_service import PlannerService, PlanResult

PlanEventsCallback = Callable[[PlanningEvent], None]
ExecEventsCallback = Callable[[ExecutionEvent], None]


@dataclass
class NodePlanner:
    """
    Node-bound facade over PlannerService.

    - Knows about NodeContext identity/session/app/agent.
    - Delegates to PlannerService but fills execution context automatically.
    """

    service: PlannerService
    node_ctx: Any  # forward reference to NodeContext

    # ---------- Planning ----------
    async def plan(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = True,
        preferred_external_keys: list[str] | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        on_event: PlanEventsCallback | None = None,
    ) -> PlanResult:
        return await self.service.plan(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            on_event=on_event,
        )

    async def plan_with_context(
        self,
        ctx: PlanningContext,
        *,
        on_event: PlanEventsCallback | None = None,
    ) -> PlanResult:
        return await self.service.plan_with_context(ctx, on_event=on_event)

    async def parse_inputs(
        self,
        *,
        message: str,
        missing_keys: list[str],
        instruction: str | None = None,
    ):
        return await self.service.parse_inputs(
            message=message,
            missing_keys=missing_keys,
            instruction=instruction,
        )

    # ---------- Execution (auto-fill context) ----------

    async def execute(
        self,
        plan: CandidatePlan,
        *,
        user_inputs: dict[str, Any] | None = None,
        on_event: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> ExecutionResult:
        """
        Execute plan; by default, uses IDs from the bound NodeContext.
        Callers can override any of these via keyword args if needed.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        # auto-add some helpful tags
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        if eff_origin is None:
            # tiny heuristic; tune later
            eff_origin = RunOrigin.agent if ctx.agent_id else RunOrigin.user

        return await self.service.execute_plan(
            plan,
            user_inputs=user_inputs,
            on_event=on_event,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
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
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        planning_events_cb: PlanEventsCallback | None = None,
        execution_events_cb: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> tuple[PlanResult, ExecutionResult | None]:
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        # if eff_origin is None:
        #     eff_origin = RunOrigin.agent if ctx.agent_id else RunOrigin.user

        return await self.service.plan_and_execute(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            planning_events_cb=planning_events_cb,
            execution_events_cb=execution_events_cb,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
        )

    # ---------- Background execution (auto-fill context) ----------

    async def execute_background(
        self,
        plan: CandidatePlan,
        *,
        user_inputs: dict[str, Any] | None = None,
        on_event: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
        on_complete: Callable[[ExecutionResult], Any] | None = None,
        exec_id: str | None = None,
    ) -> BackgroundExecutionHandle:
        """
        Fire-and-forget style execution, bound to this node.

        Returns a BackgroundExecutionHandle; execution continues in the
        background and progress is reported via on_event / on_complete.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        if eff_origin is None:
            eff_origin = RunOrigin.agent if ctx.agent_id else RunOrigin.user

        handle = await self.service.execute_background(
            plan,
            user_inputs=user_inputs,
            on_event=on_event,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
            on_complete=on_complete,
            exec_id=exec_id,
        )
        return handle

    async def plan_and_execute_background(
        self,
        *,
        goal: str,
        user_inputs: dict[str, Any] | None = None,
        external_slots: dict[str, Any] | None = None,
        flow_ids: list[str] | None = None,
        instruction: str | None = None,
        allow_partial: bool = False,
        preferred_external_keys: list[str] | None = None,
        memory_snippets: list[str] | None = None,
        artifact_snippets: list[str] | None = None,
        planning_events_cb: PlanEventsCallback | None = None,
        execution_events_cb: ExecEventsCallback | None = None,
        identity: RequestIdentity | None = None,
        visibility: RunVisibility = RunVisibility.normal,
        importance: RunImportance = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
        on_complete: Callable[[ExecutionResult], Any] | None = None,
        exec_id: str | None = None,
    ) -> tuple[PlanResult, BackgroundExecutionHandle | None]:
        """
        Plan first, then execute the accepted plan in the background
        with node-bound context.
        """
        ctx = self.node_ctx

        eff_identity = identity or ctx.identity
        eff_session_id = session_id or ctx.session_id
        eff_agent_id = agent_id or ctx.agent_id
        eff_app_id = app_id or ctx.app_id

        eff_tags = list(tags or [])
        eff_tags.append(f"graph:{ctx.graph_id}")
        eff_tags.append(f"node:{ctx.node_id}")
        if ctx.run_id:
            eff_tags.append(f"run:{ctx.run_id}")

        eff_origin = origin
        # keep same behavior as plan_and_execute (no default if None)

        plan_result, handle = await self.service.plan_and_execute_background(
            goal=goal,
            user_inputs=user_inputs,
            external_slots=external_slots,
            flow_ids=flow_ids,
            instruction=instruction,
            allow_partial=allow_partial,
            preferred_external_keys=preferred_external_keys,
            memory_snippets=memory_snippets,
            artifact_snippets=artifact_snippets,
            planning_events_cb=planning_events_cb,
            execution_events_cb=execution_events_cb,
            identity=eff_identity,
            visibility=visibility,
            importance=importance,
            session_id=eff_session_id,
            agent_id=eff_agent_id,
            app_id=eff_app_id,
            tags=eff_tags,
            origin=eff_origin,
            on_complete=on_complete,
            exec_id=exec_id,
        )

        return plan_result, handle

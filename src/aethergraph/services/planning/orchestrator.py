from __future__ import annotations

from dataclasses import dataclass

from aethergraph.contracts.services.planning import (
    IntentRouter,
    SessionState,
)
from aethergraph.services.planning.plan_executor import (
    ExecutionResult,
    PlanExecutor,
)
from aethergraph.services.planning.plan_types import CandidatePlan
from aethergraph.services.planning.planner import ActionPlanner

from .planning_context_builder import PlanningContextBuilderProtocol
from .quick_actions import QuickActionRegistry


@dataclass
class AgentTurnResult:
    mode: str
    message_to_user: str | None = None
    plan: CandidatePlan | None = None
    execution: ExecutionResult | None = None


@dataclass
class AgentOrchestrator:
    router: IntentRouter
    context_builder: PlanningContextBuilderProtocol
    planner: ActionPlanner
    executor: PlanExecutor
    quick_actions: QuickActionRegistry

    async def handle_turn(
        self,
        *,
        user_message: str,
        session_state: SessionState,
    ) -> AgentTurnResult:
        # 1) Route intent
        routed = await self.router.route(
            user_message=user_message,
            session_state=session_state,
        )

        # 2) Mode dispatch
        if routed.mode == "chat_only":
            return AgentTurnResult(
                mode="chat_only",
                message_to_user="(chat-only mode: not yet implemented)",
            )

        if routed.mode == "quick_action":
            handler = self.quick_actions.get_handler(routed.quick_action_id or "")
            if handler is None:
                return AgentTurnResult(
                    mode="quick_action",
                    message_to_user=f"Unknown quick action: {routed.quick_action_id}",
                )
            result = await handler(context={"user_message": user_message})
            return AgentTurnResult(
                mode="quick_action",
                message_to_user=f"Quick action {routed.quick_action_id} done: {result!r}",
            )

        if routed.mode == "plan_and_execute":
            # 3) Build planning context
            planning_context = await self.context_builder.build(
                user_message=user_message,
                routed=routed,
                session_state=session_state,
            )

            # 4) Plan
            plan, history = await self.planner.plan_with_loop(planning_context)
            if plan is None:
                return AgentTurnResult(
                    mode="plan_and_execute",
                    message_to_user=(
                        "I tried to build a plan but couldn't find a valid workflow. "
                        "You may need to provide more details."
                    ),
                )

            # 5) Execute
            execution_result = await self.executor.execute(
                plan,
                user_inputs=planning_context.user_inputs,
            )

            if not execution_result.ok:
                return AgentTurnResult(
                    mode="plan_and_execute",
                    message_to_user="The plan failed during execution.",
                    plan=plan,
                    execution=execution_result,
                )

            return AgentTurnResult(
                mode="plan_and_execute",
                message_to_user=f"Plan executed successfully. Outputs: {execution_result.outputs!r}",
                plan=plan,
                execution=execution_result,
            )

        # future unkonwn modes
        return AgentTurnResult(
            mode=routed.mode,
            message_to_user=f"Unsupported mode: {routed.mode}",
        )

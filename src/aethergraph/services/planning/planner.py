# aethergraph/services/planning/planner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .action_catalog import ActionCatalog
from .flow_validator import FlowValidator
from .plan_types import CandidatePlan
from .validation_types import ValidationResult


# NOTE: we need to call llm.chat directly
class LLMClientProtocol(Protocol):
    async def complete_json(
        self, *, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...


@dataclass
class PlanningContext:
    goal: str
    user_inputs: dict[str, Any]
    external_slots: dict[str, Any]  # for prompt only, validator sees IOSlot map
    memory_snippets: list[str] = None
    artifact_snippets: list[str] = None
    flow_id: str | None = None


class ActionPlanner:
    catalog: ActionCatalog
    validator: FlowValidator
    llm: LLMClientProtocol

    async def plan_with_loop(
        self,
        ctx: PlanningContext,
        *,
        max_iter: int = 3,
    ) -> tuple[CandidatePlan | None, list[ValidationResult]]:
        history: list[ValidationResult] = []

        # build static action description once
        actions_md = self.catalog.pretty_print(flow_id=ctx.flow_id)

        system_prompt = (
            "You are a planning assistant that builds executable workflows as JSON plans."
        )

        plan: CandidatePlan | None = None

        for _ in range(max_iter):
            if plan is None:
                user_prompt = self._build_initial_prompt(ctx, actions_md)
            else:
                last_v = history[-1]
                user_prompt = self._build_revision_prompt(ctx, actions_md, last_v)

            raw = await self.llm.complete_json(
                system=system_prompt,
                user=user_prompt,
                schema={"type": "object", "properties": {"steps": {"type": "array"}}},
            )
            plan = CandidatePlan.from_dict(raw)

            v = self.validator.validate(plan, external_inputs={})
            history.append(v)
            if v.ok:
                break

        return plan if history and history[-1].ok else None, history

    def _build_initial_prompt(self, ctx: PlanningContext, actions_md: str) -> str:
        # keep it simple; refine later
        return (
            f"Goal:\n{ctx.goal}\n\n"
            f"Available actions:\n{actions_md}\n\n"
            "Produce a JSON object of the form:\n"
            '{ "steps": [ { "id": "...", "action_ref": "...", "inputs": { ... } } ] }'
        )

    def _build_repair_prompt(
        self,
        ctx: PlanningContext,
        actions_md: str,
        plan: CandidatePlan,
        validation: ValidationResult,
    ) -> str:
        return (
            f"Goal:\n{ctx.goal}\n\n"
            f"Available actions:\n{actions_md}\n\n"
            "Current plan JSON:\n"
            f"{plan.to_dict()}\n\n"
            "Validation result:\n"
            f"{validation.summary()}\n\n"
            "Please return an improved plan JSON of the same shape."
        )

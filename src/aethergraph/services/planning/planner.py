# aethergraph/services/planning/planner.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from aethergraph.contracts.services.llm import LLMClientProtocol

from .action_catalog import ActionCatalog
from .flow_validator import FlowValidator
from .plan_types import CandidatePlan
from .validation_types import ValidationResult

PlanningPhase = Literal[
    "start",
    "llm_request",
    "llm_response",
    "validation",
    "success",
    "failure",
]


@dataclass
class PlanningEvent:
    """
    Lightweight event emitted during planning.

    This is designed for:
      - logging / debugging (print to console)
      - UI progress (update spinner / status text)
    """

    phase: PlanningPhase
    iteration: int
    message: str | None = None

    # Optional structured payloads:
    raw_llm_output: dict[str, Any] | None = None
    plan_dict: dict[str, Any] | None = None
    validation: ValidationResult | None = None


@dataclass
class PlanningContext:
    goal: str
    user_inputs: dict[str, Any]
    external_slots: dict[str, Any]  # for prompt only, validator sees IOSlot map
    memory_snippets: list[str] = None
    artifact_snippets: list[str] = None
    flow_id: str | None = None


@dataclass
class ActionPlanner:
    catalog: ActionCatalog
    validator: FlowValidator
    llm: LLMClientProtocol

    @staticmethod
    def _emit(
        on_event: Callable[[PlanningEvent], None] | None,
        event: PlanningEvent,
    ) -> None:
        """
        Small helper to safely emit events if a callback is provided.
        """
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                # We don't want a logging/UI bug to crash planning.
                # Swallow errors here; in the future we may want to at least log them.
                import logging

                logger = logging.getLogger(__name__)
                logger.warning("Error in planning on_event callback", exc_info=True)

    async def plan_with_loop(
        self,
        ctx: PlanningContext,
        *,
        max_iter: int = 3,
        on_event: Callable[[PlanningEvent], None] | None = None,
    ) -> tuple[CandidatePlan | None, list[ValidationResult]]:
        """
        Try up to `max_iter` times to obtain a valid plan.

        If `on_event` is provided, emit PlanningEvent instances to allow
        logging / UI progress.
        """
        history: list[ValidationResult] = []
        plan: CandidatePlan | None = None

        actions_md = self.catalog.pretty_print(flow_id=ctx.flow_id)

        system_prompt = (
            "You are a planning assistant that builds executable workflows as JSON plans. "
            "You must strictly follow the JSON schema and return ONLY JSON, no extra text."
        )

        # initial "start" event
        self._emit(
            on_event,
            PlanningEvent(
                phase="start",
                iteration=0,
                message=f"Starting planning for goal: {ctx.goal!r}",
            ),
        )

        for iter_idx in range(max_iter):
            if plan is None:
                user_prompt = self._build_initial_prompt(ctx, actions_md)
            else:
                last_v = history[-1]
                user_prompt = self._build_repair_prompt(ctx, actions_md, plan, last_v)

            # LLM request event
            self._emit(
                on_event,
                PlanningEvent(
                    phase="llm_request",
                    iteration=iter_idx,
                    message="Sending planning request to LLM.",
                ),
            )

            # 1) Ask LLM for a JSON plan
            raw_json = await self._call_llm_for_plan(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            self._emit(
                on_event,
                PlanningEvent(
                    phase="llm_response",
                    iteration=iter_idx,
                    message="Received plan JSON from LLM.",
                    raw_llm_output=raw_json,
                    plan_dict=raw_json,
                ),
            )

            plan = CandidatePlan.from_dict(raw_json)

            # 2) Validate
            v = self.validator.validate(plan, external_inputs={}, flow_id=ctx.flow_id)
            history.append(v)

            self._emit(
                on_event,
                PlanningEvent(
                    phase="validation",
                    iteration=iter_idx,
                    message=f"Validation result: {'OK' if v.ok else 'INVALID'}",
                    validation=v,
                    plan_dict=plan.to_dict(),
                ),
            )

            if v.ok:
                self._emit(
                    on_event,
                    PlanningEvent(
                        phase="success",
                        iteration=iter_idx,
                        message="Planning succeeded with a valid plan.",
                        validation=v,
                        plan_dict=plan.to_dict(),
                    ),
                )
                break

        if not history or not history[-1].ok:
            self._emit(
                on_event,
                PlanningEvent(
                    phase="failure",
                    iteration=len(history) - 1 if history else -1,
                    message="Planning failed to produce a valid plan.",
                    validation=history[-1] if history else None,
                    plan_dict=plan.to_dict() if plan else None,
                ),
            )

        return plan if history and history[-1].ok else None, history

    def _plan_schema(self) -> dict[str, Any]:
        """
        JSON schema for the plan that the LLM must output.

        We keep it intentionally simple and forgiving; FlowValidator will enforce
        correctness. We can tighten this over time.
        """
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "action_ref": {"type": "string"},
                            "inputs": {
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["id", "action_ref", "inputs"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }

    async def _call_llm_for_plan(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """
        Call the LLM with our plan schema and return the parsed JSON object.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        schema = self._plan_schema()

        raw, _usage = await self.llm.chat(
            messages,
            output_format="json",
            json_schema=schema,
            schema_name="Plan",
            strict_schema=True,
            validate_json=True,
        )

        # GenericLLMClient.chat may already return a dict for output_format="json",
        # but we defensively handle the string case.

        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            import json

            return json.loads(raw)

        # Extremely defensive fallback
        raise TypeError(
            f"LLM returned unsupported structured output type: {type(raw)}; "
            "expected dict or JSON string."
        )

    def _build_initial_prompt(self, ctx: PlanningContext, actions_md: str) -> str:
        """
        Initial prompt: describe goal, context, and actions, ask for a first plan.
        """
        user_inputs_str = repr(ctx.user_inputs or {})
        external_str = repr(ctx.external_slots or {})

        memory_str = ""
        if ctx.memory_snippets:
            memory_str = (
                "Relevant memory:\n" + "\n".join(f"- {m}" for m in ctx.memory_snippets) + "\n\n"
            )

        artifact_str = ""
        if ctx.artifact_snippets:
            artifact_str = (
                "Relevant artifacts:\n"
                + "\n".join(f"- {a}" for a in ctx.artifact_snippets)
                + "\n\n"
            )

        return (
            f"Goal:\n{ctx.goal}\n\n"
            f"User inputs (available values):\n{user_inputs_str}\n\n"
            f"External bindings (available as `{{user.<key>}}`):\n{external_str}\n\n"
            f"{memory_str}"
            f"{artifact_str}"
            "You have the following actions available:\n"
            f"{actions_md}\n\n"
            "You must create a workflow as a JSON object of the form:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "id": "load",\n'
            '      "action_ref": "<one of the action refs above>",\n'
            '      "inputs": {\n'
            '        "arg_name": <literal or binding>\n'
            "      }\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Bindings can be:\n"
            '- literals, e.g. 0.8 or {"lr": 0.01}\n'
            '- external values, using the syntax "${user.<key>}" where <key> is one of the external bindings\n'
            '- outputs from previous steps, using the syntax "${<step_id>.<output_name>}".\n\n'
            "Make sure that:\n"
            "- step ids are unique,\n"
            "- action_ref exactly matches one of the listed action refs,\n"
            "- you only reference outputs from earlier steps.\n\n"
            "Return ONLY the JSON object, with no explanation or comments."
        )

    def _build_repair_prompt(
        self,
        ctx: PlanningContext,
        actions_md: str,
        plan: CandidatePlan,
        validation: ValidationResult,
    ) -> str:
        """
        Repair prompt: show the current plan + validation summary and ask the LLM
        to return an improved JSON plan.
        """
        user_inputs_str = repr(ctx.user_inputs or {})
        external_str = repr(ctx.external_slots or {})

        return (
            f"Goal:\n{ctx.goal}\n\n"
            f"User inputs (available values):\n{user_inputs_str}\n\n"
            f"External bindings (available as `{{user.<key>}}`):\n{external_str}\n\n"
            "You have the following actions available:\n"
            f"{actions_md}\n\n"
            "Here is the current candidate plan JSON:\n"
            f"{plan.to_dict()}\n\n"
            "Validation result for this plan:\n"
            f"{validation.summary()}\n\n"
            "Some issues may also include candidate actions that can provide missing inputs.\n\n"
            "Please return a corrected plan as a JSON object of the SAME SHAPE:\n"
            '{ "steps": [ { "id": "...", "action_ref": "...", "inputs": { ... } } ] }\n\n'
            "You may:\n"
            "- add, remove, or reorder steps,\n"
            "- change action_ref values to valid actions,\n"
            "- fix input bindings and literals.\n\n"
            "Return ONLY the JSON object, with no explanation or comments."
        )

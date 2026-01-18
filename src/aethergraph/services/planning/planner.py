# aethergraph/services/planning/planner.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.planning import PlanningContext, PlanningEvent, ValidationResult

from .action_catalog import ActionCatalog
from .flow_validator import FlowValidator
from .plan_types import CandidatePlan


class PlanDecodingError(Exception):
    """
    Raised when the LLM's response cannot be parsed into a valid JSON plan.
    Carries the raw text so callers can log or surface it.
    """

    def __init__(self, message: str, raw_text: str | None = None):
        super().__init__(message)
        self.raw_text = raw_text


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
        history: list[ValidationResult] = []
        plan: CandidatePlan | None = None

        flow_ids = ctx.flow_ids  # could be None
        actions_md = self.catalog.pretty_print(
            flow_ids=flow_ids,
            include_global=True,
        )

        system_prompt = (
            "You are a planning assistant that builds executable workflows as JSON plans. "
            "You must strictly follow the JSON schema and return ONLY JSON, no extra text."
        )

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

            self._emit(
                on_event,
                PlanningEvent(
                    phase="llm_request",
                    iteration=iter_idx,
                    message="Sending planning request to LLM.",
                ),
            )

            try:
                raw_json = await self._call_llm_for_plan(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
            except PlanDecodingError as exc:
                # Treat this as a failed attempt; emit an event and move to next iteration.
                self._emit(
                    on_event,
                    PlanningEvent(
                        phase="llm_response",
                        iteration=iter_idx,
                        message=f"LLM response could not be parsed as JSON: {exc}",
                        raw_llm_output=getattr(exc, "raw_text", None),
                        plan_dict=None,
                    ),
                )
                # Do NOT append a ValidationResult here; this is a transport/format error,
                # not a semantic plan error. Just try again if we have remaining iters.
                continue

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
            external_inputs = ctx.external_slots or (ctx.user_inputs or {})
            print("=== Candidate Plan ===")
            print(plan)
            print("external inputs:", external_inputs)

            v = self.validator.validate(
                plan,
                external_inputs=external_inputs,
                flow_ids=flow_ids,
            )
            history.append(v)

            print(v.summary())
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
        JSON schema for the plan. We include a plan_version and extras for future evolution.
        """
        return {
            "type": "object",
            "properties": {
                "plan_version": {
                    "type": "string",
                    "default": "1",
                },
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
                            "extras": {  # extension point
                                "type": "object",
                                "additionalProperties": True,
                            },
                        },
                        "required": ["id", "action_ref", "inputs"],
                        "additionalProperties": False,
                    },
                },
                "extras": {  # extension point at plan level
                    "type": "object",
                    "additionalProperties": True,
                },
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

        This is robust to models that ignore output_format="json" and wrap
        the JSON in ``` fences or surrounding text.
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

        # 1) Already a dict: perfect.
        if isinstance(raw, dict):
            return raw

        # 2) String: try to recover JSON from it.
        if isinstance(raw, str):
            import json

            txt = raw.strip()
            if not txt:
                raise PlanDecodingError("Empty LLM response when expecting JSON.", raw_text=raw)

            # Handle ```json ... ``` or ``` ... ``` fences
            if txt.startswith("```"):
                # strip leading ``` or ```json / ```JSON
                if txt.lower().startswith("```json"):
                    txt = txt[len("```json") :].strip()
                else:
                    txt = txt[3:].strip()
                # strip trailing ```
                if txt.endswith("```"):
                    txt = txt[:-3].strip()

            # First attempt: parse whole string
            try:
                return json.loads(txt)
            except json.JSONDecodeError:
                # Second attempt: extract the first {...} block
                start = txt.find("{")
                end = txt.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = txt[start : end + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as exc2:
                        raise PlanDecodingError(
                            f"Cannot parse JSON from LLM response (substring). " f"Error: {exc2}",
                            raw_text=raw,
                        ) from exc2

                # No obvious JSON object found
                raise PlanDecodingError(
                    "Cannot parse JSON from LLM response (no JSON object found).",
                    raw_text=raw,
                ) from None  # from None to suppress context

        # 3) Unsupported type
        raise PlanDecodingError(
            f"LLM returned unsupported structured output type: {type(raw)}; "
            "expected dict or JSON string.",
            raw_text=str(raw),
        )

    def _build_initial_prompt(self, ctx: PlanningContext, actions_md: str) -> str:
        """
        Initial prompt: describe goal, context, and actions, ask for a first plan.
        """
        user_inputs_str = repr(ctx.user_inputs or {})
        external_dict = ctx.external_slots or (ctx.user_inputs or {})
        external_str = repr(external_dict)

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
        external_dict = ctx.external_slots or (ctx.user_inputs or {})
        external_str = repr(external_dict)

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

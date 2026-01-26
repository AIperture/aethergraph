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

        print("🍎 === Planning Context ===")
        print(ctx)
        print("========================")
        flow_ids = ctx.flow_ids  # could be None
        actions_md = self.catalog.pretty_print(
            flow_ids=flow_ids,
            include_global=True,
        )
        print("🍎 === Available Actions ===")
        print(actions_md)
        print("===========================")

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

            print("🍎 === Planning Prompt ===")
            print(user_prompt[:1000] + ("..." if len(user_prompt) > 1000 else ""))
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

            print("🍎 Raw plan JSON from LLM:", raw_json)
            plan = CandidatePlan.from_dict(raw_json)
            plan.resolve_actions(
                self.catalog,
                flow_ids=flow_ids,
                include_global=True,
            )
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
                # Fully valid plan
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

            # Accept partial plan if allowed and structurally OK
            if getattr(ctx, "allow_partial_plans", False) and v.is_partial_ok():
                self._emit(
                    on_event,
                    PlanningEvent(
                        phase="success",
                        iteration=iter_idx,
                        message=(
                            "Planning produced a structurally valid plan "
                            "with missing user bindings."
                        ),
                        validation=v,
                        plan_dict=plan.to_dict(),
                    ),
                )
                break

        last = history[-1] if history else None
        accept_partial = (
            last is not None and getattr(ctx, "allow_partial_plans", False) and last.is_partial_ok()
        )

        if not last or (not last.ok and not accept_partial):
            # Total failure: no fully valid or partial-acceptable plan
            self._emit(
                on_event,
                PlanningEvent(
                    phase="failure",
                    iteration=len(history) - 1 if history else -1,
                    message="Planning failed to produce a valid or partial-acceptable plan.",
                    validation=last,
                    plan_dict=plan.to_dict() if plan else None,
                ),
            )

        # Return the last plan if valid or partial-acceptable, else None
        return plan if last and (last.ok or accept_partial) else None, history

    def _plan_schema(self) -> dict[str, Any]:
        """
        JSON schema for the plan. We include a plan_version and extras for future evolution.
        """
        return {
            "type": "object",
            "properties": {
                "plan_version": {
                    "type": "string",
                    "default": "2",  # bump to v2
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "action": {"type": "string"},
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
                        # require action name; action_ref is optional
                        "required": ["id", "action", "inputs"],
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

    def _build_binding_hints(self, ctx: PlanningContext) -> str:
        """
        Build a small hint section about which fields should be treated as
        user-provided bindings (e.g. dataset_path, grid_spec, hyperparams).

        We prefer to get this from ctx.skill.meta["inputs"], falling back to
        the keys in ctx.user_inputs if no skill metadata is available.
        """
        # 1) Try skill metadata
        preferred_keys: list[str] = []
        meta = getattr(ctx.skill, "meta", None)
        if isinstance(meta, dict):
            inputs_meta = meta.get("inputs") or []
            if isinstance(inputs_meta, list):
                for entry in inputs_meta:
                    if isinstance(entry, dict):
                        name = entry.get("name")
                        if name:
                            preferred_keys.append(name)

        # 2) Fallback: use user_inputs keys if no skill config
        if not preferred_keys:
            preferred_keys = list((ctx.user_inputs or {}).keys())

        preferred_keys = sorted(set(preferred_keys))
        if not preferred_keys:
            return ""

        lines: list[str] = []
        lines.append(
            "Important: When you need any of the following values in the plan, "
            'you should bind them as external "${user.<key>}" references '
            "instead of inventing new literal values (unless the exact value is "
            "already present in User inputs):"
        )
        for key in preferred_keys:
            lines.append(f"- {key}")
        lines.append("")
        lines.append(
            "For example, if you need the dataset path or grid specification, "
            'use "${user.dataset_path}" or "${user.grid_spec}" instead of '
            "writing fake file paths or hand-crafted numeric grids."
        )
        lines.append("")

        return "\n".join(lines)

    def _build_initial_prompt(self, ctx: PlanningContext, actions_md: str) -> str:
        """
        Initial prompt: describe goal, context, and actions, ask for a first plan.
        """
        user_inputs = ctx.user_inputs or {}
        external_slots = ctx.external_slots or {}

        # 1) Decide which keys we *want* to advertise as potential ${user.*}
        preferred_keys = list(ctx.preferred_external_keys or [])
        # also include any keys that already have values
        preferred_keys.extend(user_inputs.keys())
        preferred_keys = sorted(set(preferred_keys))

        # 2) Build a pretty view: show value if we have it, or mark as missing
        external_view: dict[str, Any] = {}
        for key in preferred_keys:
            if key in external_slots:
                # could show a type or descriptor here
                external_view[key] = getattr(external_slots[key], "type", "<slot>")
            elif key in user_inputs:
                external_view[key] = user_inputs[key]
            else:
                external_view[key] = "<NOT PROVIDED YET>"

        user_inputs_str = repr(user_inputs)
        external_str = repr(external_view)

        # --- 1) Skill-aware planning header ---
        if ctx.skill and ctx.skill.planning_prompt:
            header = ctx.skill.planning_prompt.strip()
        else:
            header = (
                "You are a planning assistant that builds executable workflows as JSON plans. "
                "You must strictly follow the JSON schema and return ONLY JSON, no extra text."
            )

        # Optional: small skill name hint
        if ctx.skill:
            header += f"\n\nCurrent domain skill: {ctx.skill.title} — {ctx.skill.description}"

        # --- 2) Memory & artifact snippets (already LLM-friendly strings) ---
        memory_str = ""
        if ctx.memory_snippets:
            memory_str = (
                "Relevant recent context (runs, summaries, etc.):\n"
                + "\n".join(f"- {m}" for m in ctx.memory_snippets)
                + "\n\n"
            )

        artifact_str = ""
        if ctx.artifact_snippets:
            artifact_str = (
                "Relevant artifacts:\n"
                + "\n".join(f"- {a}" for a in ctx.artifact_snippets)
                + "\n\n"
            )

        binding_hints = self._build_binding_hints(ctx)

        # --- 3) Main planning instructions (unchanged core, but after header/context) ---
        return (
            f"{header}\n\n"
            f"Goal:\n{ctx.goal}\n\n"
            f"User inputs (available values):\n{user_inputs_str}\n\n"
            f"External bindings (available as `{{user.<key>}}`):\n{external_str}\n\n"
            f"{memory_str}"
            f"{artifact_str}"
            f"{binding_hints}"
            "You have the following actions available:\n"
            f"{actions_md}\n\n"
            "You must create a workflow as a JSON object of the form:\n"
            "{\n"
            '  "steps": [\n'
            "    {\n"
            '      "id": "load",\n'
            '      "action": "<one of the action names above>",\n'
            '      "inputs": {\n'
            '        "arg_name": <literal or binding>\n'
            "      }\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Bindings can be:\n"
            '- external values, using the syntax "${user.<key>}" where <key> is one of the external bindings. '
            "For configuration-like fields (such as dataset paths, grid specs, hyperparameters, or ratios), "
            "you MUST prefer external bindings over inventing new literal values.\n"
            '- literals, e.g. 0.8 or {"lr": 0.01}, but only when the value is clearly fixed by the goal or already '
            "present in User inputs.\n"
            '- outputs from previous steps, using the syntax "${<step_id>.<output_name>}".\n\n'
            "Make sure that:\n"
            "- step ids are unique,\n"
            "- action is exactly one of the listed action names,\n"
            "- you only reference outputs from earlier steps.\n\n"
            "- Do NOT invent file paths or other external values that are not already provided.\n"
            '- If you need something like a dataset path and it is not known yet, bind it as "${user.dataset_path}".\n'
            '- Never hard-code fake paths like "path/to/dataset".\n\n'
            "- If the user refers to “previous run”, “last plan”, or “same as before”,  reuse the same logical sequence of actions as past plans for this flow, unless they explicitly request structural changes.\n\n"
            "- If they say “stop after <step>” or “skip <step>”, omit that action from the workflow.\n\n"
            '            "Return ONLY the JSON object, with no explanation or comments."\n'
        )

    def _build_initial_prompt_v0(self, ctx: PlanningContext, actions_md: str) -> str:
        """
        Initial prompt: describe goal, context, and actions, ask for a first plan.
        """
        user_inputs = ctx.user_inputs or {}
        external_slots = ctx.external_slots or {}

        # 1) Decide which keys we *want* to advertise as potential ${user.*}
        preferred_keys = list(ctx.preferred_external_keys or [])
        # also include any keys that already have values
        preferred_keys.extend(user_inputs.keys())
        preferred_keys = sorted(set(preferred_keys))

        # 2) Build a pretty view: show value if we have it, or mark as missing
        external_view: dict[str, Any] = {}
        for key in preferred_keys:
            if key in external_slots:
                # could show a type or descriptor here
                external_view[key] = getattr(external_slots[key], "type", "<slot>")
            elif key in user_inputs:
                external_view[key] = user_inputs[key]
            else:
                external_view[key] = "<NOT PROVIDED YET>"

        user_inputs_str = repr(user_inputs)
        external_str = repr(external_view)

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
            '      "action": "<one of the action names above>",\n'
            '      "inputs": {\n'
            '        "arg_name": <literal or binding>\n'
            "      }\n"
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            "Bindings can be:\n"
            '- external values, using the syntax "${user.<key>}" where <key> is one of the external bindings. '
            "For configuration-like fields (such as dataset paths, grid specs, hyperparameters, or ratios), "
            "you MUST prefer external bindings over inventing new literal values.\n"
            '- literals, e.g. 0.8 or {"lr": 0.01}, but only when the value is clearly fixed by the goal or already '
            "present in User inputs.\n"
            '- outputs from previous steps, using the syntax "${<step_id>.<output_name>}".\n\n'
            "Make sure that:\n"
            "- step ids are unique,\n"
            "- action_ref exactly matches one of the listed action refs,\n"
            "- you only reference outputs from earlier steps.\n\n"
            "- Do NOT invent file paths or other external values that are not already provided.\n"
            '- If you need something like a dataset path and it is not known yet, bind it as "${user.dataset_path}".\n'
            '- Never hard-code fake paths like "path/to/dataset".\n\n'
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

        if ctx.skill and ctx.skill.planning_prompt:
            header = ctx.skill.planning_prompt.strip()
        else:
            header = (
                "You are a planning assistant that repairs and improves existing "
                "JSON workflow plans. Return ONLY valid JSON."
            )

        # Optional: include some compact context again if we want
        memory_str = ""
        if ctx.memory_snippets:
            memory_str = (
                "Relevant recent context (runs, summaries, etc.):\n"
                + "\n".join(f"- {m}" for m in ctx.memory_snippets)
                + "\n\n"
            )

        return (
            f"{header}\n\n"
            f"Goal:\n{ctx.goal}\n\n"
            f"User inputs (available values):\n{user_inputs_str}\n\n"
            f"External bindings (available as `{{user.<key>}}`):\n{external_str}\n\n"
            f"{memory_str}"
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

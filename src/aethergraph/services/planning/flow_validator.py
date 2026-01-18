# aethergraph/services/planning/flow_validator.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from aethergraph.contracts.services.planning import ValidationIssue, ValidationResult
from aethergraph.core.graph.action_spec import ActionSpec, IOSlot

from .action_catalog import ActionCatalog
from .bindings import parse_binding
from .dependency_index import DependencyIndex
from .plan_types import CandidatePlan, PlanStep


@dataclass
class FlowValidator:
    catalog: ActionCatalog
    dep_index: DependencyIndex | None = None

    def _action_index(
        self,
        *,
        flow_ids: list[str] | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> dict[str, ActionSpec]:
        idx: dict[str, ActionSpec] = {}
        for spec in self.catalog.iter_actions(
            flow_ids=flow_ids,
            kinds=kinds,
            include_global=True,
        ):
            idx[spec.ref] = spec
        return idx

    @staticmethod
    def _is_strict_type_mismatch(
        expected: str | None,
        actual: str | None,
    ) -> bool:
        if not expected or not actual:
            return False
        wildcard = {"any", "object"}
        if expected in wildcard or actual in wildcard:
            return False
        return expected != actual

    def validate(
        self,
        plan: CandidatePlan,
        *,
        external_inputs: dict[str, IOSlot] | None = None,
        flow_ids: list[str] | None = None,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        external_inputs = external_inputs or {}

        action_index = self._action_index(flow_ids=flow_ids)
        step_index: dict[str, PlanStep] = {step.id: step for step in plan.steps}

        # 1) unknown actions
        for step in plan.steps:
            if step.action_ref not in action_index:
                issues.append(
                    ValidationIssue(
                        kind="unknown_action",
                        step_id=step.id,
                        field=None,
                        message=f"Action '{step.action_ref}' is not recognized.",
                    )
                )

        if any(iss.kind == "unknown_action" for iss in issues):
            return ValidationResult(ok=False, issues=issues)

        # 2) dependency graph
        edges: dict[str, set[str]] = {step.id: set() for step in plan.steps}
        for step in plan.steps:
            for raw in (step.inputs or {}).values():
                binding = parse_binding(raw)
                if binding.kind == "step_output" and binding.source_step_id in step_index:
                    edges[step.id].add(binding.source_step_id)

        # 3) detect cycles
        visiting: set[str] = set()
        visited: set[str] = set()
        has_cycle = False

        def dfs(node: str) -> None:
            nonlocal has_cycle
            if node in visited or has_cycle:
                return
            if node in visiting:
                has_cycle = True
                return
            visiting.add(node)
            for dep in edges[node]:
                dfs(dep)
            visiting.remove(node)
            visited.add(node)

        for step_id in edges:
            if step_id not in visited:
                dfs(step_id)

        if has_cycle:
            issues.append(
                ValidationIssue(
                    kind="cycle",
                    step_id="",
                    field=None,
                    message="The plan contains cyclic dependencies among steps.",
                )
            )
            return ValidationResult(ok=False, issues=issues)

        # 4) topo order
        in_deg: dict[str, int] = {step_id: 0 for step_id in edges}
        for step_id, deps in edges.items():
            for _ in deps:
                in_deg[step_id] += 1

        ready = [sid for sid, deg in in_deg.items() if deg == 0]
        topo_order: list[str] = []
        while ready:
            sid = ready.pop()
            topo_order.append(sid)
            for consumer, deps in edges.items():
                if sid in deps:
                    in_deg[consumer] -= 1
                    if in_deg[consumer] == 0:
                        ready.append(consumer)

        # 5) validate inputs along topo order
        available_outputs: dict[str, IOSlot] = {}

        for step_id in topo_order:
            step = step_index[step_id]
            spec = action_index[step.action_ref]
            input_by_name = {slot.name: slot for slot in spec.inputs}

            for name, slot in input_by_name.items():
                raw_value = step.inputs.get(name)

                if raw_value is None and slot.required:
                    details: dict = {}
                    if self.dep_index is not None:
                        cands = self.dep_index.find_producers(
                            slot,
                            flow_ids=flow_ids,
                        )
                        details["candidates"] = [
                            {
                                "action_ref": a.ref,
                                "output": out.name,
                                "description": a.description,
                            }
                            for (a, out) in cands
                        ]
                    issues.append(
                        ValidationIssue(
                            kind="missing_input",
                            step_id=step_id,
                            field=name,
                            message=(
                                f"Required input '{name}' is not provided "
                                f"for action '{spec.name}'."
                            ),
                            details=details,
                        )
                    )
                    continue

                if raw_value is None:
                    continue

                binding = parse_binding(raw_value)

                if binding.kind == "external":
                    key = binding.external_key or ""
                    ext_slot = external_inputs.get(key)
                    if ext_slot is None:
                        issues.append(
                            ValidationIssue(
                                kind="missing_input",
                                step_id=step_id,
                                field=name,
                                message=(
                                    f"External input '{key}' is not declared in external_inputs."
                                ),
                            )
                        )
                    else:
                        # ext_slot can be an IOSlot OR a bare value (when planner passes user_inputs).
                        # In the latter case we don't have type info, so we skip strict type checking.
                        ext_type = getattr(ext_slot, "type", None)
                        if self._is_strict_type_mismatch(slot.type, ext_type):
                            issues.append(
                                ValidationIssue(
                                    kind="type_mismatch",
                                    step_id=step_id,
                                    field=name,
                                    message=(
                                        f"Type mismatch for external input '{key}': "
                                        f"expected '{slot.type}', got '{ext_slot.type}'."
                                    ),
                                )
                            )

                elif binding.kind == "step_output":
                    src_id = binding.source_step_id or ""
                    out_name = binding.source_output_name or ""
                    key = f"{src_id}.{out_name}"
                    out_slot = available_outputs.get(key)
                    if out_slot is None:
                        issues.append(
                            ValidationIssue(
                                kind="missing_input",
                                step_id=step_id,
                                field=name,
                                message=(
                                    f"Input '{name}' refers to '{key}', "
                                    "which is not produced by any previous step."
                                ),
                            )
                        )
                    else:
                        if self._is_strict_type_mismatch(slot.type, out_slot.type):
                            issues.append(
                                ValidationIssue(
                                    kind="type_mismatch",
                                    step_id=step_id,
                                    field=name,
                                    message=(
                                        f"Input '{name}' expects type '{slot.type}' "
                                        f"but is wired from '{key}' with type '{out_slot.type}'."
                                    ),
                                )
                            )

                # literals: accepted, no extra checks yet

            for out in spec.outputs:
                available_outputs[f"{step_id}.{out.name}"] = out

        return ValidationResult(ok=(len(issues) == 0), issues=issues)

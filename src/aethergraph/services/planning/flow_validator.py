from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from aethergraph.core.graph.action_spec import ActionSpec, IOSlot

from .action_catalog import ActionCatalog
from .bindings import parse_binding
from .dependency_index import DependencyIndex
from .plan_types import CandidatePlan, PlanStep
from .validation_types import ValidationIssue, ValidationResult


@dataclass
class FlowValidator:
    """
    Validates a candidate execution plan for a flow, checking for unknown actions, cyclic dependencies,
    missing or mismatched inputs, and correct wiring of step outputs.
    Args:
        plan (CandidatePlan): The candidate plan to validate, consisting of a sequence of steps.
        external_inputs (Optional[Dict[str, IOSlot]]): Mapping of external input keys to IOSlot definitions,
            used to resolve bindings to external values. Defaults to None.
        flow_id (Optional[str]): The flow identifier to restrict validation to a specific flow. Defaults to None.
    Returns:
        ValidationResult: An object indicating whether the plan is valid (`ok=True`) and a list of
            `ValidationIssue` objects describing any problems found.
    Validation Logic:
        1. Checks that all actions referenced in plan steps exist in the action catalog.
        2. Builds a dependency graph from step input bindings and detects cycles (invalid if present).
        3. Computes a topological order of steps for input validation.
        4. For each step in order:
            - Ensures all required inputs are provided.
            - Validates that external inputs are declared and type-compatible.
            - Validates that step outputs referenced as inputs exist and are type-compatible.
        5. Collects and returns all validation issues found, or marks the plan as valid if none.
    """

    catalog: ActionCatalog
    dep_index: DependencyIndex | None = None

    def _action_index(
        self,
        *,
        flow_id: str | None = None,
        kinds: Iterable[Literal["graph", "graphfn"]] | None = ("graph", "graphfn"),
    ) -> dict[str, ActionSpec]:
        idx: dict[str, ActionSpec] = {}
        for spec in self.catalog.iter_actions(flow_id=flow_id, kinds=kinds):
            idx[spec.ref] = spec
        return idx

    @staticmethod
    def _is_strict_type_mismatch(
        expected: str | None,
        actual: str | None,
    ) -> bool:
        """
        Return True only when we are confident the types are incompatible.

        - If either side is None, 'any', or 'object', treat as compatible.
        - Otherwise, require exact match.

        NOTE: currently we don't have a rich type system, so this is simple. In planning
        future we may want to enhance this with subtyping, coercions, etc from the graph_io side.
        """
        if not expected or not actual:
            return False

        # Wildcard-ish types
        wildcard = {"any", "object"}
        if expected in wildcard or actual in wildcard:
            return False

        # You can later special-case things like number vs integer here
        return expected != actual

    def validate(
        self,
        plan: CandidatePlan,
        *,
        external_inputs: dict[str, IOSlot] | None = None,
        flow_id: str | None = None,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        external_inputs = external_inputs or {}

        action_index = self._action_index(flow_id=flow_id)
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

        # early exit if nothing resolvable
        if any(iss.kind == "unknown_action" for iss in issues):
            return ValidationResult(ok=False, issues=issues)

        # 2) build dependency graph from bindings
        edges: dict[str, set[str]] = {step.id: set() for step in plan.steps}
        for step in plan.steps:
            for raw in (step.inputs or {}).values():
                binding = parse_binding(raw)
                if binding.kind == "step_output" and binding.source_step_id in step_index:
                    edges[step.id].add(binding.source_step_id)

        # 3) detect cycles via DFS
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

        # 4) topological order (simple Kahn-like, since we already checked cycles)
        in_deg: dict[str, int] = {step_id: 0 for step_id in edges}
        for step_id, deps in edges.items():
            for _ in deps:
                in_deg[step_id] += 1

        ready = [step_id for step_id, deg in in_deg.items() if deg == 0]
        topo_order: list[str] = []
        while ready:
            step_id = ready.pop()
            topo_order.append(step_id)
            for consumer, deps in edges.items():
                if step_id in deps:
                    in_deg[consumer] -= 1
                    if in_deg[consumer] == 0:
                        ready.append(consumer)

        # 5) walk in topo order, checking inputs
        # available values: mapping "step_id.output_name" -> IOSlot
        available_outputs: dict[str, IOSlot] = {}

        for step_id in topo_order:
            step = step_index[step_id]
            spec = action_index[step.action_ref]

            input_by_name = {slot.name: slot for slot in spec.inputs}

            # require inputs present?
            for name, slot in input_by_name.items():
                raw_value = step.inputs.get(name)
                if raw_value is None and slot.required:
                    details: dict = {}
                    if self.dep_index is not None:
                        cands = self.dep_index.find_producers(slot, flow_id=spec.flow_id)
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
                            message=f"Required input '{name}' is not provided for action '{spec.name}'.",
                            details=details,
                        )
                    )
                    continue

                if raw_value is None:
                    # optional and not provided: fine
                    continue

                binding = parse_binding(raw_value)

                # external binding
                if binding.kind == "external":
                    key = binding.external_key or ""
                    ext_slot = external_inputs.get(key)
                    if ext_slot is None:
                        issues.append(
                            ValidationIssue(
                                kind="missing_input",
                                step_id=step_id,
                                field=name,
                                message=f"External input '{key}' is not declared in external_inputs.",
                            )
                        )
                    else:
                        if self._is_strict_type_mismatch(slot.type, ext_slot.type):
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

                # literal: we accept for now, maybe add soft checks later
            # register outputs as available for later steps
            for out in spec.outputs:
                available_outputs[f"{step_id}.{out.name}"] = out

        return ValidationResult(ok=(len(issues) == 0), issues=issues)

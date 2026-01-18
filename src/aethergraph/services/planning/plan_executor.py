# aethergraph/services/planning/plan_executor.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from aethergraph.runner import run_async
from aethergraph.services.planning.action_catalog import ActionCatalog
from aethergraph.services.planning.bindings import parse_binding
from aethergraph.services.planning.plan_types import CandidatePlan

ExecutionPhase = Literal[
    "start",
    "step_start",
    "step_success",
    "step_failure",
    "success",
    "failure",
]


@dataclass
class ExecutionEvent:
    """
    Lightweight event emitted during plan execution.

    Useful for logging / UI progress.
    """

    phase: ExecutionPhase
    step_id: str | None = None
    message: str | None = None

    # For success phases
    step_outputs: dict[str, Any] | None = None

    # For failures
    error: Exception | None = None


@dataclass
class ExecutionResult:
    """
    Structured result of executing a plan.

    - ok: True if all steps ran successfully.
    - outputs: outputs of the final step (or {} on failure).
    - errors: list of failure events (usually 0 or 1 in fail-fast mode).
    """

    ok: bool
    outputs: dict[str, Any]
    errors: list[ExecutionEvent] = field(default_factory=list)


@dataclass
class PlanExecutor:
    """
    Execute a validated CandidatePlan by invoking graphfns/graphs via run_async.

    Assumptions:
      - The plan has already been validated by FlowValidator.
      - action_ref strings correspond to registry entries in the ActionCatalog,
        using canonical refs like 'graph:foo@0.1.0' or 'graphfn:bar@0.1.0'.
      - Bindings use the syntax:
        * "${step_id.output_name}" for step outputs
        * "${user.key}" for external/user inputs

    NOTE:
       - Here we use a simple run_async call for each step. In the future we may want
         to support more advanced execution strategies (parallelism, retries, etc.).
       - For v1, we assume this is used at the top level (agent executing plans),
         not inside a graphfn. If you embed it inside a graph, we might later
         adjust how we interact with the registry / context.
    """

    catalog: ActionCatalog

    # convenient access
    @property
    def registry(self):
        return self.catalog.registry

    async def execute(
        self,
        plan: CandidatePlan,
        *,
        user_inputs: dict[str, Any] | None = None,
        on_event: Callable[[ExecutionEvent], None] | None = None,
    ) -> ExecutionResult:
        """
        Execute all steps in the plan in order.

        Args:
            plan: The candidate plan to execute (assumed validated).
            user_inputs: Values that can be referenced as "${user.<key>}".
            on_event: Optional callback to receive ExecutionEvent updates.

        Returns:
            ExecutionResult:
              - ok=True and final step outputs on success.
              - ok=False and errors populated on first failure.
        """
        user_inputs = user_inputs or {}
        step_results: dict[str, dict[str, Any]] = {}  # step_id -> outputs dict
        errors: list[ExecutionEvent] = []

        self._emit(
            on_event,
            ExecutionEvent(
                phase="start",
                message="Starting plan execution.",
            ),
        )

        # For now we assume steps are already in topological order (validator checked cycles).
        for step in plan.steps:
            self._emit(
                on_event,
                ExecutionEvent(
                    phase="step_start",
                    step_id=step.id,
                    message=f"Executing step '{step.id}' with action_ref='{step.action_ref}'.",
                ),
            )

            try:
                # Look up the underlying action object from the registry.
                #
                # After we fixed ActionCatalog to use Key(...).canonical(),
                # action_ref is already the canonical registry key:
                #   e.g. "graph:surrogate_training_pipeline@0.1.0"
                action_obj = self.registry.get(step.action_ref)

                # Resolve inputs (bindings + literals)
                bound_inputs = self._resolve_inputs(
                    raw_inputs=step.inputs or {},
                    step_results=step_results,
                    user_inputs=user_inputs,
                )

                # Run the graphfn/graph via AG runtime
                outputs = await run_async(action_obj, inputs=bound_inputs)

                step_results[step.id] = outputs

                self._emit(
                    on_event,
                    ExecutionEvent(
                        phase="step_success",
                        step_id=step.id,
                        message=f"Step '{step.id}' completed.",
                        step_outputs=outputs,
                    ),
                )

            except Exception as exc:  # noqa: BLE001
                failure_event = ExecutionEvent(
                    phase="step_failure",
                    step_id=step.id,
                    message=f"Step '{step.id}' failed: {exc!r}",
                    error=exc,
                )
                errors.append(failure_event)
                self._emit(on_event, failure_event)

                # For v1: fail fast and mark the whole plan as failed.
                final_failure = ExecutionEvent(
                    phase="failure",
                    step_id=step.id,
                    message="Plan execution aborted due to step failure.",
                    error=exc,
                )
                errors.append(final_failure)
                self._emit(on_event, final_failure)

                return ExecutionResult(
                    ok=False,
                    outputs={},
                    errors=errors,
                )

        # If we reach here, all steps succeeded.
        final_step = plan.steps[-1]
        final_outputs = step_results.get(final_step.id, {})

        success_event = ExecutionEvent(
            phase="success",
            step_id=final_step.id,
            message="Plan execution finished successfully.",
            step_outputs=final_outputs,
        )
        self._emit(on_event, success_event)

        return ExecutionResult(
            ok=True,
            outputs=final_outputs,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _emit(
        on_event: Callable[[ExecutionEvent], None] | None,
        event: ExecutionEvent,
    ) -> None:
        if on_event is None:
            return
        try:
            on_event(event)
        except Exception:
            # Don't let logging/UI errors break execution
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Error in execution on_event callback", exc_info=True)

    def _resolve_inputs(
        self,
        *,
        raw_inputs: dict[str, Any],
        step_results: dict[str, dict[str, Any]],
        user_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Resolve all input values for a step:
          - literals: kept as-is
          - "${step_id.output_name}" -> previous step's outputs
          - "${user.key}" -> user_inputs["key"]

        We do this recursively, so bindings can appear inside nested dicts/lists.
        """

        def resolve_value(val: Any) -> Any:
            # strings: may be bindings
            if isinstance(val, str):
                binding = parse_binding(val)

                if binding.kind == "step_output":
                    src_id = binding.source_step_id or ""
                    out_name = binding.source_output_name or ""
                    try:
                        return step_results[src_id][out_name]
                    except KeyError as exc:  # should be prevented by validator
                        raise KeyError(
                            f"Unknown step output reference: {src_id}.{out_name}"
                        ) from exc

                if binding.kind == "external":
                    key = binding.external_key or ""
                    if key not in user_inputs:
                        raise KeyError(f"Missing user input for external binding: {key}")
                    return user_inputs[key]

                # literal or unsupported binding kinds: keep as-is
                return val

            # dict: resolve recursively
            if isinstance(val, dict):
                return {k: resolve_value(v) for k, v in val.items()}

            # list/tuple: resolve recursively
            if isinstance(val, (list, tuple)):  # noqa: UP038
                return [resolve_value(v) for v in val]

            # anything else: leave unchanged
            return val

        return {name: resolve_value(v) for name, v in raw_inputs.items()}

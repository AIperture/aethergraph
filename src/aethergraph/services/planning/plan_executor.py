from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    step_outputs: dict[str, Any] | None = None
    error: Exception | None = None


@dataclass
class PlanExecutor:
    """
    Execute a validated CandidatePlan by invoking graphfns/graphs via run_async.

    Assumptions:
      - The plan has already been validated by FlowValidator.
      - action_ref strings correspond to registry entries in the ActionCatalog.
      - Bindings use the syntax:
        * "${step_id.output_name}" for step outputs
        * "${user.key}" for external/user inputs (if you use that)

    NOTE:
       - Here we use a simple run_aync call for each step. In the future we may want to support
         more advanced execution strategies (e.g., parallel steps where possible, retries, etc).
         but this is far enough for a first version.
       - We need to confirm if we should use this inside a graph_fn or only at the top level with the agent
         executing plans. If inside a graph_fn, we may need to adjust how we look up action objects
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
    ) -> dict[str, Any]:
        """
        Execute all steps in the plan in order.

        Args:
            plan: The candidate plan to execute.
            user_inputs: Values that can be referenced as "${user.<key>}".
            on_event: Optional callback to receive ExecutionEvent updates.

        Returns:
            The outputs dictionary from the final step.
        """
        user_inputs = user_inputs or {}
        step_results: dict[str, dict[str, Any]] = {}  # step_id -> outputs dict

        self._emit(on_event, ExecutionEvent(phase="start", message="Starting plan execution."))

        # For now we assume steps are already in topological order (validator checked cycles).
        # If we ever need, we can add a topo sort here using the same logic as FlowValidator.
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
                # Look up the underlying action object from the registry
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
                self._emit(
                    on_event,
                    ExecutionEvent(
                        phase="step_failure",
                        step_id=step.id,
                        message=f"Step '{step.id}' failed: {exc!r}",
                        error=exc,
                    ),
                )
                # For first version: fail fast and bubble the exception
                self._emit(
                    on_event,
                    ExecutionEvent(
                        phase="failure",
                        step_id=step.id,
                        message="Plan execution aborted due to step failure.",
                        error=exc,
                    ),
                )
                raise

        # Return outputs of the final step
        final_step = plan.steps[-1]
        final_outputs = step_results.get(final_step.id, {})

        self._emit(
            on_event,
            ExecutionEvent(
                phase="success",
                step_id=final_step.id,
                message="Plan execution finished successfully.",
                step_outputs=final_outputs,
            ),
        )
        return final_outputs

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

# aethergraph/services/planning/plan_executor.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from aethergraph.api.v1.deps import RequestIdentity
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import RunImportance, RunOrigin, RunVisibility
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
    Execute a validated CandidatePlan by invoking graphfns/graphs.

    By default this uses run_async for each step (no RunStore / concurrency limits).
    If a RunManager is provided, steps are executed via:

        submit_run(...) + wait_run(..., return_outputs=True)

    which:
      - honors max_concurrent_runs
      - creates RunRecords visible in the UI
      - still exposes real Python outputs to the planner.

    Assumptions:
      - The plan has already been validated by FlowValidator.
      - action_ref strings correspond to registry entries in the ActionCatalog,
        using canonical refs like 'graph:foo@0.1.0' or 'graphfn:bar@0.1.0'.
      - Bindings use the syntax:
        * "${step_id.output_name}" for step outputs
        * "${user.key}" for external/user inputs
    """

    catalog: ActionCatalog
    # Optional: if provided, we use it instead of run_async for steps
    run_manager: RunManager | None = None

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
        # Optional execution context if using RunManager
        identity: RequestIdentity | None = None,
        visibility: RunVisibility | None = RunVisibility.normal,
        importance: RunImportance | None = RunImportance.normal,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        tags: list[str] | None = None,
        origin: RunOrigin | None = None,
    ) -> ExecutionResult:
        """
        Execute all steps in the plan in order.

        Args:
            plan: The candidate plan to execute (assumed validated).
            user_inputs: Values that can be referenced as "${user.<key>}".
            on_event: Optional callback to receive ExecutionEvent updates.
            identity/session_id/agent_id/app_id/tags/origin:
                Optional context passed through to RunManager when present.

        Returns:
            ExecutionResult:
              - ok=True and final step outputs on success.
              - ok=False and errors populated on first failure.
        """
        user_inputs = user_inputs or {}
        step_results: dict[str, dict[str, Any]] = {}  # step_id -> outputs dict
        errors: list[ExecutionEvent] = []
        base_tags = list(tags or [])

        self._emit(
            on_event,
            ExecutionEvent(
                phase="start",
                message="Starting plan execution.",
            ),
        )

        # Try to extract a plan id (best-effort, no hard dependency on field names)
        plan_id = getattr(plan, "id", None) or getattr(plan, "plan_id", None)

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
                action_obj = self.registry.get(step.action_ref)

                # Resolve inputs (bindings + literals)
                bound_inputs = self._resolve_inputs(
                    raw_inputs=step.inputs or {},
                    step_results=step_results,
                    user_inputs=user_inputs,
                )

                # Decide execution path:
                # - If run_manager is None: use run_async (legacy/simple mode)
                # - Else: use RunManager to honor concurrency + RunStore
                if self.run_manager is None:
                    outputs = await run_async(action_obj, inputs=bound_inputs)
                else:
                    graph_id = self._graph_id_from_action_ref(step.action_ref)

                    # Compose tags for this step-run
                    step_tags = list(base_tags)
                    if plan_id is not None:
                        step_tags.append(f"plan:{plan_id}")
                    step_tags.append(f"plan_step:{step.id}")

                    # Spawn the run
                    run_id = (
                        f"plan-{plan_id or 'na'}-{session_id or 'na'}-{step.id}-{uuid4().hex[:8]}"
                    )
                    run_record = await self.run_manager.submit_run(
                        graph_id=graph_id,
                        inputs=bound_inputs,
                        run_id=run_id,
                        session_id=session_id,
                        identity=identity,
                        origin=origin,
                        visibility=visibility,
                        importance=importance,
                        agent_id=agent_id,
                        app_id=app_id,
                        tags=step_tags,
                    )
                    # Wait for completion and grab real Python outputs
                    finished_rec, outputs = await self.run_manager.wait_run(
                        run_record.run_id,
                        return_outputs=True,
                    )

                    # Interpret non-succeeded as failure
                    status_str = getattr(finished_rec.status, "value", str(finished_rec.status))
                    if status_str != "succeeded":
                        # Mirror the run_async behaviour: raise so we hit the except below
                        raise RuntimeError(
                            f"Run for step '{step.id}' failed with status={status_str}, "
                            f"error={finished_rec.error!r}"
                        )

                    # outputs may still be None if something went very wrong; guard it
                    if outputs is None:
                        outputs = {}

                # Store step outputs for later bindings
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

    @staticmethod
    def _graph_id_from_action_ref(action_ref: str) -> str:
        """
        Extract the graph_id name from a canonical action_ref, e.g.:

            "graph:foo@0.1.0"   -> "foo"
            "graphfn:bar@0.1.0" -> "bar"

        If the ref doesn't match this pattern, we fall back to the tail
        after ":" (or the whole string).
        """
        ref = action_ref
        if ":" in ref:
            _, ref = ref.split(":", 1)
        if "@" in ref:
            ref, _ = ref.split("@", 1)
        return ref

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

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from aethergraph.services.planning.plan_executor import ExecutionEvent

if TYPE_CHECKING:  # avoid runtime circular import
    from .action_catalog import ActionCatalog


@dataclass
class PlanStep:
    """
    Represents a single step in a plan, referencing an action and its inputs.

    Attributes:
        id (str): Unique identifier for the plan step.
        action: short, human/LLM-readable name for the action
        action_ref (str): Reference to the associated action specification. This is a canonicial identifier used in AetherGraph.
        inputs (Dict[str, Any]): Input parameters for the action.

    Methods:
        from_dict(data): Creates a PlanStep instance from a dictionary.
        to_dict(): Serializes the PlanStep instance to a dictionary.

    Notes:
        - this class is usually not accessed unless you are building or inspecting plans directly.
    """

    id: str
    action: str
    action_ref: str  # must match ActionSpec.ref
    inputs: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        # v1 compatibility: some plans only have action_ref
        return cls(
            id=data["id"],
            action=data.get("action"),
            action_ref=data.get("action_ref"),
            inputs=data.get("inputs", {}) or {},
            extras=data.get("extras", {}) or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidatePlan:
    """
    CandidatePlan represents a plan consisting of multiple steps and additional metadata.

    Attributes:
        steps (list[PlanStep]): A list of steps that make up the plan.
        extras (dict[str, Any]): Additional metadata or information associated with the plan.

    Methods:
        from_dict(data: dict[str, Any]) -> CandidatePlan:
            Creates an instance of CandidatePlan from a dictionary representation.
        to_dict() -> dict[str, Any]:
            Converts the CandidatePlan instance into a dictionary representation.
        resolve_actions(
            include_global: bool = True
            Ensures that each step in the plan has both `action` (name) and `action_ref`
            (canonical reference) resolved. Resolves missing fields using the provided
            action catalog.
    """

    steps: list[PlanStep]
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidatePlan:
        steps_data = data.get("steps", []) or []
        steps = [PlanStep.from_dict(step) for step in steps_data]
        extras = data.get("extras", {}) or {}
        return cls(steps=steps, extras=extras)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "extras": self.extras,
        }

    def resolve_actions(
        self,
        catalog: ActionCatalog,
        flow_ids: list[str] | None = None,
        include_global: bool = True,
    ) -> None:
        """
        Ensure every step has both action (name) and action_ref (canonical ref).

        - If only action is set → look up spec by name & fill action_ref.
        - If only action_ref is set → look up spec by ref & fill action.
        - If both are set → we leave them as-is (could optionally assert they match).
        """
        for step in self.steps:
            # already fully polulated
            if step.action and step.action_ref:
                continue

            # only action name present: resolve to ref
            if step.action and not step.action_ref:
                spec = catalog.get_action_by_name(
                    step.action,
                    flow_ids=flow_ids,
                    include_global=include_global,
                )
                if spec is None:
                    raise ValueError(
                        f"Could not resolve action name '{step.action}' to an action spec."
                    )

                step.action_ref = spec.ref

            # only action_ref present: resolve to name
            elif step.action_ref and not step.action:
                spec = catalog.get_action(step.action_ref)
                if spec is None:
                    # keep action_ref, but we lose the nice name
                    continue
                step.action = spec.name


@dataclass
class ValidationIssue:
    """
    Represents a validation issue encountered in a planning process.

    Attributes:
        kind (str): The type of validation issue. Possible values include
            'missing_input', 'unknown_action', 'type_mismatch', 'cycle', etc.
        step_id (str): The identifier of the plan step where the issue occurred.
        field (str): The name of the input, output, or other field related to the issue.
        message (str): A human-readable message describing the issue.
        details (dict[str, Any]): Additional details about the issue, provided as a dictionary.
            Defaults to an empty dictionary.
    """

    kind: str  # 'missing_input', 'unknown_action', 'type_mismatch', 'cycle', ...
    step_id: str  # plan step id where the issue occurred
    field: str  # input name, output name, etc.
    message: str  # human-readable message describing the issue
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    ValidationResult represents the result of validating a plan, including its validity,
    any structural errors, and missing user-provided values.

    Attributes:
        ok (bool): Indicates whether the plan is valid (True) or invalid (False).
        issues (list[ValidationIssue]): A list of validation issues found in the plan.
        has_structural_errors (bool): Indicates if the plan has structural errors.
        missing_user_bindings (dict[str, list[str]]): A dictionary mapping keys to lists of
            locations where user-provided values are missing.
    Methods:
        is_partial_ok() -> bool:
            Checks if the plan is structurally valid but requires user-provided values
            before execution.
        summary() -> str:
            Generates a summary string describing the validity of the plan and any issues
            found. If the plan is valid, the summary states "Plan is valid." If invalid,
            the summary lists the issues with their details.
    """

    ok: bool  # True if plan is valid, False otherwise
    issues: list[ValidationIssue]  # list of validation issues found in the plan

    # differentiate structural vs “needs user values”
    has_structural_errors: bool = False

    # key → list of locations (e.g. ["load.dataset_path", "eval.target_metric"])
    missing_user_bindings: dict[str, list[str]] = field(default_factory=dict)

    def is_partial_ok(self) -> bool:
        """
        Structurally valid, but requires user-provided values
        (e.g. ${user.dataset_path}) before execution.
        """
        return (not self.has_structural_errors) and bool(self.missing_user_bindings)

    def summary(self) -> str:
        """
        Returns a summary string describing the validity of the plan and any issues found.

        Example:
        - If valid: "Plan is valid."
        - If invalid:
            Plan is invalid:
            - [missing_input] step=step1 field=inputA: Input 'inputA' is missing.
            - [unknown_action] step=step2: Action 'actionX' is not recognized.

        """
        if self.ok:
            return "Plan is valid."
        lines = ["Plan is invalid:"]
        for issue in self.issues:
            prefix = f"[{issue.kind}]"
            if issue.step_id:
                prefix += f" step={issue.step_id}"
            if issue.field:
                prefix += f" field={issue.field}"
            lines.append(f"- {prefix}: {issue.message}")
        return "\n".join(lines)


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
    Represents a lightweight event emitted during the planning process.
    This class is intended for use in logging, debugging, and updating UI progress
    (e.g., updating a spinner or status text). It provides optional structured
    payloads for additional context.

    Attributes:
        phase (PlanningPhase): The current phase of the planning process.
        iteration (int): The iteration number within the planning phase.
        message (str | None): An optional message describing the event.
        raw_llm_output (dict[str, Any] | None): An optional dictionary containing
            raw output from the language model.
        plan_dict (dict[str, Any] | None): An optional dictionary representing the
            plan structure.
        validation (ValidationResult | None): An optional validation result
            associated with the event.
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
    """
    Represents the context for planning, including user inputs, external slots,
    memory snippets, and other parameters that guide the planning process.

    Attributes:
        goal (str): The goal or objective of the planning process.
        user_inputs (dict[str, Any]): A dictionary of user-provided inputs.
        external_slots (dict[str, Any]): A dictionary of external slots or parameters.
        memory_snippets (list[str]): A list of memory snippets for narrow, LLM-facing context.
        artifact_snippets (list[str]): A list of artifact snippets for narrow, LLM-facing context.
        flow_ids (list[str] | None): A list of flow IDs representing the tools/graphs
            that are allowed to be used. Defaults to None.
        instruction (str | None): A richer agent context, used by planner code but not
            directly dumped. Can be parsed from skills. Defaults to None.
        allow_partial_plans (bool): Indicates whether the planner should accept
            structurally valid but missing user input plans. Defaults to True.
        preferred_external_keys (list[str] | None): A list of keys that are allowed or
            preferred as `${user.<key>}`. Defaults to None.
    """

    goal: str
    user_inputs: dict[str, Any]
    external_slots: dict[str, Any]

    # Narrow, LLM-facing context
    memory_snippets: list[str] = field(default_factory=list)
    artifact_snippets: list[str] = field(default_factory=list)

    # What tools/graphs we’re allowed to use
    flow_ids: list[str] | None = None

    # richer agent context (used by planner code, not directly dumped), can be parsed from skills
    instruction: str | None = None

    # should planner accept structurally-valid-but-missing-user-input plans?
    allow_partial_plans: bool = True

    # planner hint – which keys are *allowed / preferred* as ${user.<key>}
    preferred_external_keys: list[str] | None = None


@dataclass
class PlanResult:
    """
    PlanResult is a data class that encapsulates the result of a planning operation.

    Attributes:
        plan (CandidatePlan | None): The candidate plan resulting from the planning operation.
            It can be None if no valid plan was generated.
        validation (ValidationResult | None): The validation result associated with the plan.
            It can be None if validation was not performed or not applicable.
        events (list[PlanningEvent]): A list of planning events that occurred during the planning process.
            Defaults to an empty list.
    """

    plan: CandidatePlan | None
    validation: ValidationResult | None
    events: list[PlanningEvent] = field(default_factory=list)


PlanningEventCallback = Callable[[PlanningEvent], None] | Callable[[PlanningEvent], Awaitable[None]]
ExecutionEventCallback = (
    Callable[[ExecutionEvent], None] | Callable[[ExecutionEvent], Awaitable[None]]
)

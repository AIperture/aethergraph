from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

IntentMode = Literal["chat_only", "quick_action", "plan_and_execute"]


@dataclass
class RoutedIntent:
    """
    Result of routing a user turn:
      - How should we handle this? (mode)
      - If planning: which flows are in scope?
      - If quick_action: which quick action?
    """

    mode: IntentMode

    # For planning
    flow_ids: list[str] | None = None

    # For quick action, e.g. `list_recent_runs`
    quick_action_id: str | None = None

    # Freeform extention field.
    # safety flags, strategy hints, etc. without changing the dataclass structure.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    # placeholder for future session-level state
    last_flow_ids: list[str] | None = None


class IntentRouter(Protocol):
    async def route(
        self,
        *,
        user_message: str,
        session_state: SessionState,
    ) -> RoutedIntent: ...


@dataclass
class ValidationIssue:
    kind: str  # 'missing_input', 'unknown_action', 'type_mismatch', 'cycle', ...
    step_id: str  # plan step id where the issue occurred
    field: str  # input name, output name, etc.
    message: str  # human-readable message describing the issue
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool  # True if plan is valid, False otherwise
    issues: list[ValidationIssue]  # list of validation issues found in the plan

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
    flow_ids: list[str] | None = None


class PlanningContextBuilderProtocol:
    async def build(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
    ) -> PlanningContext: ...

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
class SkillInputField:
    name: str
    description: str
    required: bool = True
    infer_from_history: bool = True
    example: str | None = None


@dataclass
class SkillSpec:
    """placement for future skill spec structure."""

    id: str
    title: str
    description: str

    keywords: list[str] = field(default_factory=list)
    preferred_flows: list[str] = field(default_factory=list)

    # Defaults for this skill
    default_intent_mode: str = "chat_only"  # or "plan_and_execute"
    default_reasoning_mode: str = "direct_answer"  # or "plan_graph", etc.

    # Prompts (loaded from markdown sections)
    planning_prompt: str = ""
    chat_prompt: str = ""
    safety_notes: str = ""

    # Original data / metadata
    raw_markdown: str | None = None
    meta: dict[str, Any] | None = None
    input_fields: list[SkillInputField] = field(default_factory=list)


@dataclass
class AgentContextSnapshot:
    """
    Generic snapshot of an agent's context for a given session.

    - recent_chat / summaries / session_state_view can be used by *any* service.
    - last_plans / last_executions are planning history.
    - pending_* captures interactive planning state (clarification / approval).

    TODO: refactor into PlanningSnapshot + AgentSnapshot
    """

    # Optional identity; orchestrator can fill or ignore
    session_id: str | None = None

    # Generic conversational context
    recent_chat: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)

    # Sticky state (things like dataset_path, hyperparams, etc.)
    session_state_view: dict[str, Any] = field(default_factory=dict)

    # Planning / execution history (loose dicts on purpose)
    last_plans: list[dict[str, Any]] = field(default_factory=list)
    last_executions: list[dict[str, Any]] = field(default_factory=list)

    # Pending interactive state (for clarification / approval flows)
    pending_plan: dict[str, Any] | None = None  # serialized CandidatePlan.to_dict()
    pending_user_inputs: dict[str, Any] = field(default_factory=dict)
    # e.g. {"dataset_path": ["load.dataset_path"], "hyperparams": ["train.hyperparams"]}
    pending_missing_inputs: dict[str, list[str]] = field(default_factory=dict)
    pending_question: str | None = None  # last question we asked the user
    # "clarification" | "approval" | None  (no strict enum to keep it light)
    pending_mode: str | None = None

    # Optional: which skill is currently "active" in this session
    active_skill_id: str | None = None


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
    external_slots: dict[str, Any]

    # Narrow, LLM-facing context
    memory_snippets: list[str] = field(default_factory=list)
    artifact_snippets: list[str] = field(default_factory=list)

    # What tools/graphs we’re allowed to use
    flow_ids: list[str] | None = None

    # skill + richer agent context (used by planner code, not directly dumped)
    skill: SkillSpec | None = None
    agent_snapshot: AgentContextSnapshot | None = None

    # should planner accept structurally-valid-but-missing-user-input plans?
    allow_partial_plans: bool = True

    # planner hint – which keys are *allowed / preferred* as ${user.<key>}
    preferred_external_keys: list[str] | None = None


class PlanningContextBuilderProtocol:
    async def build(
        self,
        *,
        user_message: str,
        routed: RoutedIntent,
        session_state: SessionState,
    ) -> PlanningContext: ...

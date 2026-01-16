# aethergraph/services/planning/validation_types.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

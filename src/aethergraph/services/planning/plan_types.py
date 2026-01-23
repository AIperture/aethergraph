from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid runtime circular import
    from .action_catalog import ActionCatalog


@dataclass
class PlanStep:
    """
    Represents a single step in a plan, referencing an action and its inputs.
    Attributes:
        id (str): Unique identifier for the plan step.
        action: short, human/LLM-readable name for the action
        action_ref (str): Reference to the associated action specification.
        inputs (Dict[str, Any]): Input parameters for the action.
    Methods:
        from_dict(data): Creates a PlanStep instance from a dictionary.
        to_dict(): Serializes the PlanStep instance to a dictionary.
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
    Represents a candidate plan consisting of multiple plan steps.
    Attributes:
        steps (List[PlanStep]): The list of steps in the candidate plan.
    Methods:
        from_dict(data): Creates a CandidatePlan instance from a dictionary.
        to_dict(): Serializes the CandidatePlan instance to a dictionary.
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

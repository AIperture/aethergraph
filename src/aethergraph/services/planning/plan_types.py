from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PlanStep:
    """
    Represents a single step in a plan, referencing an action and its inputs.
    Attributes:
        id (str): Unique identifier for the plan step.
        action_ref (str): Reference to the associated action specification.
        inputs (Dict[str, Any]): Input parameters for the action.
    Methods:
        from_dict(data): Creates a PlanStep instance from a dictionary.
        to_dict(): Serializes the PlanStep instance to a dictionary.
    """

    id: str
    action_ref: str  # must match ActionSpec.ref
    inputs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlanStep:
        return cls(
            id=data["id"],
            action_ref=data["action_ref"],
            inputs=data.get("inputs", {}),
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidatePlan:
        steps_data = data.get("steps", []) or []
        steps = [PlanStep.from_dict(step) for step in steps_data]
        return cls(steps=steps)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

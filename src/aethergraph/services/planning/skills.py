# aethergraph/services/planning/skills.py
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .plan_types import SkillInputSpec, SkillSpec


@dataclass
class SkillRegistry:
    """
    Minimal registry for planning skills.

    Not a giant framework; just enough to:
      - register skills by id,
      - look them up,
      - iterate over them.
    """

    skills: dict[str, SkillSpec] = field(default_factory=dict)

    def register(self, skill: SkillSpec) -> None:
        self.skills[skill.id] = skill

    def get(self, skill_id: str) -> SkillSpec | None:
        return self.skills.get(skill_id)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self.skills

    def all(self) -> Iterable[SkillSpec]:
        return self.skills.values()


# ---------------------------------------------------------------------------
# Concrete skills
# ---------------------------------------------------------------------------

SURROGATE_SKILL: SkillSpec = SkillSpec.with_inputs(
    id="surrogate_modeling",
    title="Surrogate Modeling Workflow",
    description=(
        "End-to-end surrogate modeling for scientific / engineering simulations, "
        "including dataset loading, training, and evaluation on parameter grids."
    ),
    flow_ids=["surrogate_training_flow"],
    planning_prompt="""
You are constructing workflows for training and evaluating surrogate models
on scientific simulation data (e.g., diffraction, metalens, data-center physics).

Typical workflows include:
- loading a dataset,
- splitting into train/validation sets,
- training a surrogate model,
- evaluating it on a specified grid or parameter sweep,
- optionally exporting metrics and plots.

Prefer concise, minimal sequences of steps that are easy to debug.
Do not invent fake paths or hyperparameters; bind to user inputs instead.
""".strip(),
    parsing_prompt="""
You are extracting structured arguments for surrogate modeling workflows.
Values may include file paths, numeric ratios, JSON-like hyperparameters,
and grid specifications. Avoid hallucinating values and set unknowns to null.
""".strip(),
    inputs=[
        SkillInputSpec(
            name="dataset_path",
            description="Path to the training dataset file (CSV, parquet, etc.).",
            required=True,
            example="/data/metalens/dataset.csv",
            parse_hint="filesystem path as string",
        ),
        SkillInputSpec(
            name="grid_spec",
            description="Grid definition for evaluation (bounds and resolution).",
            required=False,
            example={"x1_bounds": [0, 1], "x2_bounds": [0, 1], "resolution": 10},
            parse_hint="JSON object with bounds and resolution",
        ),
        SkillInputSpec(
            name="hyperparams",
            description="Training hyperparameters (lr, epochs, etc.).",
            required=False,
            example={"lr": 0.01, "epochs": 20},
            parse_hint="JSON object with ML hyperparameters",
        ),
        SkillInputSpec(
            name="train_ratio",
            description="Fraction of data used for training (0-1).",
            required=False,
            example=0.8,
            parse_hint="float between 0 and 1",
        ),
    ],
)


def build_default_skill_registry() -> SkillRegistry:
    """
    Factory for a registry containing built-in planning skills.

    You can extend this without breaking API: just register more skills here.
    """
    reg = SkillRegistry()
    reg.register(SURROGATE_SKILL)
    return reg

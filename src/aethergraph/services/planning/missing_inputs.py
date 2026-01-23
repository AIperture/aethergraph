from __future__ import annotations

from dataclasses import dataclass

from aethergraph.contracts.services.planning import ValidationResult


@dataclass
class MissingUserInput:
    """
    Represents one missing external user binding like ${user.dataset_path}.

    key:       the user.<key> part, e.g. "dataset_path"
    locations: where in the plan this key is referenced, e.g. ["load.dataset_path"]
    """

    key: str
    locations: list[str]


def get_missing_user_inputs(result: ValidationResult) -> list[MissingUserInput]:
    """
    Convert ValidationResult.missing_user_bindings into a nicer list structure.
    """
    items: list[MissingUserInput] = []
    for key, locs in (result.missing_user_bindings or {}).items():
        items.append(MissingUserInput(key=key, locations=list(locs)))
    return items

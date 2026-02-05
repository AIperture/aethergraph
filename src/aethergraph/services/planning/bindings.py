# aethergraph/services/planning/bindings.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class InputBinding:
    kind: Literal["literal", "external", "step_output"]
    value: Any
    source_step_id: str | None = None
    source_output_name: str | None = None
    external_key: str | None = None


def parse_binding(raw: Any) -> InputBinding:
    """
    Parses a raw binding representation into an InputBinding object.

    Rules:
    - If `raw` is not a string, it is treated as a literal value. (which may include numbers, booleans, lists, dicts, etc.)
    - If `raw` is a string in the format `${user.key}`, it is treated as an external binding.
    - If `raw` is a string in the format `${step_id.output_name}`, it is treated as a step output binding.

    Args:
        raw (Any): The raw binding representation, which can be a literal value or a dict specifying the binding type.

    Returns:
        InputBinding: The parsed InputBinding object.
    """
    if not isinstance(raw, str):
        return InputBinding(kind="literal", value=raw)

    if raw.startswith("${") and raw.endswith("}"):
        inner = raw[2:-1].strip()
        if inner.startswith("user."):
            key = inner.split(".", 1)[1]
            return InputBinding(kind="external", value=None, external_key=key)

        # step_id.output_name
        parts = inner.split(".", 1)
        if len(parts) == 2:
            step_id, output_name = parts
            return InputBinding(
                kind="step_output",
                value=None,
                source_step_id=step_id,
                source_output_name=output_name,
            )

        # Fallback to literal if unrecognized
        return InputBinding(kind="literal", value=raw)

    return InputBinding(kind="literal", value=raw)

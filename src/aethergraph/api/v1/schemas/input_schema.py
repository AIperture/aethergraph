from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel  # type: ignore

JsonFieldType = Literal["string", "number", "boolean", "object", "array", "any"]
InputWidget = Literal["text", "textarea", "number", "switch", "json"]


class InputFieldSpec(BaseModel):
    name: str
    type: JsonFieldType | None = None
    required: bool = True
    default: Any | None = None
    description: str | None = None
    label: str | None = None
    placeholder: str | None = None
    widget: InputWidget | None = None

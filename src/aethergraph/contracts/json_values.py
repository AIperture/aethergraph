"""Public JSON value types used at authored AetherGraph boundaries."""

from __future__ import annotations

from typing import TypeAlias

from pydantic import JsonValue

JsonScalar: TypeAlias = None | bool | int | float | str

__all__ = ["JsonScalar", "JsonValue"]

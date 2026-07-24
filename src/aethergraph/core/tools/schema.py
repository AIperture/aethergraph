"""Canonical JSON Schema handling for public Tool arguments."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, best_match

_FULL_SCHEMA_ROOT_KEYS = frozenset(
    {
        "$defs",
        "$id",
        "$schema",
        "$ref",
        "additionalProperties",
        "allOf",
        "anyOf",
        "description",
        "oneOf",
        "properties",
        "required",
        "title",
        "type",
        "unevaluatedProperties",
    }
)


@dataclass(frozen=True)
class ToolSchemaIssue:
    """Describe one exact Tool argument validation failure."""

    path: str
    message: str
    validator: str


def normalize_tool_args_schema(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return one closed Draft 2020-12 object schema for Tool arguments.

    The public API accepts either a complete object schema or the explicit
    compact field-map shorthand used by simple built-in Tools. Both forms
    normalize once to the same canonical representation.
    """

    if schema is None:
        value: dict[str, Any] = {}
    elif isinstance(schema, Mapping):
        value = deepcopy(dict(schema))
    else:
        raise TypeError("tool args_schema must be a dictionary")

    if _looks_like_full_schema(value):
        if value.get("type") != "object" or not isinstance(value.get("properties"), Mapping):
            raise ValueError(
                "tool args_schema must be a Draft 2020-12 object schema with properties"
            )
        value["properties"] = deepcopy(dict(value["properties"]))
        value.setdefault("additionalProperties", False)
    else:
        value = _expand_compact_field_map(value)

    value = _standardize_nullable(value)
    _validate_local_references(value)
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as exc:
        location = "/".join(str(item) for item in exc.absolute_schema_path)
        suffix = f" at {location}" if location else ""
        raise ValueError(f"tool args_schema is invalid{suffix}: {exc.message}") from exc
    return value


def validate_tool_args(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "args",
) -> ToolSchemaIssue | None:
    """Validate one Tool argument value against its canonical schema."""

    error = best_match(Draft202012Validator(dict(schema)).iter_errors(value))
    if error is None:
        return None
    segments = [path, *(str(item) for item in error.absolute_path)]
    return ToolSchemaIssue(
        path=".".join(segments),
        message=error.message,
        validator=str(error.validator or ""),
    )


def _looks_like_full_schema(schema: Mapping[str, Any]) -> bool:
    return bool(set(schema) & _FULL_SCHEMA_ROOT_KEYS)


def _expand_compact_field_map(schema: Mapping[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for raw_name, raw_spec in schema.items():
        name = str(raw_name)
        if not name:
            raise ValueError("tool compact args_schema field names must be non-empty")
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"tool compact args_schema field {name!r} must be a dictionary")
        spec = deepcopy(dict(raw_spec))
        marker = spec.pop("required", False)
        if not isinstance(marker, bool):
            raise TypeError(f"tool compact args_schema field {name!r} required must be a boolean")
        properties[name] = spec
        if marker:
            required.append(name)
    normalized: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        normalized["required"] = required
    return normalized


def _standardize_nullable(value: Any) -> Any:
    if isinstance(value, list):
        return [_standardize_nullable(item) for item in value]
    if not isinstance(value, Mapping):
        return deepcopy(value)
    normalized = {
        str(key): _standardize_nullable(item) for key, item in value.items() if key != "nullable"
    }
    nullable = value.get("nullable")
    if nullable is None or nullable is False:
        return normalized
    if nullable is not True:
        raise TypeError("tool args_schema nullable must be a boolean")
    if normalized.get("type") == "null":
        return normalized
    return {"anyOf": [normalized, {"type": "null"}]}


def _validate_local_references(schema: dict[str, Any]) -> None:
    def resolve(reference: str) -> Any:
        if reference == "#":
            return schema
        if not reference.startswith("#/"):
            raise ValueError("tool args_schema supports only local $ref values beginning with '#/'")
        current: Any = schema
        for raw_token in reference[2:].split("/"):
            token = raw_token.replace("~1", "/").replace("~0", "~")
            if not isinstance(current, Mapping) or token not in current:
                raise ValueError(
                    f"tool args_schema contains unresolved local reference {reference!r}"
                )
            current = current[token]
        return current

    def visit(node: Any, active: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item, active)
            return
        if not isinstance(node, Mapping):
            return
        reference = node.get("$ref")
        if reference is not None:
            if not isinstance(reference, str):
                raise TypeError("tool args_schema $ref must be a string")
            if reference in active:
                cycle = " -> ".join((*active, reference))
                raise ValueError(f"tool args_schema contains a recursive reference cycle: {cycle}")
            visit(resolve(reference), (*active, reference))
        discriminator = node.get("discriminator")
        if isinstance(discriminator, Mapping):
            mapping = discriminator.get("mapping")
            if isinstance(mapping, Mapping):
                for target in mapping.values():
                    if not isinstance(target, str):
                        raise TypeError("tool args_schema discriminator targets must be strings")
                    resolve(target)
        for key, item in node.items():
            if key != "$ref":
                visit(item, active)

    visit(schema, ())


__all__ = [
    "ToolSchemaIssue",
    "normalize_tool_args_schema",
    "validate_tool_args",
]

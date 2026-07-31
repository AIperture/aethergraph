"""Provider-neutral JSON Schema validation diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match


@dataclass(frozen=True)
class SchemaValidationIssue:
    """Describe one exact, bounded JSON Schema validation failure."""

    path: str
    message: str
    validator: str
    schema_path: str = ""
    invalid_value: str = ""
    expected: tuple[Any, ...] = ()
    repair_hint: str = ""


def first_schema_issue(
    value: Any,
    schema: Mapping[str, Any],
    *,
    path: str = "$",
) -> SchemaValidationIssue | None:
    """Return the most specific bounded Draft 2020-12 validation issue.

    Discriminated ``oneOf`` and ``anyOf`` schemas are resolved using a shared
    ``const`` field when present. This prevents callers from receiving only a
    generic "not valid under any schemas" message for operator-shaped inputs.
    """

    errors = list(Draft202012Validator(dict(schema)).iter_errors(value))
    if not errors:
        return None
    error = errors[0] if len(errors) == 1 else best_match(errors)
    if error is None:
        return None
    selected = _select_specific_error(error)
    if isinstance(selected, SchemaValidationIssue):
        return SchemaValidationIssue(
            path=_join_instance_path(path, selected.path),
            message=selected.message,
            validator=selected.validator,
            schema_path=selected.schema_path,
            invalid_value=selected.invalid_value,
            expected=selected.expected,
            repair_hint=selected.repair_hint,
        )
    return SchemaValidationIssue(
        path=_join_instance_path(path, selected.absolute_path),
        message=str(selected.message),
        validator=str(selected.validator or ""),
        schema_path=_schema_pointer(selected.absolute_schema_path),
        invalid_value=_bounded_json(selected.instance),
        expected=_expected_values(selected),
    )


def _select_specific_error(
    error: ValidationError,
) -> ValidationError | SchemaValidationIssue:
    additional = _additional_properties_issue(error)
    if additional is not None:
        return additional
    prohibited = _prohibited_fields_issue(error)
    if prohibited is not None:
        return prohibited
    discriminated = _discriminated_union_issue(error)
    if discriminated is not None:
        return discriminated
    if not error.context:
        return error
    nested = best_match(error.context)
    return _select_specific_error(nested) if nested is not None else error


def _additional_properties_issue(
    error: ValidationError,
) -> SchemaValidationIssue | None:
    if error.validator != "additionalProperties" or not isinstance(error.instance, dict):
        return None
    schema = error.schema if isinstance(error.schema, Mapping) else {}
    declared = set(str(key) for key in dict(schema.get("properties") or {}))
    patterns = tuple(str(key) for key in dict(schema.get("patternProperties") or {}))
    extras = [
        str(key)
        for key in error.instance
        if str(key) not in declared
        and not any(re.search(pattern, str(key)) for pattern in patterns)
    ]
    if not extras:
        return None
    first_field = extras[0]
    quoted = ", ".join(repr(item) for item in extras)
    return SchemaValidationIssue(
        path=_join_instance_path("", [*error.absolute_path, first_field]),
        message=f"Undeclared field{'s' if len(extras) != 1 else ''} {quoted} are not allowed.",
        validator="additionalProperties",
        schema_path=_schema_pointer(error.absolute_schema_path),
        invalid_value=_bounded_json(error.instance.get(first_field)),
        repair_hint=(
            f"Remove undeclared field{'s' if len(extras) != 1 else ''} {quoted} "
            "or use the declared argument names."
        ),
    )


def _prohibited_fields_issue(error: ValidationError) -> SchemaValidationIssue | None:
    """Explain a matched ``not`` branch using the fields that made it match."""

    if error.validator != "not" or not isinstance(error.instance, dict):
        return None
    forbidden = _matched_required_fields(error.validator_value, error.instance)
    if not forbidden:
        return None
    field_names = tuple(dict.fromkeys(forbidden))
    first_field = field_names[0]
    quoted = ", ".join(repr(item) for item in field_names)
    return SchemaValidationIssue(
        path=_join_instance_path("", [*error.absolute_path, first_field]),
        message=f"Prohibited field{'s' if len(field_names) != 1 else ''} {quoted} matched a forbidden schema branch.",
        validator="not",
        schema_path=_schema_pointer(error.absolute_schema_path),
        invalid_value=_bounded_json(error.instance.get(first_field)),
        repair_hint=(
            f"Remove prohibited field{'s' if len(field_names) != 1 else ''} "
            f"{quoted} from this tool call."
        ),
    )


def _matched_required_fields(schema: Any, instance: dict[str, Any]) -> list[str]:
    if not isinstance(schema, Mapping):
        return []
    required = schema.get("required")
    if (
        isinstance(required, list)
        and required
        and all(str(field) in instance for field in required)
    ):
        return [str(field) for field in required]
    fields: list[str] = []
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if not isinstance(branches, list):
            continue
        for branch in branches:
            fields.extend(_matched_required_fields(branch, instance))
    return fields


def _discriminated_union_issue(
    error: ValidationError,
) -> SchemaValidationIssue | ValidationError | None:
    if error.validator not in {"oneOf", "anyOf"} or not isinstance(error.instance, dict):
        return None
    branches = list(error.validator_value or [])
    if not branches or any(not isinstance(branch, Mapping) for branch in branches):
        return None

    candidates: dict[str, tuple[Any, ...]] = {}
    for branch in branches:
        properties = branch.get("properties")
        if not isinstance(properties, Mapping):
            continue
        for field_name, field_schema in properties.items():
            if not isinstance(field_schema, Mapping) or "const" not in field_schema:
                continue
            name = str(field_name)
            candidates[name] = (
                *candidates.get(name, ()),
                deepcopy(field_schema["const"]),
            )
    shared_fields = [
        field_name for field_name, values in candidates.items() if len(values) == len(branches)
    ]
    if len(shared_fields) != 1:
        return None

    field_name = shared_fields[0]
    allowed = candidates[field_name]
    actual = error.instance.get(field_name)
    if actual not in allowed:
        return SchemaValidationIssue(
            path=_join_instance_path("", [*error.absolute_path, field_name]),
            message=f"{actual!r} is not an allowed {field_name}.",
            validator="const",
            schema_path=_schema_pointer(error.absolute_schema_path),
            invalid_value=_bounded_json(actual),
            expected=allowed,
        )

    selected_index = allowed.index(actual)
    parent_schema_path = tuple(error.absolute_schema_path)
    matching = [
        item
        for item in error.context
        if tuple(item.absolute_schema_path)[: len(parent_schema_path)] == parent_schema_path
        and len(tuple(item.absolute_schema_path)) > len(parent_schema_path)
        and tuple(item.absolute_schema_path)[len(parent_schema_path)] == selected_index
    ]
    nested = best_match(matching)
    return _select_specific_error(nested) if nested is not None else None


def _join_instance_path(root: str, segments: Any) -> str:
    result = str(root)
    raw_segments = [segments] if isinstance(segments, str) else list(segments or [])
    for segment in raw_segments:
        if isinstance(segment, int):
            result += f"[{segment}]"
        elif result:
            result += f".{segment}"
        else:
            result = str(segment)
    return result


def _schema_pointer(segments: Any) -> str:
    values = [str(item).replace("~", "~0").replace("/", "~1") for item in list(segments or [])]
    return "#" + "".join(f"/{item}" for item in values)


def _expected_values(error: ValidationError) -> tuple[Any, ...]:
    if error.validator == "const":
        return (deepcopy(error.validator_value),)
    if error.validator == "enum":
        return tuple(deepcopy(list(error.validator_value or [])))
    if error.validator == "type":
        raw = error.validator_value
        return tuple(raw) if isinstance(raw, list) else (raw,)
    return ()


def _bounded_json(value: Any, *, limit: int = 500) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = ["SchemaValidationIssue", "first_schema_issue"]

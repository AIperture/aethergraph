from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

TOOL_DEFINITION_API_VERSION = "aethergraph.tool/v1"
TOOL_APPROVAL_TIERS = frozenset({"none", "expensive", "always"})


def schema_from_annotation(annotation: Any) -> dict[str, Any]:
    """Return the bounded public JSON Schema projection for one annotation."""

    if annotation in {inspect.Signature.empty, Any}:
        return {}
    if annotation is None or annotation is NoneType:
        return {"type": "null"}
    primitive = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }.get(annotation)
    if primitive is not None:
        return {"type": primitive}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        return {
            "type": "array",
            "items": schema_from_annotation(args[0]) if args else {},
        }
    if origin is dict:
        if args and args[0] not in {str, Any}:
            raise TypeError("Tool dictionary annotations require string keys")
        return {
            "type": "object",
            "additionalProperties": (schema_from_annotation(args[1]) if len(args) > 1 else True),
        }
    if origin in {Union, UnionType}:
        variants = [schema_from_annotation(item) for item in args]
        non_null = [item for item in variants if item.get("type") != "null"]
        if len(non_null) == 1 and len(non_null) != len(variants):
            return {**non_null[0], "nullable": True}
        return {"anyOf": variants}
    if origin is Literal:
        values = list(args)
        schema: dict[str, Any] = {"enum": values}
        value_types = {type(item) for item in values}
        if len(value_types) == 1:
            kind = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(
                next(iter(value_types))
            )
            if kind is not None:
                schema["type"] = kind
        return schema
    if inspect.isclass(annotation) and hasattr(annotation, "__aether_state_definition__"):
        definition = annotation.__aether_state_definition__
        value = getattr(definition, "schema", None)
        return deepcopy(value) if isinstance(value, dict) else {"type": "object"}
    return {}


def injection_kind(name: str, annotation: Any) -> str | None:
    """Return the stable runtime-injection kind for one Tool parameter."""

    declared = getattr(annotation, "__aether_injection_kind__", None)
    if isinstance(declared, str) and declared:
        return declared
    annotation_name = str(getattr(annotation, "__name__", "") or "")
    annotation_module = str(getattr(annotation, "__module__", "") or "")
    if annotation_name == "NodeContext" and annotation_module.startswith("aethergraph"):
        return "node_context"
    reserved = {
        "domain_state": "domain_state",
        "agent_context": "agent_context",
        "node_context": "node_context",
    }
    return reserved.get(name)


@dataclass(frozen=True)
class ToolDefinition:
    """Stable SDK declaration attached to every public ``@tool`` callable."""

    name: str
    description: str
    version: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    args_schema: dict[str, Any]
    result_schema: dict[str, Any]
    approval: Literal["none", "expensive", "always"] = "none"
    injections: tuple[tuple[str, str], ...] = ()
    implementation_module: str = ""
    implementation_symbol: str = ""
    api_version: Literal["aethergraph.tool/v1"] = TOOL_DEFINITION_API_VERSION
    kind: Literal["tool"] = "tool"

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_version": self.api_version,
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "args_schema": deepcopy(self.args_schema),
            "result_schema": deepcopy(self.result_schema),
            "approval": self.approval,
            "injections": [
                {"parameter": parameter, "kind": kind} for parameter, kind in self.injections
            ],
            "implementation_module": self.implementation_module,
            "implementation_symbol": self.implementation_symbol,
        }


def build_tool_definition(
    impl: Any,
    *,
    name: str | None,
    description: str | None,
    version: str,
    inputs: list[str] | None,
    outputs: list[str] | None,
    args_schema: dict[str, Any] | None,
    result_schema: dict[str, Any] | None,
    approval: str,
) -> ToolDefinition:
    if approval not in TOOL_APPROVAL_TIERS:
        raise ValueError("tool approval must be none, expensive, or always")
    signature = inspect.signature(impl)
    try:
        type_hints = get_type_hints(impl)
    except (NameError, TypeError):
        type_hints = {}
    injections: list[tuple[str, str]] = []
    inferred_inputs: list[str] = []
    properties: dict[str, Any] = {}
    required: list[str] = []
    accepts_variadic_keywords = False
    for parameter in signature.parameters.values():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            # Graph/runtime helpers may intentionally accept variadic values.
            # Agent authoring inspection applies its narrower source contract.
            accepts_variadic_keywords = (
                accepts_variadic_keywords or parameter.kind is parameter.VAR_KEYWORD
            )
            continue
        annotation = type_hints.get(parameter.name, parameter.annotation)
        kind = injection_kind(parameter.name, annotation)
        if kind is not None:
            injections.append((parameter.name, kind))
            continue
        inferred_inputs.append(parameter.name)
        properties[parameter.name] = schema_from_annotation(annotation)
        if parameter.default is inspect.Signature.empty:
            required.append(parameter.name)
        else:
            properties[parameter.name]["default"] = deepcopy(parameter.default)
    selected_inputs = tuple(inputs or inferred_inputs)
    unknown_inputs = sorted(set(selected_inputs) - set(inferred_inputs))
    if unknown_inputs and not accepts_variadic_keywords:
        raise ValueError(
            "tool inputs contain unknown or injected parameters: " + ", ".join(unknown_inputs)
        )
    inferred_args_schema: dict[str, Any] = {
        "type": "object",
        "properties": {key: properties.get(key, {}) for key in selected_inputs},
        "additionalProperties": False,
    }
    selected_required = [key for key in required if key in selected_inputs]
    if selected_required:
        inferred_args_schema["required"] = selected_required
    return ToolDefinition(
        name=name or str(getattr(impl, "__name__", "tool")),
        description=(description or inspect.getdoc(impl) or "").strip(),
        version=str(version or "1"),
        inputs=selected_inputs,
        outputs=tuple(outputs or ["result"]),
        args_schema=deepcopy(args_schema) if args_schema is not None else inferred_args_schema,
        result_schema=(
            deepcopy(result_schema)
            if result_schema is not None
            else schema_from_annotation(type_hints.get("return", signature.return_annotation))
        ),
        approval=approval,  # type: ignore[arg-type]
        injections=tuple(injections),
        implementation_module=str(getattr(impl, "__module__", "") or ""),
        implementation_symbol=str(getattr(impl, "__name__", "") or ""),
    )


__all__ = [
    "TOOL_APPROVAL_TIERS",
    "TOOL_DEFINITION_API_VERSION",
    "ToolDefinition",
    "build_tool_definition",
    "injection_kind",
    "schema_from_annotation",
]

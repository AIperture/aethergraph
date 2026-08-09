from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import re
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from .schema import normalize_tool_args_schema

TOOL_DEFINITION_API_VERSION = "aethergraph.tool/v6"
TOOL_APPROVAL_TIERS = frozenset({"none", "expensive", "always"})
TOOL_AVAILABILITY = frozenset({"normal", "plan_proposal", "plan_lifecycle"})
TOOL_EXPOSURES = frozenset({"immediate", "deferred"})


def _normalized_discovery_strings(
    values: list[str] | tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value or "").strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"tool discovery {field_name} must not contain empty values")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"tool discovery {field_name} must not contain duplicates")
    if len(normalized) > 32:
        raise ValueError(f"tool discovery {field_name} must not contain more than 32 values")
    if any(len(value) > 200 for value in normalized):
        raise ValueError(f"tool discovery {field_name} values must not exceed 200 characters")
    return normalized


@dataclass(frozen=True)
class ToolDiscoveryMetadata:
    """Declare compact provider-neutral discovery metadata for one Tool."""

    path: str
    summary: str = ""
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize compact discovery metadata.

        Metadata is authored, schema-free catalog text. It never grants Tool
        execution permission or replaces runtime policy checks.

        Examples:
            Declare a searchable change Tool:
                ```python
                metadata = ToolDiscoveryMetadata(
                    path="studio.change.agents",
                    summary="Replace Agent Tool assignments.",
                    aliases=("assign tools",),
                    tags=("agent",),
                    effects=("project_write_proposal",),
                )
                assert metadata.path == "studio.change.agents"
                ```

            Declare path-only metadata:
                ```python
                metadata = ToolDiscoveryMetadata(path="studio.read")
                assert metadata.summary == ""
                ```

        Args:
            self: Newly initialized discovery metadata.

        Returns:
            None: Validates and normalizes the frozen declaration.

        Notes:
            Full Tool schemas remain in the executable Tool definition and are
            intentionally excluded from this value.
        """

        path = str(self.path or "").strip()
        if re.fullmatch(
            r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*)*",
            path,
        ) is None or len(path) > 120:
            raise ValueError(
                "tool discovery path must be a lowercase dotted stable identifier"
            )
        summary = str(self.summary or "").strip()
        if len(summary) > 500:
            raise ValueError("tool discovery summary must not exceed 500 characters")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(
            self,
            "aliases",
            _normalized_discovery_strings(self.aliases, field_name="aliases"),
        )
        object.__setattr__(
            self,
            "tags",
            _normalized_discovery_strings(self.tags, field_name="tags"),
        )
        object.__setattr__(
            self,
            "effects",
            _normalized_discovery_strings(self.effects, field_name="effects"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical serialized discovery metadata.

        The projection uses ordered lists so contract digests remain stable
        across process boundaries and JSON encoders.

        Examples:
            Serialize complete metadata:
                ```python
                metadata = ToolDiscoveryMetadata(
                    path="studio.read", aliases=("lookup",)
                )
                assert metadata.to_dict()["aliases"] == ["lookup"]
                ```

            Serialize empty optional fields:
                ```python
                metadata = ToolDiscoveryMetadata(path="studio.read")
                assert metadata.to_dict()["effects"] == []
                ```

        Args:
            self: Normalized discovery metadata.

        Returns:
            dict[str, Any]: Detached canonical metadata mapping.

        Notes:
            The returned mapping contains no executable handler or schema.
        """

        return {
            "path": self.path,
            "summary": self.summary,
            "aliases": list(self.aliases),
            "tags": list(self.tags),
            "effects": list(self.effects),
        }


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
    examples: tuple[dict[str, Any], ...] = ()
    slot_outputs: tuple[dict[str, Any], ...] = ()
    availability: Literal["normal", "plan_proposal", "plan_lifecycle"] = "normal"
    approval: Literal["none", "expensive", "always"] = "none"
    exposure: Literal["immediate", "deferred"] = "immediate"
    discovery: ToolDiscoveryMetadata | None = None
    injections: tuple[tuple[str, str], ...] = ()
    implementation_module: str = ""
    implementation_symbol: str = ""
    api_version: Literal["aethergraph.tool/v5"] = TOOL_DEFINITION_API_VERSION
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
            "examples": deepcopy(list(self.examples)),
            "slot_outputs": deepcopy(list(self.slot_outputs)),
            "availability": self.availability,
            "approval": self.approval,
            "exposure": self.exposure,
            "discovery": (None if self.discovery is None else self.discovery.to_dict()),
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
    examples: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    slot_outputs: list[Any] | tuple[Any, ...] | None,
    availability: str,
    approval: str,
    exposure: str,
    discovery: ToolDiscoveryMetadata | None,
) -> ToolDefinition:
    if approval not in TOOL_APPROVAL_TIERS:
        raise ValueError("tool approval must be none, expensive, or always")
    if availability not in TOOL_AVAILABILITY:
        raise ValueError("tool availability must be normal, plan_proposal, or plan_lifecycle")
    if exposure not in TOOL_EXPOSURES:
        raise ValueError("tool exposure must be immediate or deferred")
    if discovery is not None and not isinstance(discovery, ToolDiscoveryMetadata):
        raise TypeError("tool discovery must be ToolDiscoveryMetadata or None")
    if exposure == "deferred" and discovery is None:
        raise ValueError("deferred tools require discovery metadata")
    if exposure == "deferred" and discovery is not None and not discovery.summary:
        raise ValueError("deferred tools require a non-empty discovery summary")
    if result_schema is not None and result_schema.get("type") != "object":
        raise ValueError(
            "tool result_schema must be an object schema for the structured " "result data payload"
        )
    normalized_examples = _normalize_mapping_items(examples, field_name="examples")
    normalized_slot_outputs = _normalize_slot_outputs(slot_outputs)
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
        args_schema=normalize_tool_args_schema(
            args_schema if args_schema is not None else inferred_args_schema
        ),
        result_schema=(
            deepcopy(result_schema)
            if result_schema is not None
            else schema_from_annotation(type_hints.get("return", signature.return_annotation))
        ),
        examples=normalized_examples,
        slot_outputs=normalized_slot_outputs,
        availability=availability,  # type: ignore[arg-type]
        approval=approval,  # type: ignore[arg-type]
        exposure=exposure,  # type: ignore[arg-type]
        discovery=discovery,
        injections=tuple(injections),
        implementation_module=str(getattr(impl, "__module__", "") or ""),
        implementation_symbol=str(getattr(impl, "__name__", "") or ""),
    )


def _normalize_mapping_items(
    values: list[Any] | tuple[Any, ...] | None,
    *,
    field_name: str,
    allow_to_dict: bool = False,
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    for value in values or ():
        item = value
        if allow_to_dict and callable(getattr(item, "to_dict", None)):
            item = item.to_dict()
        if not isinstance(item, dict):
            raise TypeError(f"tool {field_name} entries must be dictionaries")
        normalized.append(deepcopy(item))
    return tuple(normalized)


def _normalize_slot_outputs(
    values: list[Any] | tuple[Any, ...] | None,
) -> tuple[dict[str, Any], ...]:
    items = _normalize_mapping_items(
        values,
        field_name="slot_outputs",
        allow_to_dict=True,
    )
    normalized: list[dict[str, Any]] = []
    keys: set[str] = set()
    for item in items:
        unexpected = sorted(set(item) - {"slot_key", "required"})
        if unexpected:
            raise ValueError(
                "tool slot_outputs contain unsupported fields: " + ", ".join(unexpected)
            )
        key = str(item.get("slot_key") or "")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", key):
            raise ValueError("tool slot_outputs require stable semantic slot_key values")
        if key in keys:
            raise ValueError("tool slot_outputs cannot contain duplicate slot keys")
        required = item.get("required", False)
        if not isinstance(required, bool):
            raise TypeError("tool slot_outputs required must be a boolean")
        keys.add(key)
        normalized.append({"slot_key": key, "required": required})
    return tuple(normalized)


__all__ = [
    "TOOL_APPROVAL_TIERS",
    "TOOL_AVAILABILITY",
    "TOOL_DEFINITION_API_VERSION",
    "TOOL_EXPOSURES",
    "ToolDiscoveryMetadata",
    "ToolDefinition",
    "build_tool_definition",
    "injection_kind",
    "schema_from_annotation",
]

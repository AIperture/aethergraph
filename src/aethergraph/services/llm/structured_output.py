"""Provider capability and schema projection for structured LLM output."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import re
from typing import Any, Literal

from .types import LLMStructuredOutputCapabilityError, StructuredOutputRequest

StructuredOutputPolicy = Literal["best_available", "native_required"]
StructuredOutputMode = Literal[
    "native_strict",
    "native_schema",
    "json_object",
    "prompt_json",
]


@dataclass(frozen=True)
class StructuredOutputCapabilities:
    """Declared structured-output capabilities for one provider/model pair."""

    provider: str
    model: str
    native_strict_schema: bool
    native_schema: bool
    json_object: bool
    prompt_json: bool
    source: str


@dataclass(frozen=True)
class SchemaProjectionDiagnostic:
    """One deterministic reason a stronger schema mode was not selected."""

    code: str
    path: str
    message: str


@dataclass(frozen=True)
class PreparedStructuredOutput:
    """Resolved provider-facing view of one canonical structured request."""

    mode: StructuredOutputMode
    policy: StructuredOutputPolicy
    canonical_schema: dict[str, Any]
    provider_schema: dict[str, Any] | None
    provider_schema_name: str
    provider_strict: bool
    prompt_guidance: bool
    provider_request_fields: dict[str, Any]
    capabilities: StructuredOutputCapabilities
    diagnostics: tuple[SchemaProjectionDiagnostic, ...] = ()


def resolve_structured_output_capabilities(
    provider: str,
    model: str,
) -> StructuredOutputCapabilities:
    """
    Resolve conservative structured-output capabilities for a provider/model.

    Examples:
        Resolve a current OpenAI model:
            ```python
            caps = resolve_structured_output_capabilities("openai", "gpt-5-mini")
            assert caps.native_strict_schema
            ```

        Keep an unknown model conservative:
            ```python
            caps = resolve_structured_output_capabilities("custom", "unknown")
            assert not caps.native_schema
            assert caps.prompt_json
            ```

    Args:
        provider: Configured AG provider name.
        model: Configured model or deployment identifier.

    Returns:
        StructuredOutputCapabilities: Deterministic declared capabilities.

    Notes:
        Unknown provider/model combinations never claim native enforcement.
        Capability discovery is AG infrastructure and has no engine dependency.
    """

    normalized_provider = str(provider or "").lower()
    normalized_model = str(model or "").lower()
    openai_native = _starts_with(
        normalized_model,
        (
            "gpt-4o",
            "gpt-4.1",
            "gpt-5",
            "o1",
            "o3",
            "o4",
        ),
    )
    if normalized_provider == "openai":
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=openai_native,
            native_schema=openai_native,
            json_object=True,
            prompt_json=True,
            source="ag_static/openai_responses_v1",
        )
    if normalized_provider == "azure":
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=openai_native,
            native_schema=openai_native,
            json_object=True,
            prompt_json=True,
            source="ag_static/azure_openai_v1",
        )
    if normalized_provider == "anthropic":
        native = _anthropic_native_model(normalized_model)
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=False,
            native_schema=native,
            json_object=False,
            prompt_json=True,
            source="ag_static/anthropic_messages_v1",
        )
    if normalized_provider == "google":
        native = normalized_model.startswith("gemini-")
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=False,
            native_schema=native,
            json_object=True,
            prompt_json=True,
            source="ag_static/gemini_generate_content_v1",
        )
    if normalized_provider == "deepseek":
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=False,
            native_schema=False,
            json_object=True,
            prompt_json=True,
            source="ag_static/deepseek_json_output_v1",
        )
    if normalized_provider == "openrouter":
        routed_native = _starts_with(
            normalized_model,
            ("openai/", "google/", "anthropic/"),
        )
        routed_strict = normalized_model.startswith("openai/") and _starts_with(
            normalized_model.removeprefix("openai/"),
            ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"),
        )
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=routed_strict,
            native_schema=routed_native,
            json_object=True,
            prompt_json=True,
            source="ag_static/openrouter_compatible_models_v1",
        )
    if normalized_provider == "lmstudio":
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=False,
            native_schema=True,
            json_object=True,
            prompt_json=True,
            source="ag_static/lmstudio_openai_compat_v1",
        )
    if normalized_provider == "ollama":
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=False,
            native_schema=False,
            json_object=True,
            prompt_json=True,
            source="ag_static/ollama_openai_compat_v1",
        )
    return StructuredOutputCapabilities(
        provider=normalized_provider,
        model=model,
        native_strict_schema=False,
        native_schema=False,
        json_object=False,
        prompt_json=True,
        source="ag_static/unknown_conservative_v1",
    )


def prepare_structured_output(
    request: StructuredOutputRequest,
    *,
    provider: str,
    model: str,
    policy: StructuredOutputPolicy = "best_available",
    allow_native_strict: bool = True,
) -> PreparedStructuredOutput:
    """
    Select and prepare the strongest safe structured-output mode.

    Examples:
        Select strict native output for a closed OpenAI schema:
            ```python
            prepared = prepare_structured_output(
                StructuredOutputRequest(
                    "Answer",
                    {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                ),
                provider="openai",
                model="gpt-5-mini",
            )
            assert prepared.mode == "native_strict"
            ```

        Preserve a free-form object by selecting non-strict native schema:
            ```python
            prepared = prepare_structured_output(
                StructuredOutputRequest(
                    "Fields",
                    {"type": "object", "additionalProperties": True},
                ),
                provider="openai",
                model="gpt-5-mini",
            )
            assert prepared.mode == "native_schema"
            ```

    Args:
        request: Canonical provider-neutral schema request.
        provider: Configured AG provider.
        model: Configured model or deployment identifier.
        policy: Profile-owned capability requirement.
        allow_native_strict: Whether a deprecated caller explicitly disabled strict mode.

    Returns:
        PreparedStructuredOutput: Effective mode and provider projection.

    Notes:
        The canonical schema is never weakened or rewritten. Provider
        projections are detached copies, and final validation must always use
        `canonical_schema`.
    """

    if policy not in {"best_available", "native_required"}:
        raise ValueError(f"Unknown structured output policy: {policy}")
    capabilities = resolve_structured_output_capabilities(provider, model)
    canonical = copy.deepcopy(request.schema)
    diagnostics: list[SchemaProjectionDiagnostic] = []

    if allow_native_strict and capabilities.native_strict_schema:
        strict_diagnostics = _openai_strict_diagnostics(canonical)
        if not strict_diagnostics:
            return _prepared(
                request,
                policy=policy,
                mode="native_strict",
                canonical=canonical,
                capabilities=capabilities,
                strict=True,
            )
        diagnostics.extend(strict_diagnostics)

    if capabilities.native_schema:
        native_diagnostics = _native_schema_diagnostics(
            capabilities.provider,
            canonical,
        )
        if not native_diagnostics:
            return _prepared(
                request,
                policy=policy,
                mode="native_schema",
                canonical=canonical,
                capabilities=capabilities,
                strict=False,
                diagnostics=diagnostics,
            )
        diagnostics.extend(native_diagnostics)

    if policy == "native_required":
        detail = "; ".join(item.message for item in diagnostics[:5])
        raise LLMStructuredOutputCapabilityError(
            provider=provider,
            model=model,
            policy=policy,
            detail=detail or "No declared native schema capability.",
        )
    if capabilities.json_object:
        return _prepared(
            request,
            policy=policy,
            mode="json_object",
            canonical=canonical,
            capabilities=capabilities,
            strict=False,
            prompt_guidance=True,
            diagnostics=diagnostics,
        )
    if capabilities.prompt_json:
        return _prepared(
            request,
            policy=policy,
            mode="prompt_json",
            canonical=canonical,
            capabilities=capabilities,
            strict=False,
            prompt_guidance=True,
            diagnostics=diagnostics,
        )
    raise LLMStructuredOutputCapabilityError(
        provider=provider,
        model=model,
        policy=policy,
        detail="No safe structured-output mode is declared.",
    )


def _prepared(
    request: StructuredOutputRequest,
    *,
    policy: StructuredOutputPolicy,
    mode: StructuredOutputMode,
    canonical: dict[str, Any],
    capabilities: StructuredOutputCapabilities,
    strict: bool,
    prompt_guidance: bool = False,
    diagnostics: list[SchemaProjectionDiagnostic] | None = None,
) -> PreparedStructuredOutput:
    return PreparedStructuredOutput(
        mode=mode,
        policy=policy,
        canonical_schema=canonical,
        provider_schema=copy.deepcopy(canonical)
        if mode in {"native_strict", "native_schema"}
        else None,
        provider_schema_name=_provider_schema_name(request.name),
        provider_strict=strict,
        prompt_guidance=prompt_guidance,
        provider_request_fields=_provider_request_fields(
            capabilities.provider,
            mode=mode,
            schema_name=_provider_schema_name(request.name),
            schema=canonical,
            strict=strict,
        ),
        capabilities=capabilities,
        diagnostics=tuple(diagnostics or ()),
    )


def _provider_request_fields(
    provider: str,
    *,
    mode: StructuredOutputMode,
    schema_name: str,
    schema: dict[str, Any],
    strict: bool,
) -> dict[str, Any]:
    if mode == "prompt_json":
        return {}
    if provider == "openai":
        response_format: dict[str, Any] = {"type": "json_object"}
        if mode in {"native_strict", "native_schema"}:
            response_format = {
                "type": "json_schema",
                "name": schema_name,
                "schema": copy.deepcopy(schema),
                "strict": strict,
            }
        return {"text": {"format": response_format}}
    if provider == "anthropic":
        return {
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": copy.deepcopy(schema),
                }
            }
        }
    if provider == "google":
        generation_config: dict[str, Any] = {
            "responseMimeType": "application/json",
        }
        if mode in {"native_strict", "native_schema"}:
            generation_config["responseJsonSchema"] = copy.deepcopy(schema)
        return {"generationConfig": generation_config}
    if provider in {"azure", "openrouter", "lmstudio", "ollama", "deepseek"}:
        response_format: dict[str, Any] = {"type": "json_object"}
        if mode in {"native_strict", "native_schema"}:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": copy.deepcopy(schema),
                    "strict": strict,
                },
            }
        return {"response_format": response_format}
    return {}


def _provider_schema_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return (normalized or "output")[:64]


def _openai_strict_diagnostics(
    schema: dict[str, Any],
) -> list[SchemaProjectionDiagnostic]:
    diagnostics: list[SchemaProjectionDiagnostic] = []
    if schema.get("type") != "object":
        diagnostics.append(
            SchemaProjectionDiagnostic(
                code="strict_root_not_object",
                path="$",
                message="OpenAI strict output requires an object root.",
            )
        )
    _check_strict_node(schema, path="$", diagnostics=diagnostics)
    return diagnostics


def _check_strict_node(
    schema: Any,
    *,
    path: str,
    diagnostics: list[SchemaProjectionDiagnostic],
) -> None:
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            _check_strict_node(item, path=f"{path}[{index}]", diagnostics=diagnostics)
        return
    if not isinstance(schema, dict):
        return
    if schema.get("type") == "object" or "properties" in schema:
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        if schema.get("additionalProperties") is not False:
            diagnostics.append(
                SchemaProjectionDiagnostic(
                    code="strict_object_not_closed",
                    path=path,
                    message=(
                        f"{path} must explicitly set additionalProperties to false "
                        "for OpenAI strict output."
                    ),
                )
            )
        required = schema.get("required")
        if not isinstance(required, list) or set(required) != set(properties):
            diagnostics.append(
                SchemaProjectionDiagnostic(
                    code="strict_properties_not_required",
                    path=path,
                    message=(
                        f"{path} must list every property as required for OpenAI strict output."
                    ),
                )
            )
    for key, value in schema.items():
        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            for child_name, child in value.items():
                _check_strict_node(
                    child,
                    path=f"{path}.{key}.{child_name}",
                    diagnostics=diagnostics,
                )
        elif key in {"items", "anyOf"}:
            _check_strict_node(
                value,
                path=f"{path}.{key}",
                diagnostics=diagnostics,
            )


def _native_schema_diagnostics(
    provider: str,
    schema: dict[str, Any],
) -> list[SchemaProjectionDiagnostic]:
    if provider != "google":
        return []
    unsupported = {
        "allOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
        "patternProperties",
        "unevaluatedProperties",
        "dependentSchemas",
    }
    diagnostics: list[SchemaProjectionDiagnostic] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
            return
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            if key in unsupported:
                diagnostics.append(
                    SchemaProjectionDiagnostic(
                        code="gemini_keyword_unsupported",
                        path=f"{path}.{key}",
                        message=f"Gemini native schema does not declare support for {key}.",
                    )
                )
            visit(child, f"{path}.{key}")

    visit(schema, "$")
    return diagnostics


def _anthropic_native_model(model: str) -> bool:
    return any(
        marker in model
        for marker in (
            "sonnet-4-5",
            "sonnet-4.5",
            "opus-4-1",
            "opus-4.1",
            "opus-4-5",
            "opus-4.5",
            "haiku-4-5",
            "haiku-4.5",
        )
    )


def _starts_with(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value.startswith(prefix) for prefix in prefixes)


__all__ = [
    "PreparedStructuredOutput",
    "SchemaProjectionDiagnostic",
    "StructuredOutputCapabilities",
    "StructuredOutputMode",
    "StructuredOutputPolicy",
    "prepare_structured_output",
    "resolve_structured_output_capabilities",
]

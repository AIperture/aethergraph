"""Provider capability and schema projection for structured LLM output."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Literal

from .catalog import resolve_model_catalog_capability_entry
from .registry import resolve_endpoint_adapter
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
    """Represent one resolved provider-facing structured-output request.

    The value keeps canonical validation truth separate from the exact
    provider request fragment and records stable schema identities.

    Examples:
        Inspect the effective mode:
        ```python
        assert prepared.mode in {"native_strict", "native_schema"}
        ```

        Correlate canonical and projected schemas:
        ```python
        assert len(prepared.canonical_schema_fingerprint) == 64
        ```

    Args:
        mode: Effective provider enforcement mode.
        policy: Requested profile-owned capability policy.
        canonical_schema: Original caller-owned validation schema copy.
        canonical_schema_fingerprint: Stable canonical schema SHA-256.
        provider_schema: Exact projected provider schema, when present.
        provider_schema_fingerprint: Stable projected schema SHA-256, when
            present.
        provider_schema_name: Provider-safe schema name.
        provider_strict: Whether provider strict enforcement is requested.
        prompt_guidance: Whether JSON/schema guidance is added to the prompt.
        provider_request_fields: Exact provider request fragment.
        capabilities: Resolved provider/model capabilities.
        diagnostics: Reasons a stronger enforcement mode was unavailable.

    Returns:
        PreparedStructuredOutput: Immutable resolved request description.

    Notes:
        Fingerprints identify schema content but never replace canonical local
        validation.
    """

    mode: StructuredOutputMode
    policy: StructuredOutputPolicy
    canonical_schema: dict[str, Any]
    canonical_schema_fingerprint: str
    provider_schema: dict[str, Any] | None
    provider_schema_fingerprint: str | None
    provider_schema_name: str
    provider_strict: bool
    prompt_guidance: bool
    provider_request_fields: dict[str, Any]
    capabilities: StructuredOutputCapabilities
    diagnostics: tuple[SchemaProjectionDiagnostic, ...] = ()


def resolve_structured_output_capabilities(
    provider: str,
    model: str,
    *,
    endpoint_id: str | None = None,
) -> StructuredOutputCapabilities:
    """Resolve conservative structured-output capabilities for one model binding.

    Intro:
        Resolution uses the selected endpoint when supplied and otherwise
        preserves the registered provider default for public compatibility.

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

        Resolve an explicitly selected endpoint:
            ```python
            caps = resolve_structured_output_capabilities(
                "openai",
                "gpt-5-mini",
                endpoint_id="openai_responses",
            )
            assert caps.native_schema
            ```

    Args:
        provider: Configured AG provider name.
        model: Configured model or deployment identifier.
        endpoint_id: Optional endpoint adapter selected before request inspection.

    Returns:
        StructuredOutputCapabilities: Deterministic declared capabilities.

    Notes:
        Unknown provider/model combinations never claim native enforcement.
        Capability discovery is AG infrastructure and has no engine dependency.
    """

    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    try:
        endpoint = resolve_endpoint_adapter(
            normalized_provider,
            "chat",
            endpoint_id=endpoint_id,
        )
        entry = resolve_model_catalog_capability_entry(
            normalized_provider,
            normalized_model,
            "chat",
            endpoint.adapter_id,
            capability="structured_output",
        )
    except (KeyError, ValueError):
        if endpoint_id is not None:
            raise
        entry = None
    if entry is not None and entry.structured_output is not None:
        facts = entry.structured_output
        return StructuredOutputCapabilities(
            provider=normalized_provider,
            model=model,
            native_strict_schema=facts.native_strict_schema,
            native_schema=facts.native_schema,
            json_object=facts.json_object,
            prompt_json=facts.prompt_json,
            source=facts.capability_source,
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
    endpoint_id: str | None = None,
) -> PreparedStructuredOutput:
    """Select and prepare the strongest safe structured-output mode.

    Intro:
        Preparation resolves capability facts for one preselected endpoint,
        retains the canonical schema, and emits one deterministic projection.

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
        endpoint_id: Optional endpoint adapter selected before request inspection.

    Returns:
        PreparedStructuredOutput: Effective mode and provider projection.

    Notes:
        The canonical schema is never weakened or rewritten. Provider
        projections are detached copies, and final validation must always use
        `canonical_schema`.
    """

    if policy not in {"best_available", "native_required"}:
        raise ValueError(f"Unknown structured output policy: {policy}")
    capabilities = resolve_structured_output_capabilities(
        provider,
        model,
        endpoint_id=endpoint_id,
    )
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
    provider_schema = (
        copy.deepcopy(canonical) if mode in {"native_strict", "native_schema"} else None
    )
    return PreparedStructuredOutput(
        mode=mode,
        policy=policy,
        canonical_schema=canonical,
        canonical_schema_fingerprint=_schema_fingerprint(canonical),
        provider_schema=provider_schema,
        provider_schema_fingerprint=(
            _schema_fingerprint(provider_schema) if provider_schema is not None else None
        ),
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


def _schema_fingerprint(schema: dict[str, Any]) -> str:
    body = json.dumps(
        schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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


__all__ = [
    "PreparedStructuredOutput",
    "SchemaProjectionDiagnostic",
    "StructuredOutputCapabilities",
    "StructuredOutputMode",
    "StructuredOutputPolicy",
    "prepare_structured_output",
    "resolve_structured_output_capabilities",
]

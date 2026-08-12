"""Deterministic whole-request compatibility validation for canonical generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .contracts import ModelRequest
from .registry import EndpointAdapterDescriptor
from .types import ImageInput, LLMError, StructuredOutputRequest

RequestFeature = Literal[
    "image_input",
    "native_tool_calling",
    "native_tool_search",
    "parallel_tool_calls",
    "structured_output",
    "tool_result_continuation",
]
RequestModelCapability = Literal[
    "image_input",
    "native_tool_calling",
    "tool_result_continuation",
    "parallel_tool_calls",
    "structured_output",
    "prompt_cache",
    "native_tool_search_hosted",
    "native_tool_search_client",
]


@dataclass(frozen=True)
class RequestCompatibilityDiagnostic:
    """Describe one deterministic incompatibility in a complete model request."""

    code: str
    features: tuple[RequestFeature, ...]
    message: str


@dataclass(frozen=True)
class RequestCompatibilityReport:
    """Carry required adapter features and all ordered request diagnostics."""

    required_adapter_capabilities: tuple[str, ...]
    required_model_capabilities: tuple[RequestModelCapability, ...]
    diagnostics: tuple[RequestCompatibilityDiagnostic, ...]
    valid: bool


class LLMRequestCompatibilityError(LLMError):
    """Report that a complete canonical request cannot be represented safely."""

    def __init__(self, report: RequestCompatibilityReport) -> None:
        """Initialize one whole-request preflight failure.

        Intro:
            The error retains the complete deterministic report and summarizes
            its first diagnostic without attempting a reduced-feature request.

        Examples:
            Inspect the first diagnostic:
            ```python
            diagnostic = RequestCompatibilityDiagnostic(
                code="invalid_combination",
                features=("native_tool_calling",),
                message="request cannot be represented",
            )
            report = RequestCompatibilityReport(
                required_adapter_capabilities=("native_tools",),
                required_model_capabilities=("native_tool_calling",),
                diagnostics=(diagnostic,),
                valid=False,
            )
            error = LLMRequestCompatibilityError(report)
            assert error.report.diagnostics
            ```

            Inspect the stable summary:
            ```python
            diagnostic = RequestCompatibilityDiagnostic(
                code="invalid_combination",
                features=("structured_output",),
                message="structured output cannot be represented",
            )
            report = RequestCompatibilityReport((), (), (diagnostic,), False)
            error = LLMRequestCompatibilityError(report)
            assert "cannot" in str(error).lower()
            ```

        Args:
            report: Invalid whole-request compatibility report.

        Returns:
            None.

        Notes:
            This error occurs before estimation, quota reservation, and provider
            dispatch. No compatibility retry is performed.
        """
        if report.valid or not report.diagnostics:
            raise ValueError("request compatibility error requires an invalid report")
        first = report.diagnostics[0]
        suffix = (
            ""
            if len(report.diagnostics) == 1
            else f" ({len(report.diagnostics) - 1} additional diagnostic(s))"
        )
        super().__init__(f"Model request cannot be represented: {first.message}{suffix}")
        self.report = report


def validate_model_request(
    request: ModelRequest,
    *,
    adapter: EndpointAdapterDescriptor | None = None,
) -> RequestCompatibilityReport:
    """Validate one complete canonical request before runtime preparation.

    Intro:
        Validation derives the exact adapter features required by the request,
        rejects unsafe cross-feature combinations, and optionally clamps those
        requirements to one already-selected endpoint adapter.

        Examples:
            Validate a direct completion:
            ```python
            request = ModelRequest(
                messages=(ChatMessage("user", (TextPart("Hello"),)),),
            )
            report = validate_model_request(request)
            assert report.valid
            ```

            Check an already-selected adapter:
            ```python
            tool = ToolDefinition(
                name="lookup",
                description="Look up a value.",
                input_schema={"type": "object", "properties": {}},
            )
            request = ModelRequest(
                messages=(ChatMessage("user", (TextPart("Look up"),)),),
                tools=(tool,),
                tool_choice="auto",
            )
            adapter = get_endpoint_adapter("openai_responses")
            report = validate_model_request(request, adapter=adapter)
            assert "native_tools" in report.required_adapter_capabilities
            ```

    Args:
        request: Immutable canonical generation request.
        adapter: Optional endpoint adapter selected before request inspection.

    Returns:
        RequestCompatibilityReport: Ordered requirements and deterministic
        diagnostics without side effects.

    Notes:
        Catalog model facts are resolved separately. This validator never selects
        or substitutes an adapter based on request features.
    """
    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")

    diagnostics: list[RequestCompatibilityDiagnostic] = []
    required_adapter: list[str] = []
    required_model: list[RequestModelCapability] = []
    has_tools = bool(request.tools)
    has_structured_output = (
        isinstance(
            request.response_format,
            StructuredOutputRequest,
        )
        or request.response_format == "json_object"
    )
    has_images = any(
        isinstance(part, ImageInput) for message in request.messages for part in message.content
    )
    has_continuation = request.continuation is not None or bool(request.tool_outputs)

    if has_images:
        required_adapter.append("image_input")
        required_model.append("image_input")
    if has_tools:
        required_adapter.append("native_tools")
        required_model.append("native_tool_calling")
        if request.max_tool_calls > 1:
            required_model.append("parallel_tool_calls")
    if has_continuation:
        required_model.append("tool_result_continuation")
    if request.native_tool_search is not None:
        required_adapter.append("native_tool_search")
        required_model.append(
            "native_tool_search_hosted"
            if request.native_tool_search.mode == "native_hosted"
            else "native_tool_search_client"
        )
    if has_structured_output:
        required_adapter.append("structured_output")
        required_model.append("structured_output")
    if request.prompt_cache is not None:
        required_model.append("prompt_cache")

    if has_tools and has_structured_output:
        diagnostics.append(
            RequestCompatibilityDiagnostic(
                code="structured_output_with_native_tools",
                features=("structured_output", "native_tool_calling"),
                message="structured output cannot be combined with native Tool calling",
            )
        )
    if has_continuation and not has_tools:
        diagnostics.append(
            RequestCompatibilityDiagnostic(
                code="tool_continuation_without_tools",
                features=("tool_result_continuation", "native_tool_calling"),
                message="Tool-result continuation requires the request Tool catalog",
            )
        )

    if adapter is not None:
        implemented = set(adapter.implementation_capabilities)
        for capability in required_adapter:
            if capability not in implemented:
                diagnostics.append(
                    RequestCompatibilityDiagnostic(
                        code="adapter_capability_unimplemented",
                        features=(_feature_for_adapter_capability(capability),),
                        message=(
                            f"endpoint adapter {adapter.adapter_id!r} does not implement "
                            f"required capability {capability!r}"
                        ),
                    )
                )

    ordered_adapter_requirements = tuple(dict.fromkeys(required_adapter))
    ordered_model_requirements = tuple(dict.fromkeys(required_model))
    ordered_diagnostics = tuple(diagnostics)
    return RequestCompatibilityReport(
        required_adapter_capabilities=ordered_adapter_requirements,
        required_model_capabilities=ordered_model_requirements,
        diagnostics=ordered_diagnostics,
        valid=not ordered_diagnostics,
    )


def _feature_for_adapter_capability(capability: str) -> RequestFeature:
    return {
        "image_input": "image_input",
        "native_tools": "native_tool_calling",
        "native_tool_search": "native_tool_search",
        "structured_output": "structured_output",
    }[capability]


__all__ = [
    "LLMRequestCompatibilityError",
    "RequestCompatibilityDiagnostic",
    "RequestCompatibilityReport",
    "RequestFeature",
    "RequestModelCapability",
    "validate_model_request",
]

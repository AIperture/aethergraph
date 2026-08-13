"""Fail-closed model and adapter capability resolution with provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .catalog import (
    catalog_digest,
    resolve_model_catalog_capability_entry,
    resolve_model_catalog_entry,
)
from .contracts import ModelRequest
from .profiles import (
    CapabilityState,
    ChatProfile,
    EmbeddingProfileSpec,
    ImageGenerationProfile,
)
from .registry import get_endpoint_adapter, get_provider_descriptor
from .request_validation import RequestCompatibilityReport, validate_model_request

ChatCapabilityName = Literal[
    "image_input",
    "streaming",
    "native_tool_calling",
    "tool_result_continuation",
    "parallel_tool_calls",
    "structured_output",
    "prompt_cache",
    "native_tool_search_hosted",
    "native_tool_search_client",
]
EmbeddingCapabilityName = Literal["text_embeddings", "dimensions"]
ImageGenerationCapabilityName = Literal["text_to_image", "image_editing", "multiple_outputs"]
ModelCapabilityName = ChatCapabilityName | EmbeddingCapabilityName | ImageGenerationCapabilityName


class CapabilityContract(BaseModel):
    """Base class for closed immutable resolved-capability records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CapabilityEvidence(CapabilityContract):
    """One ordered reason contributing to an effective capability state."""

    source: Literal["catalog", "override", "adapter", "unknown"]
    reference: str
    state: CapabilityState


class EffectiveCapability(CapabilityContract):
    """One effective capability state with complete ordered provenance."""

    state: CapabilityState
    provenance: tuple[CapabilityEvidence, ...]


class CapabilityDiagnostic(CapabilityContract):
    """One deterministic capability-resolution or requirement diagnostic."""

    code: str
    capability: ModelCapabilityName
    message: str


class ResolvedChatCapabilities(CapabilityContract):
    """Effective Chat capabilities for one exact model and adapter binding."""

    image_input: EffectiveCapability
    streaming: EffectiveCapability
    native_tool_calling: EffectiveCapability
    tool_result_continuation: EffectiveCapability
    parallel_tool_calls: EffectiveCapability
    structured_output: EffectiveCapability
    prompt_cache: EffectiveCapability
    native_tool_search_hosted: EffectiveCapability
    native_tool_search_client: EffectiveCapability


class ResolvedEmbeddingCapabilities(CapabilityContract):
    """Effective Embedding capabilities for one exact model and adapter binding."""

    text_embeddings: EffectiveCapability
    dimensions: EffectiveCapability


class ResolvedImageGenerationCapabilities(CapabilityContract):
    """Effective Image Generation capabilities for one exact binding."""

    text_to_image: EffectiveCapability
    image_editing: EffectiveCapability
    multiple_outputs: EffectiveCapability


class ResolvedOperationBinding(CapabilityContract):
    """Common pinned identity and diagnostics for one exact model operation."""

    operation: Literal["chat", "embeddings", "image_generation"]
    provider_id: str
    endpoint_id: str
    model_id: str
    catalog_key: str | None
    catalog_keys: tuple[str, ...] = ()
    catalog_digest: str
    diagnostics: tuple[CapabilityDiagnostic, ...] = ()

    @property
    def valid(self) -> bool:
        """Return whether all requested capabilities passed preflight.

        Intro:
            A resolved binding is valid only when no fail-closed requirement
            diagnostic was produced.

        Examples:
            Check a valid binding:
                ```python
                assert binding.valid is True
                ```

            Check a rejected binding:
                ```python
                if not binding.valid:
                    assert binding.diagnostics
                ```

        Args:
            None.

        Returns:
            bool: `True` when the binding has no diagnostics.

        Notes:
            Resolution itself is side-effect free and never calls a provider.
        """

        return not self.diagnostics


class ResolvedModelBinding(ResolvedOperationBinding):
    """Pinned model binding, effective Chat facts, and preflight diagnostics."""

    operation: Literal["chat"] = "chat"
    capabilities: ResolvedChatCapabilities


class ResolvedEmbeddingBinding(ResolvedOperationBinding):
    """Pinned embedding binding, effective facts, and preflight diagnostics."""

    operation: Literal["embeddings"] = "embeddings"
    capabilities: ResolvedEmbeddingCapabilities


class ResolvedImageGenerationBinding(ResolvedOperationBinding):
    """Pinned image binding, effective facts, and preflight diagnostics."""

    operation: Literal["image_generation"] = "image_generation"
    capabilities: ResolvedImageGenerationCapabilities


@dataclass(frozen=True)
class ResolvedModelRequest:
    """Carry one pinned model binding and complete request-validation report."""

    binding: ResolvedModelBinding
    compatibility: RequestCompatibilityReport
    valid: bool


_CAPABILITY_NAMES: tuple[ChatCapabilityName, ...] = (
    "image_input",
    "streaming",
    "native_tool_calling",
    "tool_result_continuation",
    "parallel_tool_calls",
    "structured_output",
    "prompt_cache",
    "native_tool_search_hosted",
    "native_tool_search_client",
)

_ADAPTER_FLAGS: dict[ChatCapabilityName, str] = {
    "image_input": "image_input",
    "streaming": "streaming",
    "native_tool_calling": "native_tools",
    "tool_result_continuation": "native_tools",
    "parallel_tool_calls": "native_tools",
    "structured_output": "structured_output",
    "native_tool_search_hosted": "native_tool_search",
    "native_tool_search_client": "native_tool_search",
}

_EMBEDDING_CAPABILITY_NAMES: tuple[EmbeddingCapabilityName, ...] = (
    "text_embeddings",
    "dimensions",
)
_IMAGE_GENERATION_CAPABILITY_NAMES: tuple[ImageGenerationCapabilityName, ...] = (
    "text_to_image",
    "image_editing",
    "multiple_outputs",
)


def _resolve_operation_capabilities(
    profile: EmbeddingProfileSpec | ImageGenerationProfile,
    *,
    operation: Literal["embeddings", "image_generation"],
    capability_names: tuple[EmbeddingCapabilityName, ...]
    | tuple[ImageGenerationCapabilityName, ...],
    required: tuple[EmbeddingCapabilityName, ...] | tuple[ImageGenerationCapabilityName, ...],
) -> tuple[
    dict[str, EffectiveCapability],
    tuple[CapabilityDiagnostic, ...],
    str | None,
]:
    if len(required) != len(set(required)):
        raise ValueError(f"required {operation} capabilities must be unique")
    if any(name not in capability_names for name in required):
        raise ValueError(f"required capability does not belong to {operation}")
    adapter = get_endpoint_adapter(profile.connection.endpoint_id)
    provider = get_provider_descriptor(profile.connection.provider_id)
    if profile.connection.endpoint_id not in provider.endpoint_ids:
        raise ValueError(f"{operation} profile endpoint is not registered for its provider")
    if operation not in adapter.implemented_operations:
        raise ValueError(f"profile endpoint does not implement {operation}")
    entry = resolve_model_catalog_capability_entry(
        profile.connection.provider_id,
        profile.model.model_id,
        operation,
        profile.connection.endpoint_id,
        capability=operation,
    )
    catalog_capabilities = getattr(entry, operation, None) if entry is not None else None
    overrides = profile.capability_overrides.model_dump()
    effective: dict[str, EffectiveCapability] = {}
    for name in capability_names:
        state: CapabilityState = (
            getattr(catalog_capabilities, name) if catalog_capabilities is not None else "unknown"
        )
        evidence = [
            CapabilityEvidence(
                source="catalog" if entry is not None else "unknown",
                reference=entry.catalog_key if entry is not None else "no_catalog_match",
                state=state,
            )
        ]
        override = overrides[name]
        if override != "unknown":
            state = override
            evidence.append(
                CapabilityEvidence(
                    source="override",
                    reference=f"profile.capability_overrides.{name}",
                    state=override,
                )
            )
        if name not in adapter.implementation_capabilities:
            state = "unsupported"
            evidence.append(
                CapabilityEvidence(
                    source="adapter",
                    reference=f"{adapter.adapter_id}:{name}:unimplemented",
                    state="unsupported",
                )
            )
        effective[name] = EffectiveCapability(state=state, provenance=tuple(evidence))
    diagnostics = tuple(
        CapabilityDiagnostic(
            code=(
                "required_capability_unknown"
                if effective[name].state == "unknown"
                else "required_capability_unsupported"
            ),
            capability=name,
            message=(
                f"Required {operation} capability {name!r} resolved to {effective[name].state!r}."
            ),
        )
        for name in required
        if effective[name].state != "supported"
    )
    return effective, diagnostics, entry.catalog_key if entry is not None else None


def resolve_chat_profile(
    profile: ChatProfile,
    *,
    required: tuple[ChatCapabilityName, ...] = (),
) -> ResolvedModelBinding:
    """Resolve effective Chat capabilities for one canonical profile.

    Intro:
        Resolution combines catalog facts, explicit model overrides, and the
        selected adapter implementation. Adapter limitations always clamp a
        positive catalog or override assertion. Unknown never satisfies a
        required capability.

    Examples:
        Resolve ordinary Chat without advanced requirements:
            ```python
            binding = resolve_chat_profile(profile)
            assert binding.provider_id == profile.connection.provider_id
            ```

        Require native client Tool search:
            ```python
            binding = resolve_chat_profile(
                profile,
                required=("native_tool_search_client",),
            )
            ```

    Args:
        profile: Canonical immutable Chat profile.
        required: Capabilities that must resolve to `supported`.

    Returns:
        ResolvedModelBinding: Pinned binding, effective facts, provenance, and
        deterministic fail-closed diagnostics.

    Notes:
        Engine-projected Tool search is not a model capability and is therefore
        absent from both `required` and the resolved capability record.
    """

    if not isinstance(profile, ChatProfile):
        raise TypeError("Chat capability resolution requires ChatProfile")
    if len(required) != len(set(required)):
        raise ValueError("required Chat capabilities must be unique")
    adapter = get_endpoint_adapter(profile.connection.endpoint_id)
    provider = get_provider_descriptor(profile.connection.provider_id)
    if profile.connection.endpoint_id not in provider.endpoint_ids:
        raise ValueError("Chat profile endpoint is not registered for its provider")
    if "chat" not in adapter.implemented_operations:
        raise ValueError("Chat profile endpoint does not implement Chat")
    native_entry = resolve_model_catalog_entry(
        profile.connection.provider_id,
        profile.model.model_id,
        "chat",
        profile.connection.endpoint_id,
    )
    chat_tools_entry = resolve_model_catalog_capability_entry(
        profile.connection.provider_id,
        profile.model.model_id,
        "chat",
        profile.connection.endpoint_id,
        capability="chat_tools",
    )
    structured_entry = resolve_model_catalog_capability_entry(
        profile.connection.provider_id,
        profile.model.model_id,
        "chat",
        profile.connection.endpoint_id,
        capability="structured_output",
    )
    prompt_cache_entry = resolve_model_catalog_capability_entry(
        profile.connection.provider_id,
        profile.model.model_id,
        "chat",
        profile.connection.endpoint_id,
        capability="prompt_cache",
    )
    catalog_states: dict[ChatCapabilityName, CapabilityState] = {
        name: "unknown" for name in _CAPABILITY_NAMES
    }
    if native_entry is not None:
        native_modes = {item.mode for item in native_entry.native_tool_search}
        catalog_states["native_tool_search_hosted"] = (
            "supported" if "native_hosted" in native_modes else "unsupported"
        )
        catalog_states["native_tool_search_client"] = (
            "supported" if "native_client" in native_modes else "unsupported"
        )
    if chat_tools_entry is not None and chat_tools_entry.chat_tools is not None:
        chat_tools = chat_tools_entry.chat_tools
        catalog_states["native_tool_calling"] = chat_tools.native_tool_calling
        catalog_states["tool_result_continuation"] = chat_tools.tool_result_continuation
        catalog_states["parallel_tool_calls"] = chat_tools.parallel_tool_calls
    if structured_entry is not None and structured_entry.structured_output is not None:
        structured = structured_entry.structured_output
        catalog_states["structured_output"] = (
            "supported" if structured.native_schema or structured.json_object else "unsupported"
        )
    if prompt_cache_entry is not None and prompt_cache_entry.prompt_cache is not None:
        catalog_states["prompt_cache"] = (
            "supported" if prompt_cache_entry.prompt_cache.mode != "unavailable" else "unsupported"
        )
    capability_entries = {
        "native_tool_calling": chat_tools_entry,
        "tool_result_continuation": chat_tools_entry,
        "parallel_tool_calls": chat_tools_entry,
        "structured_output": structured_entry,
        "prompt_cache": prompt_cache_entry,
        "native_tool_search_hosted": native_entry,
        "native_tool_search_client": native_entry,
    }
    override_values = profile.capability_overrides.model_dump()
    effective: dict[str, EffectiveCapability] = {}
    for name in _CAPABILITY_NAMES:
        state = catalog_states[name]
        entry = capability_entries.get(name)
        evidence = [
            CapabilityEvidence(
                source="catalog" if entry is not None else "unknown",
                reference=entry.catalog_key if entry is not None else "no_catalog_match",
                state=state,
            )
        ]
        override = override_values[name]
        if override != "unknown":
            state = override
            evidence.append(
                CapabilityEvidence(
                    source="override",
                    reference=f"profile.capability_overrides.{name}",
                    state=override,
                )
            )
        adapter_flag = _ADAPTER_FLAGS.get(name)
        if adapter_flag is not None and adapter_flag not in adapter.implementation_capabilities:
            state = "unsupported"
            evidence.append(
                CapabilityEvidence(
                    source="adapter",
                    reference=f"{adapter.adapter_id}:{adapter_flag}:unimplemented",
                    state="unsupported",
                )
            )
        effective[name] = EffectiveCapability(state=state, provenance=tuple(evidence))
    capabilities = ResolvedChatCapabilities(**effective)
    diagnostics = tuple(
        CapabilityDiagnostic(
            code=(
                "required_capability_unknown"
                if getattr(capabilities, name).state == "unknown"
                else "required_capability_unsupported"
            ),
            capability=name,
            message=(
                f"Required Chat capability {name!r} resolved to "
                f"{getattr(capabilities, name).state!r}."
            ),
        )
        for name in required
        if getattr(capabilities, name).state != "supported"
    )
    catalog_entries = tuple(
        dict.fromkeys(
            item.catalog_key
            for item in (native_entry, chat_tools_entry, structured_entry, prompt_cache_entry)
            if item is not None
        )
    )
    return ResolvedModelBinding(
        provider_id=profile.connection.provider_id,
        endpoint_id=profile.connection.endpoint_id,
        model_id=profile.model.model_id,
        catalog_key=catalog_entries[0] if catalog_entries else None,
        catalog_keys=catalog_entries,
        catalog_digest=catalog_digest(),
        capabilities=capabilities,
        diagnostics=diagnostics,
    )


def resolve_embedding_profile(
    profile: EmbeddingProfileSpec,
    *,
    required: tuple[EmbeddingCapabilityName, ...] = (),
) -> ResolvedEmbeddingBinding:
    """Resolve effective Embedding capabilities for one canonical profile.

    Intro:
        Combines one operation-scoped catalog record, explicit profile
        overrides, and the exact selected endpoint implementation. Unknown
        model facts remain unknown, while an unimplemented adapter feature
        always clamps a positive catalog or override assertion.

    Examples:
        Resolve a cataloged embedding model:
            ```python
            binding = resolve_embedding_profile(profile)
            assert binding.operation == "embeddings"
            ```

        Require configurable dimensions:
            ```python
            binding = resolve_embedding_profile(
                profile,
                required=("dimensions",),
            )
            ```

    Args:
        profile: Canonical immutable Embedding profile.
        required: Embedding capabilities that must resolve to `supported`.

    Returns:
        ResolvedEmbeddingBinding: Pinned identity, effective capability
        provenance, catalog identity, and fail-closed diagnostics.

    Notes:
        Resolution performs no provider I/O and never infers facts from a
        provider name or discovered model list.
    """

    if not isinstance(profile, EmbeddingProfileSpec):
        raise TypeError("Embedding capability resolution requires EmbeddingProfileSpec")
    effective, diagnostics, catalog_key = _resolve_operation_capabilities(
        profile,
        operation="embeddings",
        capability_names=_EMBEDDING_CAPABILITY_NAMES,
        required=required,
    )
    return ResolvedEmbeddingBinding(
        provider_id=profile.connection.provider_id,
        endpoint_id=profile.connection.endpoint_id,
        model_id=profile.model.model_id,
        catalog_key=catalog_key,
        catalog_keys=(catalog_key,) if catalog_key is not None else (),
        catalog_digest=catalog_digest(),
        capabilities=ResolvedEmbeddingCapabilities(**effective),
        diagnostics=diagnostics,
    )


def resolve_image_generation_profile(
    profile: ImageGenerationProfile,
    *,
    required: tuple[ImageGenerationCapabilityName, ...] = (),
) -> ResolvedImageGenerationBinding:
    """Resolve effective Image Generation capabilities for one profile.

    Intro:
        Combines image-model catalog facts, explicit profile overrides, and the
        exact selected endpoint implementation without changing the requested
        provider, model, or adapter.

    Examples:
        Resolve a cataloged image model:
            ```python
            binding = resolve_image_generation_profile(profile)
            assert binding.operation == "image_generation"
            ```

        Require image-conditioned editing:
            ```python
            binding = resolve_image_generation_profile(
                profile,
                required=("image_editing",),
            )
            ```

    Args:
        profile: Canonical immutable Image Generation profile.
        required: Image capabilities that must resolve to `supported`.

    Returns:
        ResolvedImageGenerationBinding: Pinned identity, effective capability
        provenance, catalog identity, and fail-closed diagnostics.

    Notes:
        Adapter clamping exposes incomplete request projection as unsupported;
        it does not retry through another endpoint.
    """

    if not isinstance(profile, ImageGenerationProfile):
        raise TypeError("Image capability resolution requires ImageGenerationProfile")
    effective, diagnostics, catalog_key = _resolve_operation_capabilities(
        profile,
        operation="image_generation",
        capability_names=_IMAGE_GENERATION_CAPABILITY_NAMES,
        required=required,
    )
    return ResolvedImageGenerationBinding(
        provider_id=profile.connection.provider_id,
        endpoint_id=profile.connection.endpoint_id,
        model_id=profile.model.model_id,
        catalog_key=catalog_key,
        catalog_keys=(catalog_key,) if catalog_key is not None else (),
        catalog_digest=catalog_digest(),
        capabilities=ResolvedImageGenerationCapabilities(**effective),
        diagnostics=diagnostics,
    )


def resolve_model_request(
    profile: ChatProfile,
    request: ModelRequest,
) -> ResolvedModelRequest:
    """Resolve one canonical profile and complete generation request together.

    Intro:
        Resolution pins the configured endpoint before inspecting request
        features, validates cross-feature and adapter combinations, and then
        applies fail-closed model capability requirements with full provenance.

    Examples:
        Resolve a direct completion:
        ```python
        profile = ChatProfile(
            connection=ProviderConnection(
                provider_id="openai",
                endpoint_id="openai_responses",
            ),
            model=ModelSelection(model_id="gpt-5.6"),
        )
        request = ModelRequest(
            messages=(ChatMessage("user", (TextPart("Hello"),)),),
        )
        resolved = resolve_model_request(profile, request)
        assert resolved.binding.endpoint_id == profile.connection.endpoint_id
        ```

        Inspect a rejected Tool request:
        ```python
        tool = ToolDefinition(
            name="lookup",
            description="Look up one value.",
            input_schema={"type": "object", "properties": {}},
        )
        request = ModelRequest(
            messages=(ChatMessage("user", (TextPart("Look up"),)),),
            tools=(tool,),
            tool_choice="auto",
        )
        resolved = resolve_model_request(profile, request)
        if not resolved.valid:
            assert resolved.compatibility.diagnostics or resolved.binding.diagnostics
        ```

    Args:
        profile: Canonical immutable Chat profile with a preselected endpoint.
        request: Complete immutable canonical generation request.

    Returns:
        ResolvedModelRequest: Pinned binding, combination report, and aggregate
        validity without constructing a runtime client.

    Notes:
        Unknown model capabilities fail required checks. Resolution never changes
        provider or endpoint according to request features.
    """
    if not isinstance(profile, ChatProfile):
        raise TypeError("model request resolution requires a ChatProfile")
    if not isinstance(request, ModelRequest):
        raise TypeError("model request resolution requires a ModelRequest")
    adapter = get_endpoint_adapter(profile.connection.endpoint_id)
    compatibility = validate_model_request(request, adapter=adapter)
    required = tuple(compatibility.required_model_capabilities)
    binding = resolve_chat_profile(
        profile,
        required=required,  # type: ignore[arg-type]
    )
    return ResolvedModelRequest(
        binding=binding,
        compatibility=compatibility,
        valid=compatibility.valid and binding.valid,
    )


__all__ = [
    "CapabilityDiagnostic",
    "CapabilityEvidence",
    "ChatCapabilityName",
    "EmbeddingCapabilityName",
    "EffectiveCapability",
    "ImageGenerationCapabilityName",
    "ModelCapabilityName",
    "ResolvedChatCapabilities",
    "ResolvedEmbeddingBinding",
    "ResolvedEmbeddingCapabilities",
    "ResolvedImageGenerationBinding",
    "ResolvedImageGenerationCapabilities",
    "ResolvedModelBinding",
    "ResolvedModelRequest",
    "ResolvedOperationBinding",
    "resolve_chat_profile",
    "resolve_embedding_profile",
    "resolve_image_generation_profile",
    "resolve_model_request",
]

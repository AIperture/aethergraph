"""Versioned API schemas for model registry, catalog, and binding resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aethergraph.services.llm.capabilities import (
    ChatCapabilityName,
    EmbeddingCapabilityName,
    ImageGenerationCapabilityName,
    ResolvedEmbeddingBinding,
    ResolvedImageGenerationBinding,
    ResolvedModelBinding,
)
from aethergraph.services.llm.catalog import ModelCatalogEntry
from aethergraph.services.llm.profiles import (
    ChatCapabilityOverrides,
    EmbeddingCapabilityOverrides,
    ImageGenerationCapabilityOverrides,
)
from aethergraph.services.llm.registry import ModelOperation


class LLMApiContract(BaseModel):
    """Reject undeclared fields across the public model-metadata API."""

    model_config = ConfigDict(extra="forbid")


def _require_unique_capabilities(
    capabilities: tuple[str, ...],
    *,
    operation: str,
) -> None:
    """Reject duplicate ordered capability requirements.

    Intro:
        All operation-specific resolution requests share one deterministic
        uniqueness rule while retaining their narrower public literal types.

    Examples:
        Accept distinct requirements:
            ```python
            _require_unique_capabilities(("streaming", "prompt_cache"), operation="Chat")
            ```

        Reject a duplicate requirement:
            ```python
            _require_unique_capabilities(("dimensions", "dimensions"), operation="Embedding")
            ```

    Args:
        capabilities: Ordered requested capability names.
        operation: User-facing operation label for validation errors.

    Returns:
        None: The requirements are unique.

    Notes:
        Literal membership is enforced separately by each Pydantic field.
    """

    if len(capabilities) != len(set(capabilities)):
        raise ValueError(f"required {operation} capabilities must be unique")


class LLMEndpointAdapterView(LLMApiContract):
    """Describe one provider-owned selectable endpoint adapter."""

    adapter_id: str
    adapter_revision: int = Field(ge=1)
    protocol_family: str
    implemented_operations: tuple[ModelOperation, ...]
    implementation_capabilities: tuple[str, ...] = ()


class LLMProviderView(LLMApiContract):
    """Describe one registered provider and its operation defaults."""

    provider_id: str
    display_name: str
    studio_visible: bool
    default_endpoints: dict[ModelOperation, str]
    default_base_url: str | None
    base_url_env: str | None
    credential_envs: tuple[str, ...]
    model_discovery_adapter_id: str | None
    endpoints: tuple[LLMEndpointAdapterView, ...]


class LLMRegistryResponse(LLMApiContract):
    """Return the complete deterministic provider and endpoint registry view."""

    schema_version: Literal["aethergraph.llm-registry/v1"] = "aethergraph.llm-registry/v1"
    providers: tuple[LLMProviderView, ...]


class LLMModelCatalogResponse(LLMApiContract):
    """Return the validated production catalog with its identity digest."""

    schema_version: Literal["aethergraph.model-catalog-api/v1"] = "aethergraph.model-catalog-api/v1"
    catalog_schema_version: Literal["aethergraph.model-catalog/v1"]
    catalog_revision: int = Field(ge=1)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[ModelCatalogEntry, ...]


class LLMChatResolveRequest(LLMApiContract):
    """Request capability resolution for one explicit Chat binding."""

    schema_version: Literal["aethergraph.chat-resolve-request/v1"] = (
        "aethergraph.chat-resolve-request/v1"
    )
    provider_id: str = Field(min_length=1, max_length=128)
    endpoint_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    required_capabilities: tuple[ChatCapabilityName, ...] = ()
    capability_overrides: ChatCapabilityOverrides = Field(default_factory=ChatCapabilityOverrides)

    @model_validator(mode="after")
    def validate_required_capabilities(self) -> LLMChatResolveRequest:
        """Require deterministic unique capability requirements.

        Intro:
            Duplicate requirements would produce redundant diagnostics and an
            unstable request identity, so the API rejects them before resolution.

        Examples:
            Validate one requirement:
                ```python
                request = LLMChatResolveRequest(
                    provider_id="openai",
                    endpoint_id="openai_responses",
                    model_id="gpt-5.6",
                    required_capabilities=("native_tool_calling",),
                )
                ```

            Reject a duplicate requirement:
                ```python
                try:
                    LLMChatResolveRequest(
                        provider_id="openai",
                        endpoint_id="openai_responses",
                        model_id="gpt-5.6",
                        required_capabilities=("streaming", "streaming"),
                    )
                except ValueError:
                    pass
                ```

        Args:
            self: Fully parsed Chat binding resolution request.

        Returns:
            LLMChatResolveRequest: Unchanged request with unique requirements.

        Notes:
            Requirement order remains caller-authored and appears in diagnostic
            order when more than one capability fails.
        """

        _require_unique_capabilities(self.required_capabilities, operation="Chat")
        return self


class LLMChatResolveResponse(LLMApiContract):
    """Return one pinned Chat binding and its capability diagnostics."""

    schema_version: Literal["aethergraph.chat-resolve-response/v1"] = (
        "aethergraph.chat-resolve-response/v1"
    )
    valid: bool
    binding: ResolvedModelBinding


class LLMEmbeddingResolveRequest(LLMApiContract):
    """Request capability resolution for one explicit Embedding binding."""

    schema_version: Literal["aethergraph.embedding-resolve-request/v1"] = (
        "aethergraph.embedding-resolve-request/v1"
    )
    provider_id: str = Field(min_length=1, max_length=128)
    endpoint_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    required_capabilities: tuple[EmbeddingCapabilityName, ...] = ()
    capability_overrides: EmbeddingCapabilityOverrides = Field(
        default_factory=EmbeddingCapabilityOverrides
    )

    @model_validator(mode="after")
    def validate_required_capabilities(self) -> LLMEmbeddingResolveRequest:
        """Require unique Embedding capability requirements.

        Intro:
            Duplicate requirements create redundant diagnostics and are rejected
            before canonical binding resolution.

        Examples:
            Require adjustable dimensions:
                ```python
                request = LLMEmbeddingResolveRequest(
                    provider_id="openai",
                    endpoint_id="openai_embeddings",
                    model_id="text-embedding-3-small",
                    required_capabilities=("dimensions",),
                )
                ```

            Reject duplicate requirements:
                ```python
                LLMEmbeddingResolveRequest(
                    provider_id="openai",
                    endpoint_id="openai_embeddings",
                    model_id="text-embedding-3-small",
                    required_capabilities=("dimensions", "dimensions"),
                )
                ```

        Args:
            self: Fully parsed Embedding binding request.

        Returns:
            LLMEmbeddingResolveRequest: Unchanged request with unique requirements.

        Notes:
            Caller order remains the deterministic diagnostic order.
        """

        _require_unique_capabilities(self.required_capabilities, operation="Embedding")
        return self


class LLMEmbeddingResolveResponse(LLMApiContract):
    """Return one pinned Embedding binding and capability diagnostics."""

    schema_version: Literal["aethergraph.embedding-resolve-response/v1"] = (
        "aethergraph.embedding-resolve-response/v1"
    )
    valid: bool
    binding: ResolvedEmbeddingBinding


class LLMImageGenerationResolveRequest(LLMApiContract):
    """Request capability resolution for one explicit Image Generation binding."""

    schema_version: Literal["aethergraph.image-resolve-request/v1"] = (
        "aethergraph.image-resolve-request/v1"
    )
    provider_id: str = Field(min_length=1, max_length=128)
    endpoint_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=512)
    required_capabilities: tuple[ImageGenerationCapabilityName, ...] = ()
    capability_overrides: ImageGenerationCapabilityOverrides = Field(
        default_factory=ImageGenerationCapabilityOverrides
    )

    @model_validator(mode="after")
    def validate_required_capabilities(self) -> LLMImageGenerationResolveRequest:
        """Require unique Image Generation capability requirements.

        Intro:
            Duplicate requirements create redundant diagnostics and are rejected
            before canonical binding resolution.

        Examples:
            Require text-to-image support:
                ```python
                request = LLMImageGenerationResolveRequest(
                    provider_id="openai",
                    endpoint_id="openai_images",
                    model_id="gpt-image-1",
                    required_capabilities=("text_to_image",),
                )
                ```

            Reject duplicate requirements:
                ```python
                LLMImageGenerationResolveRequest(
                    provider_id="openai",
                    endpoint_id="openai_images",
                    model_id="gpt-image-1",
                    required_capabilities=("text_to_image", "text_to_image"),
                )
                ```

        Args:
            self: Fully parsed Image Generation binding request.

        Returns:
            LLMImageGenerationResolveRequest: Unchanged request with unique requirements.

        Notes:
            Caller order remains the deterministic diagnostic order.
        """

        _require_unique_capabilities(
            self.required_capabilities,
            operation="Image Generation",
        )
        return self


class LLMImageGenerationResolveResponse(LLMApiContract):
    """Return one pinned Image Generation binding and diagnostics."""

    schema_version: Literal["aethergraph.image-resolve-response/v1"] = (
        "aethergraph.image-resolve-response/v1"
    )
    valid: bool
    binding: ResolvedImageGenerationBinding


__all__ = [
    "LLMChatResolveRequest",
    "LLMChatResolveResponse",
    "LLMEmbeddingResolveRequest",
    "LLMEmbeddingResolveResponse",
    "LLMEndpointAdapterView",
    "LLMModelCatalogResponse",
    "LLMImageGenerationResolveRequest",
    "LLMImageGenerationResolveResponse",
    "LLMProviderView",
    "LLMRegistryResponse",
]

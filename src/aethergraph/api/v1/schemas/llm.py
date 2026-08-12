"""Versioned API schemas for model registry, catalog, and binding resolution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aethergraph.services.llm.capabilities import (
    ChatCapabilityName,
    ResolvedModelBinding,
)
from aethergraph.services.llm.catalog import ModelCatalogEntry
from aethergraph.services.llm.profiles import ChatCapabilityOverrides
from aethergraph.services.llm.registry import ModelOperation


class LLMApiContract(BaseModel):
    """Reject undeclared fields across the public model-metadata API."""

    model_config = ConfigDict(extra="forbid")


class LLMEndpointAdapterView(LLMApiContract):
    """Describe one provider-owned selectable endpoint adapter."""

    adapter_id: str
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

        if len(self.required_capabilities) != len(set(self.required_capabilities)):
            raise ValueError("required Chat capabilities must be unique")
        return self


class LLMChatResolveResponse(LLMApiContract):
    """Return one pinned Chat binding and its capability diagnostics."""

    schema_version: Literal["aethergraph.chat-resolve-response/v1"] = (
        "aethergraph.chat-resolve-response/v1"
    )
    valid: bool
    binding: ResolvedModelBinding


__all__ = [
    "LLMChatResolveRequest",
    "LLMChatResolveResponse",
    "LLMEndpointAdapterView",
    "LLMModelCatalogResponse",
    "LLMProviderView",
    "LLMRegistryResponse",
]

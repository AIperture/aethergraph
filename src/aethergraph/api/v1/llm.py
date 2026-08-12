"""Read-only model registry, catalog, and capability-resolution routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas.llm import (
    LLMChatResolveRequest,
    LLMChatResolveResponse,
    LLMEndpointAdapterView,
    LLMModelCatalogResponse,
    LLMProviderView,
    LLMRegistryResponse,
)
from aethergraph.services.llm.capabilities import resolve_chat_profile
from aethergraph.services.llm.catalog import catalog_digest, load_model_catalog
from aethergraph.services.llm.profiles import (
    ChatProfile,
    ModelSelection,
    ProviderConnection,
)
from aethergraph.services.llm.registry import ENDPOINT_ADAPTERS, PROVIDERS

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/registry", response_model=LLMRegistryResponse)
async def list_llm_registry(
    identity: Annotated[RequestIdentity, Depends(get_identity)],
    include_hidden: Annotated[bool, Query()] = False,
) -> LLMRegistryResponse:
    """Return registered providers and their selectable operation endpoints.

    Intro:
        The route projects AG's immutable runtime registry without constructing a
        client, reading a credential value, discovering models, or calling a
        provider.

    Examples:
        List Studio-visible providers:
            ```python
            response = client.get("/api/v1/llm/registry")
            providers = response.json()["providers"]
            ```

        Include test-only or internal providers:
            ```python
            response = client.get("/api/v1/llm/registry?include_hidden=true")
            ```

    Args:
        identity: Authenticated request identity.
        include_hidden: Whether to include providers not intended for Studio.

    Returns:
        LLMRegistryResponse: Deterministically ordered provider and endpoint data.

    Notes:
        Environment-variable names are configuration hints; no environment values
        or secret material are returned.
    """

    del identity
    providers = tuple(
        LLMProviderView(
            provider_id=provider.provider_id,
            display_name=provider.display_name,
            studio_visible=provider.studio_visible,
            default_endpoints=dict(provider.default_endpoints),
            default_base_url=provider.default_base_url,
            base_url_env=provider.base_url_env,
            credential_envs=provider.credential_envs,
            model_discovery_adapter_id=provider.model_discovery_adapter_id,
            endpoints=tuple(
                LLMEndpointAdapterView(
                    adapter_id=adapter.adapter_id,
                    protocol_family=adapter.protocol_family,
                    implemented_operations=adapter.implemented_operations,
                    implementation_capabilities=adapter.implementation_capabilities,
                )
                for adapter in (
                    ENDPOINT_ADAPTERS[adapter_id] for adapter_id in provider.endpoint_ids
                )
            ),
        )
        for provider in sorted(PROVIDERS.values(), key=lambda item: item.provider_id)
        if include_hidden or provider.studio_visible
    )
    return LLMRegistryResponse(providers=providers)


@router.get("/catalog", response_model=LLMModelCatalogResponse)
async def get_llm_model_catalog(
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> LLMModelCatalogResponse:
    """Return the validated production capability catalog and its digest.

    Intro:
        The route loads AG's packaged catalog through the same validator and
        digest function used by runtime resolution.

    Examples:
        Read catalog identity:
            ```python
            response = client.get("/api/v1/llm/catalog")
            digest = response.json()["digest"]
            ```

        Inspect evidence-backed entries:
            ```python
            entries = client.get("/api/v1/llm/catalog").json()["entries"]
            ```

    Args:
        identity: Authenticated request identity.

    Returns:
        LLMModelCatalogResponse: Validated entries, revision, and canonical digest.

    Notes:
        Catalog source URLs are public provenance. The route performs no live
        documentation refresh or model discovery.
    """

    del identity
    catalog = load_model_catalog()
    return LLMModelCatalogResponse(
        catalog_schema_version=catalog.schema_version,
        catalog_revision=catalog.catalog_revision,
        digest=catalog_digest(catalog),
        entries=catalog.entries,
    )


@router.post("/resolve/chat", response_model=LLMChatResolveResponse)
async def resolve_llm_chat_binding(
    body: LLMChatResolveRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> LLMChatResolveResponse:
    """Resolve capabilities for one caller-selected Chat endpoint binding.

    Intro:
        The route pins the submitted provider, endpoint, and model before applying
        overrides and required-capability checks with complete provenance.

    Examples:
        Resolve an OpenAI Responses binding:
            ```python
            response = client.post(
                "/api/v1/llm/resolve/chat",
                json={
                    "provider_id": "openai",
                    "endpoint_id": "openai_responses",
                    "model_id": "gpt-5.6",
                },
            )
            ```

        Preflight native client Tool search:
            ```python
            response = client.post(
                "/api/v1/llm/resolve/chat",
                json={
                    "provider_id": "azure",
                    "endpoint_id": "azure_responses",
                    "model_id": "gpt-5.5",
                    "required_capabilities": ["native_tool_search_client"],
                },
            )
            ```

    Args:
        body: Explicit Chat binding, overrides, and required capabilities.
        identity: Authenticated request identity.

    Returns:
        LLMChatResolveResponse: Pinned effective binding and deterministic validity.

    Notes:
        Resolution is side-effect free. An invalid provider/endpoint combination
        returns HTTP 400 and is never replaced with a provider default.
    """

    del identity
    profile = ChatProfile(
        connection=ProviderConnection(
            provider_id=body.provider_id,
            endpoint_id=body.endpoint_id,
        ),
        model=ModelSelection(model_id=body.model_id),
        capability_overrides=body.capability_overrides,
    )
    try:
        binding = resolve_chat_profile(
            profile,
            required=body.required_capabilities,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return LLMChatResolveResponse(valid=binding.valid, binding=binding)


__all__ = ["router"]

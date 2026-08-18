"""Read-only model registry, catalog, and capability-resolution routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
import httpx

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.schemas.llm import (
    LLMChatResolveRequest,
    LLMChatResolveResponse,
    LLMEmbeddingResolveRequest,
    LLMEmbeddingResolveResponse,
    LLMEndpointAdapterView,
    LLMImageGenerationResolveRequest,
    LLMImageGenerationResolveResponse,
    LLMModelCatalogResponse,
    LLMModelDiscoveryRequest,
    LLMProviderView,
    LLMRegistryResponse,
)
from aethergraph.services.llm.capabilities import (
    resolve_chat_profile,
    resolve_embedding_profile,
    resolve_image_generation_profile,
)
from aethergraph.services.llm.catalog import catalog_digest, load_model_catalog
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.model_discovery import (
    ModelDiscoveryError,
    ModelDiscoveryResult,
    discover_provider_models,
)
from aethergraph.services.llm.profiles import (
    ChatProfile,
    EmbeddingProfileSpec,
    ImageGenerationProfile,
    ModelSelection,
    ProviderConnection,
)
from aethergraph.services.llm.registry import ENDPOINT_ADAPTERS, PROVIDERS

router = APIRouter(prefix="/llm", tags=["llm"])


def build_llm_registry_response(*, include_hidden: bool = False) -> LLMRegistryResponse:
    """Build the versioned provider and endpoint registry projection.

    Intro:
        The builder gives embedded hosts the exact response used by the AG HTTP
        route without requiring a second server or reimplementing registry truth.

    Examples:
        Build the Studio-visible registry:
            ```python
            response = build_llm_registry_response()
            ```

        Include internal providers for diagnostics:
            ```python
            response = build_llm_registry_response(include_hidden=True)
            ```

    Args:
        include_hidden: Whether to include providers not intended for Studio.

    Returns:
        LLMRegistryResponse: Deterministically ordered versioned registry view.

    Notes:
        The builder reads immutable in-process descriptors and no environment
        values, credentials, or provider APIs.
    """

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
                    adapter_revision=adapter.adapter_revision,
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


def build_llm_model_catalog_response() -> LLMModelCatalogResponse:
    """Build the versioned validated production-catalog projection.

    Intro:
        Embedded hosts receive the same catalog revision, entries, and canonical
        digest as the AG HTTP route and live resolver.

    Examples:
        Read the catalog digest:
            ```python
            digest = build_llm_model_catalog_response().digest
            ```

        Inspect catalog entries:
            ```python
            entries = build_llm_model_catalog_response().entries
            ```

    Args:
        None.

    Returns:
        LLMModelCatalogResponse: Validated catalog identity and entries.

    Notes:
        The builder performs no live documentation refresh or model discovery.
    """

    catalog = load_model_catalog()
    return LLMModelCatalogResponse(
        catalog_schema_version=catalog.schema_version,
        catalog_revision=catalog.catalog_revision,
        digest=catalog_digest(catalog),
        entries=catalog.entries,
    )


async def build_llm_model_discovery_response(
    body: LLMModelDiscoveryRequest,
    *,
    credential: str | None = None,
    base_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelDiscoveryResult:
    """Refresh one provider's model list through its registered AG adapter.

    Intro:
        Embedded hosts can supply their already-resolved profile connection,
        while the AG HTTP route uses only registry and environment configuration.

    Examples:
        Discover from AG registry configuration:
            ```python
            result = await build_llm_model_discovery_response(request)
            ```

        Discover for an embedded settings profile:
            ```python
            result = await build_llm_model_discovery_response(
                request,
                credential=resolved_key,
                base_url=profile.base_url,
            )
            ```

    Args:
        body: Provider identity and bounded result limit.
        credential: Optional already-resolved provider inference credential.
        base_url: Optional embedded-host profile base URL.
        transport: Optional injected HTTP transport for deterministic tests.

    Returns:
        ModelDiscoveryResult: Provider-reported models or explicit unavailability.

    Notes:
        When no credential is supplied, AG resolves only registry-declared
        environment variables. Secret values never enter the response.
    """

    resolved_credential = (
        credential
        if credential is not None
        else resolve_provider_credential(
            provider_id=body.provider_id,
            direct=None,
            secret_ref=None,
            secrets=None,
        ).value
    )
    return await discover_provider_models(
        body.provider_id,
        credential=resolved_credential,
        base_url=base_url,
        limit=body.limit,
        transport=transport,
    )


def build_llm_chat_resolve_response(
    body: LLMChatResolveRequest,
) -> LLMChatResolveResponse:
    """Build one side-effect-free explicit Chat binding resolution.

    Intro:
        Embedded hosts and the AG route share endpoint membership validation,
        capability provenance, override handling, and fail-closed requirements.

    Examples:
        Resolve an ordinary binding:
            ```python
            response = build_llm_chat_resolve_response(request)
            ```

        Inspect failed requirements:
            ```python
            response = build_llm_chat_resolve_response(strict_request)
            assert response.valid or response.binding.diagnostics
            ```

    Args:
        body: Explicit provider, endpoint, model, overrides, and requirements.

    Returns:
        LLMChatResolveResponse: Pinned effective binding and validity.

    Notes:
        Invalid provider or endpoint combinations raise `KeyError` or `ValueError`
        for the owning API boundary to translate without selecting a fallback.
    """

    profile = ChatProfile(
        connection=ProviderConnection(
            provider_id=body.provider_id,
            endpoint_id=body.endpoint_id,
        ),
        model=ModelSelection(model_id=body.model_id),
        capability_overrides=body.capability_overrides,
    )
    binding = resolve_chat_profile(
        profile,
        required=body.required_capabilities,
    )
    return LLMChatResolveResponse(valid=binding.valid, binding=binding)


def build_llm_embedding_resolve_response(
    body: LLMEmbeddingResolveRequest,
) -> LLMEmbeddingResolveResponse:
    """Build one side-effect-free explicit Embedding binding resolution.

    Intro:
        Embedded hosts and HTTP routes share exact endpoint membership,
        capability provenance, overrides, and fail-closed requirements.

    Examples:
        Resolve a known model:
            ```python
            response = build_llm_embedding_resolve_response(request)
            ```

        Inspect a failed requirement:
            ```python
            assert response.valid or response.binding.diagnostics
            ```

    Args:
        body: Explicit Embedding binding, overrides, and requirements.

    Returns:
        LLMEmbeddingResolveResponse: Pinned effective binding and validity.

    Notes:
        Resolution performs no provider request and selects no fallback endpoint.
    """

    profile = EmbeddingProfileSpec(
        connection=ProviderConnection(
            provider_id=body.provider_id,
            endpoint_id=body.endpoint_id,
        ),
        model=ModelSelection(model_id=body.model_id),
        capability_overrides=body.capability_overrides,
    )
    binding = resolve_embedding_profile(profile, required=body.required_capabilities)
    return LLMEmbeddingResolveResponse(valid=binding.valid, binding=binding)


def build_llm_image_generation_resolve_response(
    body: LLMImageGenerationResolveRequest,
) -> LLMImageGenerationResolveResponse:
    """Build one side-effect-free explicit Image Generation resolution.

    Intro:
        Embedded hosts and HTTP routes share exact endpoint membership,
        capability provenance, overrides, and fail-closed requirements.

    Examples:
        Resolve a known model:
            ```python
            response = build_llm_image_generation_resolve_response(request)
            ```

        Inspect a failed requirement:
            ```python
            assert response.valid or response.binding.diagnostics
            ```

    Args:
        body: Explicit Image Generation binding, overrides, and requirements.

    Returns:
        LLMImageGenerationResolveResponse: Pinned effective binding and validity.

    Notes:
        Resolution performs no provider request and selects no fallback endpoint.
    """

    profile = ImageGenerationProfile(
        connection=ProviderConnection(
            provider_id=body.provider_id,
            endpoint_id=body.endpoint_id,
        ),
        model=ModelSelection(model_id=body.model_id),
        capability_overrides=body.capability_overrides,
    )
    binding = resolve_image_generation_profile(
        profile,
        required=body.required_capabilities,
    )
    return LLMImageGenerationResolveResponse(valid=binding.valid, binding=binding)


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
    return build_llm_registry_response(include_hidden=include_hidden)


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
    return build_llm_model_catalog_response()


@router.post("/discovery/models", response_model=ModelDiscoveryResult)
async def discover_llm_models(
    body: LLMModelDiscoveryRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> ModelDiscoveryResult:
    """Refresh one provider-native model list without fallback or caching.

    Intro:
        The route selects the provider registry's exact discovery adapter and
        uses only server-owned environment configuration for its connection.

    Examples:
        Refresh OpenAI models:
            ```python
            response = client.post(
                "/api/v1/llm/discovery/models",
                json={"provider_id": "openai"},
            )
            ```

        Inspect unavailable discovery:
            ```python
            assert response.json()["status"] in {"success", "unavailable"}
            ```

    Args:
        body: Provider identity and bounded result limit.
        identity: Authenticated request identity.

    Returns:
        ModelDiscoveryResult: Provider-reported models or explicit unavailability.

    Notes:
        Custom base URLs and credentials are not accepted at the HTTP boundary.
        Sanitized provider transport failures return HTTP 502.
    """

    del identity
    try:
        return await build_llm_model_discovery_response(body)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelDiscoveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


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
    try:
        return build_llm_chat_resolve_response(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/resolve/embeddings", response_model=LLMEmbeddingResolveResponse)
async def resolve_llm_embedding_binding(
    body: LLMEmbeddingResolveRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> LLMEmbeddingResolveResponse:
    """Resolve one caller-selected Embedding endpoint binding.

    Intro:
        The route applies AG catalog facts, explicit overrides, and adapter
        implementation clamps to an exact provider/endpoint/model identity.

    Examples:
        Resolve an OpenAI embedding model:
            ```python
            response = client.post("/api/v1/llm/resolve/embeddings", json=request)
            ```

        Inspect adjustable-dimension support:
            ```python
            state = response.json()["binding"]["capabilities"]["dimensions"]["state"]
            ```

    Args:
        body: Explicit Embedding binding, overrides, and required capabilities.
        identity: Authenticated request identity.

    Returns:
        LLMEmbeddingResolveResponse: Pinned effective binding and validity.

    Notes:
        Invalid provider/endpoint combinations return HTTP 400 without fallback.
    """

    del identity
    try:
        return build_llm_embedding_resolve_response(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/resolve/image-generation", response_model=LLMImageGenerationResolveResponse)
async def resolve_llm_image_generation_binding(
    body: LLMImageGenerationResolveRequest,
    identity: Annotated[RequestIdentity, Depends(get_identity)],
) -> LLMImageGenerationResolveResponse:
    """Resolve one caller-selected Image Generation endpoint binding.

    Intro:
        The route applies AG catalog facts, explicit overrides, and adapter
        implementation clamps to an exact provider/endpoint/model identity.

    Examples:
        Resolve an OpenAI image model:
            ```python
            response = client.post("/api/v1/llm/resolve/image-generation", json=request)
            ```

        Inspect image-editing support:
            ```python
            state = response.json()["binding"]["capabilities"]["image_editing"]["state"]
            ```

    Args:
        body: Explicit Image Generation binding, overrides, and requirements.
        identity: Authenticated request identity.

    Returns:
        LLMImageGenerationResolveResponse: Pinned effective binding and validity.

    Notes:
        Invalid provider/endpoint combinations return HTTP 400 without fallback.
    """

    del identity
    try:
        return build_llm_image_generation_resolve_response(body)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = [
    "build_llm_chat_resolve_response",
    "build_llm_embedding_resolve_response",
    "build_llm_image_generation_resolve_response",
    "build_llm_model_catalog_response",
    "build_llm_model_discovery_response",
    "build_llm_registry_response",
    "router",
]

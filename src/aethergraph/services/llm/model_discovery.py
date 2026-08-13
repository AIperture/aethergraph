"""Bounded provider-native model discovery with no capability invention."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from .catalog import load_model_catalog
from .registry import ModelOperation, get_provider_descriptor, provider_default_base_url

ModelDiscoveryStatus = Literal["success", "unavailable"]


class ModelDiscoveryContract(BaseModel):
    """Reject undeclared fields in provider model-discovery results."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DiscoveredModel(ModelDiscoveryContract):
    """Describe one provider-reported model without asserting unknown facts."""

    model_id: str = Field(min_length=1, max_length=512)
    display_name: str | None = Field(default=None, min_length=1, max_length=512)
    reported_methods: tuple[str, ...] = ()
    catalog_operations: tuple[ModelOperation, ...] = ()


class ModelDiscoveryDiagnostic(ModelDiscoveryContract):
    """Explain why provider-native discovery is currently unavailable."""

    code: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=1000)


class ModelDiscoveryResult(ModelDiscoveryContract):
    """Return one bounded provider-native discovery attempt."""

    schema_version: Literal["aethergraph.model-discovery/v1"] = (
        "aethergraph.model-discovery/v1"
    )
    provider_id: str = Field(min_length=1, max_length=128)
    adapter_id: str | None = Field(default=None, min_length=1, max_length=128)
    status: ModelDiscoveryStatus
    models: tuple[DiscoveredModel, ...] = ()
    diagnostics: tuple[ModelDiscoveryDiagnostic, ...] = ()
    discovered_at: datetime


class ModelDiscoveryError(RuntimeError):
    """Report a sanitized provider model-discovery transport failure."""

    def __init__(self, *, provider_id: str, status_code: int | None = None) -> None:
        self.provider_id = provider_id
        self.status_code = status_code
        detail = f"Model discovery failed for provider {provider_id!r}"
        if status_code is not None:
            detail += f" with HTTP status {status_code}"
        super().__init__(detail + ".")


@dataclass(frozen=True)
class ModelDiscoveryAdapterDescriptor:
    """Declare one exact model-list protocol and credential scope."""

    adapter_id: str
    protocol_family: Literal[
        "openai.models",
        "anthropic.models",
        "gemini.models",
        "azure.management.deployments",
    ]
    credential_scope: Literal["inference", "management"]
    available: bool = True


MODEL_DISCOVERY_ADAPTERS: Mapping[str, ModelDiscoveryAdapterDescriptor] = {
    descriptor.adapter_id: descriptor
    for descriptor in (
        ModelDiscoveryAdapterDescriptor(
            "openai_models",
            "openai.models",
            "inference",
        ),
        ModelDiscoveryAdapterDescriptor(
            "openai_compatible_models",
            "openai.models",
            "inference",
        ),
        ModelDiscoveryAdapterDescriptor(
            "anthropic_models",
            "anthropic.models",
            "inference",
        ),
        ModelDiscoveryAdapterDescriptor(
            "gemini_models",
            "gemini.models",
            "inference",
        ),
        ModelDiscoveryAdapterDescriptor(
            "azure_openai_deployments",
            "azure.management.deployments",
            "management",
            available=False,
        ),
    )
}


async def discover_provider_models(
    provider_id: str,
    *,
    credential: str | None,
    base_url: str | None = None,
    limit: int = 200,
    timeout_s: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> ModelDiscoveryResult:
    """Discover models through one registered provider-native list protocol.

    Intro:
        The function selects exactly the provider registry's discovery adapter,
        performs at most one bounded HTTP request, and enriches results only with
        matching packaged-catalog operation evidence.

    Examples:
        Discover an OpenAI-compatible endpoint:
            ```python
            result = await discover_provider_models(
                "openai_compatible",
                credential="secret",
                base_url="http://localhost:1234/v1",
            )
            ```

        Inspect explicit unavailability:
            ```python
            result = await discover_provider_models("azure", credential=None)
            assert result.status == "unavailable"
            ```

    Args:
        provider_id: Exact registered provider identity.
        credential: Resolved inference credential, never persisted or returned.
        base_url: Optional explicit provider API base URL.
        limit: Maximum returned models and provider page size.
        timeout_s: Whole discovery request timeout in seconds.
        transport: Optional injected HTTP transport for deterministic tests.

    Returns:
        ModelDiscoveryResult: Bounded models or an explicit unavailable result.

    Notes:
        Discovery has no retry, pagination loop, alternate endpoint, or cache.
        Provider error bodies are never included in exceptions or results.
    """

    if limit < 1 or limit > 1000:
        raise ValueError("model discovery limit must be between 1 and 1000")
    if timeout_s <= 0 or timeout_s > 60:
        raise ValueError("model discovery timeout must be greater than 0 and at most 60")
    provider = get_provider_descriptor(provider_id)
    adapter_id = provider.model_discovery_adapter_id
    now = datetime.now(UTC)
    if adapter_id is None:
        return _unavailable(
            provider_id,
            None,
            "model_discovery_not_registered",
            "This provider has no registered model-discovery adapter.",
            now,
        )
    try:
        adapter = MODEL_DISCOVERY_ADAPTERS[adapter_id]
    except KeyError as exc:
        raise RuntimeError(f"Unknown model-discovery adapter: {adapter_id!r}") from exc
    if not adapter.available:
        return _unavailable(
            provider_id,
            adapter.adapter_id,
            "management_credentials_required",
            "Azure deployment discovery requires explicit management-plane resource identity and OAuth credentials.",
            now,
        )
    if provider.credential_envs and not credential:
        return _unavailable(
            provider_id,
            adapter.adapter_id,
            "model_discovery_credential_required",
            "Configure this provider credential before refreshing its model list.",
            now,
        )
    resolved_base_url = (base_url or provider_default_base_url(provider_id) or "").rstrip("/")
    if not resolved_base_url:
        return _unavailable(
            provider_id,
            adapter.adapter_id,
            "model_discovery_base_url_required",
            "Configure a base URL before refreshing this provider's model list.",
            now,
        )

    url, headers, params = _request_parts(
        adapter,
        base_url=resolved_base_url,
        credential=credential,
        limit=limit,
    )
    try:
        async with httpx.AsyncClient(
            timeout=timeout_s,
            transport=transport,
            follow_redirects=False,
        ) as client, client.stream("GET", url, headers=headers, params=params) as response:
            response.raise_for_status()
            content = bytearray()
            async for chunk in response.aiter_bytes():
                content.extend(chunk)
                if len(content) > 2_000_000:
                    raise ModelDiscoveryError(provider_id=provider_id)
        payload = json.loads(content)
    except httpx.HTTPStatusError as exc:
        raise ModelDiscoveryError(
            provider_id=provider_id,
            status_code=exc.response.status_code,
        ) from exc
    except (httpx.TransportError, ValueError) as exc:
        raise ModelDiscoveryError(provider_id=provider_id) from exc

    discovered = _parse_models(adapter, payload, provider_id=provider_id)[:limit]
    operations = _catalog_operations(provider_id)
    models = tuple(
        model.model_copy(
            update={"catalog_operations": operations.get(model.model_id, ())}
        )
        for model in discovered
    )
    return ModelDiscoveryResult(
        provider_id=provider_id,
        adapter_id=adapter.adapter_id,
        status="success",
        models=models,
        discovered_at=now,
    )


def _request_parts(
    adapter: ModelDiscoveryAdapterDescriptor,
    *,
    base_url: str,
    credential: str | None,
    limit: int,
) -> tuple[str, dict[str, str], dict[str, str | int]]:
    if adapter.protocol_family == "openai.models":
        headers = {"Authorization": f"Bearer {credential}"} if credential else {}
        return f"{base_url}/models", headers, {}
    if adapter.protocol_family == "anthropic.models":
        return (
            f"{base_url}/v1/models",
            {
                "x-api-key": credential or "",
                "anthropic-version": "2023-06-01",
            },
            {"limit": limit},
        )
    if adapter.protocol_family == "gemini.models":
        return (
            f"{base_url}/v1beta/models",
            {"x-goog-api-key": credential or ""},
            {"pageSize": limit},
        )
    raise RuntimeError(f"Unsupported model-discovery protocol: {adapter.protocol_family}")


def _parse_models(
    adapter: ModelDiscoveryAdapterDescriptor,
    payload: Any,
    *,
    provider_id: str,
) -> tuple[DiscoveredModel, ...]:
    if not isinstance(payload, dict):
        raise ModelDiscoveryError(provider_id=provider_id)
    if adapter.protocol_family in {"openai.models", "anthropic.models"}:
        items = payload.get("data")
    else:
        items = payload.get("models")
    if not isinstance(items, list):
        raise ModelDiscoveryError(provider_id=provider_id)
    discovered: dict[str, DiscoveredModel] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id") or item.get("name")
        if not isinstance(raw_id, str) or not raw_id.strip():
            continue
        model_id = raw_id.strip().removeprefix("models/")
        display_name = item.get("display_name") or item.get("displayName")
        methods = item.get("supportedGenerationMethods")
        discovered[model_id] = DiscoveredModel(
            model_id=model_id,
            display_name=(
                display_name.strip()
                if isinstance(display_name, str) and display_name.strip()
                else None
            ),
            reported_methods=tuple(
                sorted({str(method) for method in methods if str(method).strip()})
            )
            if isinstance(methods, list)
            else (),
        )
    return tuple(discovered[key] for key in sorted(discovered))


def _catalog_operations(provider_id: str) -> dict[str, tuple[ModelOperation, ...]]:
    collected: dict[str, set[ModelOperation]] = {}
    for entry in load_model_catalog().entries:
        if entry.provider_id != provider_id:
            continue
        if entry.model_id is not None:
            collected.setdefault(entry.model_id, set()).add(entry.operation)
    order = {"chat": 0, "embeddings": 1, "image_generation": 2}
    return {
        model_id: tuple(sorted(operations, key=order.__getitem__))
        for model_id, operations in collected.items()
    }


def _unavailable(
    provider_id: str,
    adapter_id: str | None,
    code: str,
    message: str,
    discovered_at: datetime,
) -> ModelDiscoveryResult:
    return ModelDiscoveryResult(
        provider_id=provider_id,
        adapter_id=adapter_id,
        status="unavailable",
        diagnostics=(ModelDiscoveryDiagnostic(code=code, message=message),),
        discovered_at=discovered_at,
    )


__all__ = [
    "DiscoveredModel",
    "MODEL_DISCOVERY_ADAPTERS",
    "ModelDiscoveryAdapterDescriptor",
    "ModelDiscoveryDiagnostic",
    "ModelDiscoveryError",
    "ModelDiscoveryResult",
    "ModelDiscoveryStatus",
    "discover_provider_models",
]

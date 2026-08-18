from __future__ import annotations

import json

import httpx
import pytest

from aethergraph.services.llm import ModelDiscoveryError, discover_provider_models


@pytest.mark.asyncio
async def test_openai_compatible_discovery_is_bounded_sorted_and_enriched() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://localhost:1234/v1/models"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "z-local"},
                    {"id": "text-embedding-3-small"},
                    {"id": "z-local"},
                ]
            },
        )

    result = await discover_provider_models(
        "openai_compatible",
        credential="secret",
        base_url="http://localhost:1234/v1/",
        limit=2,
        transport=httpx.MockTransport(handler),
    )

    assert result.status == "success"
    assert [model.model_id for model in result.models] == [
        "text-embedding-3-small",
        "z-local",
    ]
    assert result.models[0].catalog_operations == ()


@pytest.mark.asyncio
async def test_gemini_discovery_preserves_reported_methods() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["pageSize"] == "100"
        assert request.headers["x-goog-api-key"] == "google-secret"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "models/gemini-embedding-001",
                        "displayName": "Gemini Embedding",
                        "supportedGenerationMethods": [
                            "embedContent",
                            "batchEmbedContents",
                        ],
                    }
                ]
            },
        )

    result = await discover_provider_models(
        "google",
        credential="google-secret",
        limit=100,
        transport=httpx.MockTransport(handler),
    )

    assert result.models[0].model_id == "gemini-embedding-001"
    assert result.models[0].display_name == "Gemini Embedding"
    assert result.models[0].reported_methods == (
        "batchEmbedContents",
        "embedContent",
    )
    assert result.models[0].catalog_operations == ()


@pytest.mark.asyncio
async def test_anthropic_discovery_uses_required_version_header() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.anthropic.com/v1/models?limit=20"
        assert request.headers["x-api-key"] == "anthropic-secret"
        assert request.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            json={"data": [{"id": "claude-test", "display_name": "Claude Test"}]},
        )

    result = await discover_provider_models(
        "anthropic",
        credential="anthropic-secret",
        limit=20,
        transport=httpx.MockTransport(handler),
    )

    assert result.models[0].display_name == "Claude Test"


@pytest.mark.asyncio
async def test_discovery_reports_missing_credentials_without_transport() -> None:
    result = await discover_provider_models("openai", credential=None)

    assert result.status == "unavailable"
    assert result.models == ()
    assert result.diagnostics[0].code == "model_discovery_credential_required"


@pytest.mark.asyncio
async def test_azure_discovery_requires_management_plane_identity() -> None:
    result = await discover_provider_models("azure", credential="inference-key")

    assert result.status == "unavailable"
    assert result.diagnostics[0].code == "management_credentials_required"


@pytest.mark.asyncio
async def test_discovery_sanitizes_provider_error_bodies() -> None:
    secret_body = {"error": {"message": "credential=never-expose"}}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=json.dumps(secret_body).encode())

    with pytest.raises(ModelDiscoveryError) as error:
        await discover_provider_models(
            "openai",
            credential="bad-secret",
            transport=httpx.MockTransport(handler),
        )

    assert error.value.status_code == 401
    assert "never-expose" not in str(error.value)


@pytest.mark.asyncio
async def test_discovery_rejects_oversized_responses() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{" + b" " * 2_000_001 + b"}")

    with pytest.raises(ModelDiscoveryError):
        await discover_provider_models(
            "openai",
            credential="secret",
            transport=httpx.MockTransport(handler),
        )

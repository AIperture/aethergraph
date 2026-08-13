from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aethergraph.api.v1.deps import RequestIdentity, get_identity
from aethergraph.api.v1.llm import router
from aethergraph.services.llm.catalog import catalog_digest, load_model_catalog


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_identity] = lambda: RequestIdentity(
        user_id="local-user",
        mode="local",
    )
    return TestClient(app)


def test_registry_api_returns_visible_providers_and_exact_endpoints() -> None:
    with _client() as client:
        response = client.get("/api/v1/llm/registry")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "aethergraph.llm-registry/v1"
    providers = {item["provider_id"]: item for item in payload["providers"]}
    assert "dummy" not in providers
    assert providers["azure"]["default_endpoints"]["chat"] == ("azure_chat_completions")
    assert [item["adapter_id"] for item in providers["azure"]["endpoints"]] == [
        "azure_responses",
        "azure_chat_completions",
        "azure_embeddings",
        "azure_images",
    ]
    assert all(item["adapter_revision"] == 1 for item in providers["azure"]["endpoints"])
    assert all("api_key" not in provider for provider in providers.values())


def test_registry_api_can_include_non_studio_providers_explicitly() -> None:
    with _client() as client:
        response = client.get("/api/v1/llm/registry?include_hidden=true")

    assert response.status_code == 200
    providers = {item["provider_id"]: item for item in response.json()["providers"]}
    assert providers["dummy"]["studio_visible"] is False


def test_catalog_api_uses_runtime_catalog_loader_and_digest() -> None:
    with _client() as client:
        response = client.get("/api/v1/llm/catalog")

    assert response.status_code == 200
    payload = response.json()
    catalog = load_model_catalog()
    assert payload["schema_version"] == "aethergraph.model-catalog-api/v1"
    assert payload["catalog_schema_version"] == catalog.schema_version
    assert payload["catalog_revision"] == catalog.catalog_revision
    assert payload["digest"] == catalog_digest(catalog)
    assert len(payload["entries"]) == len(catalog.entries)


def test_chat_resolve_api_distinguishes_azure_endpoint_capabilities() -> None:
    body = {
        "provider_id": "azure",
        "model_id": "gpt-5.5",
        "required_capabilities": ["native_tool_search_client"],
    }
    with _client() as client:
        responses = client.post(
            "/api/v1/llm/resolve/chat",
            json={**body, "endpoint_id": "azure_responses"},
        )
        chat_completions = client.post(
            "/api/v1/llm/resolve/chat",
            json={**body, "endpoint_id": "azure_chat_completions"},
        )

    assert responses.status_code == 200
    assert responses.json()["valid"] is True
    assert responses.json()["binding"]["endpoint_id"] == "azure_responses"
    assert (
        responses.json()["binding"]["capabilities"]["native_tool_search_client"]["state"]
        == "supported"
    )
    assert chat_completions.status_code == 200
    assert chat_completions.json()["valid"] is False
    diagnostics = chat_completions.json()["binding"]["diagnostics"]
    assert diagnostics[0]["code"] == "required_capability_unsupported"


def test_embedding_resolve_api_returns_exact_capability_binding() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/llm/resolve/embeddings",
            json={
                "provider_id": "openai",
                "endpoint_id": "openai_embeddings",
                "model_id": "text-embedding-3-small",
                "required_capabilities": ["text_embeddings", "dimensions"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "aethergraph.embedding-resolve-response/v1"
    assert payload["valid"] is True
    assert payload["binding"]["endpoint_id"] == "openai_embeddings"
    assert payload["binding"]["capabilities"]["dimensions"]["state"] == "supported"


def test_image_resolve_api_reports_adapter_clamped_capability() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/llm/resolve/image-generation",
            json={
                "provider_id": "openai",
                "endpoint_id": "openai_images",
                "model_id": "gpt-image-1",
                "required_capabilities": ["image_editing"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "aethergraph.image-resolve-response/v1"
    assert payload["valid"] is False
    assert payload["binding"]["endpoint_id"] == "openai_images"
    assert payload["binding"]["capabilities"]["image_editing"]["state"] == "unsupported"


def test_operation_resolve_apis_reject_cross_operation_endpoints() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/llm/resolve/embeddings",
            json={
                "provider_id": "openai",
                "endpoint_id": "openai_images",
                "model_id": "text-embedding-3-small",
            },
        )

    assert response.status_code == 400
    assert "does not implement embeddings" in response.json()["detail"]


def test_chat_resolve_api_rejects_cross_provider_endpoint_without_fallback() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/llm/resolve/chat",
            json={
                "provider_id": "openai",
                "endpoint_id": "azure_responses",
                "model_id": "gpt-5.6",
            },
        )

    assert response.status_code == 400
    assert "not registered" in response.json()["detail"]


def test_chat_resolve_api_rejects_duplicate_requirements() -> None:
    with _client() as client:
        response = client.post(
            "/api/v1/llm/resolve/chat",
            json={
                "provider_id": "openai",
                "endpoint_id": "openai_responses",
                "model_id": "gpt-5.6",
                "required_capabilities": ["streaming", "streaming"],
            },
        )

    assert response.status_code == 422

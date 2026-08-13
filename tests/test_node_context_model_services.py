from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.core.runtime.runtime_env import RuntimeEnv
from aethergraph.services.llm.embedding_service import EmbeddingService
from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient


class _EmbeddingService:
    def __init__(self, clients: dict[str, object]) -> None:
        self.clients = clients

    def get(self, name: str = "default") -> object:
        return self.clients[name]


def _context(*, embedding: object | None) -> NodeContext:
    return NodeContext(
        run_id="run",
        session_id="session",
        graph_id="graph",
        node_id="node",
        services=NodeServices(
            channels=SimpleNamespace(),
            continuation_store=SimpleNamespace(),
            artifact_store=SimpleNamespace(),
            embedding=embedding,  # type: ignore[arg-type]
        ),
    )


def test_node_context_embedding_returns_the_exact_named_client() -> None:
    default = SimpleNamespace(name="default")
    search = SimpleNamespace(name="search")
    context = _context(embedding=_EmbeddingService({"default": default, "search": search}))

    assert context.embedding() is default
    assert context.embedding("search") is search


def test_node_context_embedding_fails_when_service_is_not_configured() -> None:
    context = _context(embedding=None)

    with pytest.raises(RuntimeError, match="Embedding service not available"):
        context.embedding()


def test_runtime_env_projects_the_container_embedding_service() -> None:
    service = _EmbeddingService({"default": SimpleNamespace()})
    env = RuntimeEnv(
        run_id="run",
        container=SimpleNamespace(embed_service=service),  # type: ignore[arg-type]
    )

    assert env.embedding_service is service


def test_embedding_hot_reload_replaces_all_connection_derived_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    original_transport = SimpleNamespace()
    client = GenericEmbeddingClient(
        provider="openai",
        model="embed-old",
        api_key="openai-secret",
    )
    client._client = original_transport  # type: ignore[assignment]
    client._bound_loop = object()
    service = EmbeddingService({"default": client})

    updated = service.configure_profile(
        provider="google",
        model="text-embedding-004",
    )

    assert updated is client
    assert client.provider == "google"
    assert client.endpoint_id == "gemini_embeddings"
    assert client.base_url == "https://generativelanguage.googleapis.com"
    assert client.api_key == "google-secret"
    assert client._client is None
    assert client._retired_http_clients == [original_transport]


@pytest.mark.asyncio
async def test_embedding_service_closes_each_distinct_client_once() -> None:
    class _Closeable:
        def __init__(self) -> None:
            self.close_count = 0

        async def aclose(self) -> None:
            self.close_count += 1

    client = _Closeable()
    service = EmbeddingService({"default": client, "alias": client})  # type: ignore[dict-item]

    await service.aclose()

    assert client.close_count == 1

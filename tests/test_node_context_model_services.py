from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.core.runtime.runtime_env import RuntimeEnv


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

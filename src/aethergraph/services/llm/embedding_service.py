# aethergraph/services/llm/embedding_service.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

from aethergraph.contracts.services.llm import EmbeddingClientProtocol


class EmbeddingService:
    def __init__(self, clients: Mapping[str, EmbeddingClientProtocol]):
        self._clients = dict(clients)

    def get(self, name: str = "default") -> EmbeddingClientProtocol:
        return self._clients[name]

    async def embed(
        self,
        texts: Sequence[str],
        *,
        profile: str = "default",
        model: str | None = None,
        **kw,
    ) -> list[list[float]]:
        client = self.get(profile)
        return await client.embed(texts, model=model, **kw)

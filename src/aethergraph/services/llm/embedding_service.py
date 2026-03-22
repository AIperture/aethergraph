# aethergraph/services/llm/embedding_service.py
from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging

from aethergraph.contracts.services.llm import EmbeddingClientProtocol

logger = logging.getLogger("aethergraph.services.embedding")


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

    # --- Runtime profile helpers ---------------------------------
    def configure_profile(
        self,
        name: str = "default",
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> EmbeddingClientProtocol:
        """Create or update an embedding profile in memory.

        Does NOT persist anything outside this process.
        """
        if name not in self._clients:
            from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient

            client = GenericEmbeddingClient(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout or 60.0,
            )
            self._clients[name] = client
            return client

        c = self._clients[name]
        if provider is not None:
            c.provider = provider  # type: ignore[attr-defined]
        if model is not None:
            c.model = model  # type: ignore[attr-defined]
        if base_url is not None:
            c.base_url = base_url  # type: ignore[attr-defined]
        if api_key is not None:
            c.api_key = api_key  # type: ignore[attr-defined]
        if timeout is not None:
            c.timeout = timeout  # type: ignore[attr-defined]
        return c

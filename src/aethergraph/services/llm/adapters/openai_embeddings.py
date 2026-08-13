"""Physical OpenAI-compatible Embeddings adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)


class OpenAIEmbeddingsAdapter:
    """Physical adapter for OpenAI-compatible Embeddings endpoints."""

    @staticmethod
    async def invoke(
        host: Any,
        texts: Sequence[str],
        *,
        model: str,
        extra_body: dict[str, Any],
    ) -> ProviderCallResult[list[list[float]]]:
        """Embed one text batch through an OpenAI-compatible endpoint.

        Intro:
            Projects a prepared batch into one physical request and validates
            that the provider returns exactly one vector for every input.

        Examples:
            Embed one text:
                ```python
                result = await OpenAIEmbeddingsAdapter.invoke(
                    client,
                    ("hello",),
                    model="text-embedding-3-small",
                    extra_body={},
                )
                ```

            Request provider-specific dimensions:
                ```python
                result = await OpenAIEmbeddingsAdapter.invoke(
                    client,
                    ("hello", "world"),
                    model="text-embedding-3-large",
                    extra_body={"dimensions": 256},
                )
                ```

        Args:
            host: Bound embedding client owning the HTTP transport and connection.
            texts: Ordered text inputs for one batch.
            model: Exact configured embedding model identity.
            extra_body: Provider-compatible request body extensions.

        Returns:
            ProviderCallResult[list[list[float]]]: Ordered vectors and sanitized
                transport metadata.

        Notes:
            The provider identity is retained in metadata for compatible gateways.
            The adapter performs exactly one physical attempt.
        """

        assert host._client is not None
        url = f"{host.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if host.api_key:
            headers["Authorization"] = f"Bearer {host.api_key}"
        body: dict[str, Any] = {"model": model, "input": list(texts)}
        body.update(extra_body)

        response = await host._client.post(url, headers=headers, json=body)
        metadata = checked_response_metadata(host.provider, model, "embedding", response)
        items = response.json().get("data", []) or []
        embeddings = [item.get("embedding") for item in items]
        if len(embeddings) != len(texts) or any(item is None for item in embeddings):
            raise RuntimeError(
                "Embeddings response shape mismatch: "
                f"got {len(embeddings)} items for {len(texts)} inputs"
            )
        return ProviderCallResult(embeddings, metadata)  # type: ignore[arg-type]

"""Physical Gemini Embeddings adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.types import EmbeddingResult, EmbeddingUsage


class GeminiEmbeddingsAdapter:
    """Physical adapter for Gemini batch embedding endpoints."""

    @staticmethod
    async def invoke(
        host: Any,
        texts: Sequence[str],
        *,
        model: str,
    ) -> ProviderCallResult[EmbeddingResult]:
        """Embed one text batch through Gemini `batchEmbedContents`.

        Intro:
            Projects each text into a Gemini content request and validates one
            returned vector per input.

        Examples:
            Embed one text:
                ```python
                result = await GeminiEmbeddingsAdapter.invoke(
                    client, ("hello",), model="text-embedding-004"
                )
                ```

            Embed two texts:
                ```python
                result = await GeminiEmbeddingsAdapter.invoke(
                    client,
                    ("hello", "world"),
                    model="text-embedding-004",
                )
                ```

        Args:
            host: Bound embedding client owning the HTTP transport and connection.
            texts: Ordered text inputs for one batch.
            model: Exact configured Gemini embedding model identity.

        Returns:
            ProviderCallResult[list[list[float]]]: Ordered vectors and sanitized
                transport metadata.

        Notes:
            The adapter performs exactly one physical attempt. Retry and metering
            remain facade-owned.
        """

        assert host._client is not None
        base = host.base_url.rstrip("/")
        api_key = host.api_key or ""
        url = f"{base}/v1/models/{model}:batchEmbedContents?key={api_key}"
        headers = {"Content-Type": "application/json"}
        body = {"requests": [{"content": {"parts": [{"text": text}]}} for text in texts]}

        response = await host._client.post(url, headers=headers, json=body)
        metadata = checked_response_metadata("google", model, "embedding", response)
        data = response.json()
        embeddings = [(item or {}).get("values") or [] for item in data.get("embeddings") or []]
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Gemini batch embeddings mismatch: got {len(embeddings)} for {len(texts)}"
            )
        return ProviderCallResult(
            EmbeddingResult(
                vectors=embeddings,
                usage=EmbeddingUsage.from_provider_usage(
                    data.get("usageMetadata") or data.get("usage")
                ),
            ),
            metadata,
        )

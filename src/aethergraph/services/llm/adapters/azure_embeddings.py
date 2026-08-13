"""Physical Azure OpenAI Embeddings adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.types import EmbeddingResult, EmbeddingUsage


class AzureEmbeddingsAdapter:
    """Physical adapter for Azure OpenAI Embeddings endpoints."""

    @staticmethod
    async def invoke(
        host: Any,
        texts: Sequence[str],
        *,
        model: str,
        dimensions: int | None,
        azure_deployment: str | None,
        azure_api_version: str | None,
        extra_body: dict[str, Any],
    ) -> ProviderCallResult[EmbeddingResult]:
        """Embed one text batch through an Azure deployment.

        Intro:
            Projects a prepared batch into one deployment-scoped request and
            validates one returned vector per input.

        Examples:
            Embed one text:
                ```python
                result = await AzureEmbeddingsAdapter.invoke(
                    client,
                    ("hello",),
                    model="embedding-model",
                    dimensions=None,
                    azure_deployment="embedding-prod",
                    azure_api_version=None,
                    extra_body={},
                )
                ```

            Select an API version:
                ```python
                result = await AzureEmbeddingsAdapter.invoke(
                    client,
                    ("hello", "world"),
                    model="embedding-model",
                    dimensions=256,
                    azure_deployment="embedding-prod",
                    azure_api_version="2024-08-01-preview",
                    extra_body={},
                )
                ```

        Args:
            host: Bound embedding client owning the HTTP transport and connection.
            texts: Ordered text inputs for one batch.
            model: Exact configured embedding model identity.
            dimensions: Optional requested output-vector dimensionality.
            azure_deployment: Required Azure deployment identity.
            azure_api_version: Optional Azure Embeddings API version.
            extra_body: Provider-compatible request body extensions.

        Returns:
            ProviderCallResult[list[list[float]]]: Ordered vectors and sanitized
                transport metadata.

        Notes:
            The adapter performs exactly one physical attempt. Retry and rate
            gating remain facade-owned.
        """

        if not azure_deployment:
            raise RuntimeError(
                "Azure embeddings requires AZURE_OPENAI_DEPLOYMENT (azure_deployment)"
            )
        assert host._client is not None
        api_version = azure_api_version or "2024-08-01-preview"
        url = (
            f"{host.base_url}/openai/deployments/"
            f"{azure_deployment}/embeddings?api-version={api_version}"
        )
        headers = {"api-key": host.api_key or "", "Content-Type": "application/json"}
        body: dict[str, Any] = {"input": list(texts)}
        if model:
            body["model"] = model
        body.update(extra_body)
        if dimensions is not None:
            body["dimensions"] = dimensions

        response = await host._client.post(url, headers=headers, json=body)
        metadata = checked_response_metadata("azure", model, "embedding", response)
        data = response.json()
        items = data.get("data", []) or []
        embeddings = [item.get("embedding") for item in items]
        if len(embeddings) != len(texts) or any(item is None for item in embeddings):
            raise RuntimeError(
                "Azure embeddings response shape mismatch: "
                f"got {len(embeddings)} items for {len(texts)} inputs"
            )
        return ProviderCallResult(
            EmbeddingResult(
                vectors=embeddings,  # type: ignore[arg-type]
                usage=EmbeddingUsage.from_provider_usage(data.get("usage")),
            ),
            metadata,
        )

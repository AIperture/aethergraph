"""Exact embedding endpoint-adapter dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, field
from typing import Any

from aethergraph.services.llm.adapters.azure_embeddings import AzureEmbeddingsAdapter
from aethergraph.services.llm.adapters.gemini_embeddings import GeminiEmbeddingsAdapter
from aethergraph.services.llm.adapters.openai_embeddings import OpenAIEmbeddingsAdapter
from aethergraph.services.llm.provider_transport import ProviderCallResult
from aethergraph.services.llm.types import EmbeddingResult, EmbeddingUsage

EmbeddingAdapterResult = ProviderCallResult[EmbeddingResult]
EmbeddingHandler = Callable[[Any, "EmbeddingAdapterInvocation"], Awaitable[EmbeddingAdapterResult]]


@dataclass(frozen=True)
class EmbeddingAdapterInvocation:
    """Carry one prepared single-attempt embedding invocation.

    Intro:
        Freezes provider-neutral embedding facade state before exact endpoint
        dispatch and detaches mutable request body extensions.

    Examples:
        Build an OpenAI-compatible request:
            ```python
            invocation = EmbeddingAdapterInvocation(
                texts=("north", "south"),
                model="text-embedding-3-small",
            )
            ```

        Build an Azure request with a deployment:
            ```python
            invocation = EmbeddingAdapterInvocation(
                texts=("north",),
                model="embedding-model",
                azure_deployment="embedding-prod",
                azure_api_version="2024-08-01-preview",
            )
            ```

    Args:
        texts: Detached ordered input texts for one batch.
        model: Exact configured embedding model identity.
        azure_deployment: Optional Azure deployment identity.
        azure_api_version: Optional Azure Embeddings API version.
        extra_body: Detached provider-compatible request body extensions.

    Returns:
        EmbeddingAdapterInvocation: Immutable prepared adapter-call state.

    Notes:
        Retry, rate gating, accounting, and metering remain owned by the shared
        embedding lifecycle outside this single-attempt value.
    """

    texts: tuple[str, ...]
    model: str
    azure_deployment: str | None = None
    azure_api_version: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach embedding invocation state.

        Intro:
            Rejects empty batches, non-text inputs, and empty model identities
            before a physical adapter can issue transport I/O.

        Examples:
            Validate a normal batch:
                ```python
                invocation = EmbeddingAdapterInvocation(
                    texts=("hello",), model="embed-model"
                )
                assert invocation.texts == ("hello",)
                ```

            Reject an empty batch:
                ```python
                try:
                    EmbeddingAdapterInvocation(texts=(), model="embed-model")
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized embedding invocation.

        Returns:
            None: Completes after normalized detached values are stored.

        Notes:
            Empty strings remain valid provider inputs for compatibility with
            the existing embedding facade.
        """

        texts = tuple(self.texts)
        model = str(self.model or "").strip()
        if not texts:
            raise ValueError("embedding adapter invocation requires at least one text")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("embedding adapter invocation expects text inputs")
        if not model:
            raise ValueError("embedding adapter invocation requires a model")
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "extra_body", copy.deepcopy(self.extra_body))

    def extra_body_dict(self) -> dict[str, Any]:
        """Return detached provider-compatible body extensions.

        Intro:
            Supplies one mutable request extension mapping per physical attempt.

        Examples:
            Read an empty mapping:
                ```python
                invocation = EmbeddingAdapterInvocation(
                    texts=("hello",), model="embed-model"
                )
                assert invocation.extra_body_dict() == {}
                ```

            Mutate without changing the invocation:
                ```python
                invocation = EmbeddingAdapterInvocation(
                    texts=("hello",),
                    model="embed-model",
                    extra_body={"dimensions": 256},
                )
                invocation.extra_body_dict()["dimensions"] = 128
                assert invocation.extra_body["dimensions"] == 256
                ```

        Args:
            self: Prepared embedding invocation.

        Returns:
            dict[str, Any]: Deep-copied request body extensions.

        Notes:
            Retry attempts cannot share mutations through this projection.
        """

        return copy.deepcopy(self.extra_body)


async def _invoke_openai_embeddings(
    host: Any, invocation: EmbeddingAdapterInvocation
) -> EmbeddingAdapterResult:
    return await OpenAIEmbeddingsAdapter.invoke(
        host,
        invocation.texts,
        model=invocation.model,
        extra_body=invocation.extra_body_dict(),
    )


async def _invoke_azure_embeddings(
    host: Any, invocation: EmbeddingAdapterInvocation
) -> EmbeddingAdapterResult:
    return await AzureEmbeddingsAdapter.invoke(
        host,
        invocation.texts,
        model=invocation.model,
        azure_deployment=invocation.azure_deployment,
        azure_api_version=invocation.azure_api_version,
        extra_body=invocation.extra_body_dict(),
    )


async def _invoke_gemini_embeddings(
    host: Any, invocation: EmbeddingAdapterInvocation
) -> EmbeddingAdapterResult:
    return await GeminiEmbeddingsAdapter.invoke(host, invocation.texts, model=invocation.model)


async def _invoke_dummy_embeddings(
    host: Any, invocation: EmbeddingAdapterInvocation
) -> EmbeddingAdapterResult:
    del host
    return ProviderCallResult(
        EmbeddingResult(
            vectors=[[float(len(text))] for text in invocation.texts],
            usage=EmbeddingUsage.from_provider_usage(None),
        )
    )


_EMBEDDING_HANDLERS: dict[str, EmbeddingHandler] = {
    "openai_embeddings": _invoke_openai_embeddings,
    "azure_embeddings": _invoke_azure_embeddings,
    "gemini_embeddings": _invoke_gemini_embeddings,
    "dummy_embeddings": _invoke_dummy_embeddings,
}


def registered_embedding_adapter_ids() -> frozenset[str]:
    """Return exact endpoint IDs with physical embedding handlers.

    Intro:
        Exposes immutable runtime coverage for canonical registry conformance
        without exposing mutable handler state.

    Examples:
        Check OpenAI Embeddings coverage:
            ```python
            assert "openai_embeddings" in registered_embedding_adapter_ids()
            ```

        Check a Chat endpoint:
            ```python
            assert "openai_responses" not in registered_embedding_adapter_ids()
            ```

    Args:
        This function accepts no arguments.

    Returns:
        frozenset[str]: Exact registered physical embedding adapter identities.

    Notes:
        Capability descriptors remain owned by the canonical endpoint registry.
    """

    return frozenset(_EMBEDDING_HANDLERS)


async def invoke_embedding_adapter(
    host: Any,
    *,
    adapter_id: str,
    invocation: EmbeddingAdapterInvocation,
) -> EmbeddingAdapterResult:
    """Invoke one exact registered embedding adapter.

    Intro:
        Resolves physical embedding behavior solely by canonical endpoint ID and
        fails closed when no implementation is registered.

    Examples:
        Invoke OpenAI Embeddings:
            ```python
            result = await invoke_embedding_adapter(
                client,
                adapter_id="openai_embeddings",
                invocation=invocation,
            )
            ```

        Invoke Gemini Embeddings:
            ```python
            result = await invoke_embedding_adapter(
                client,
                adapter_id="gemini_embeddings",
                invocation=invocation,
            )
            ```

    Args:
        host: Bound embedding client owning shared transport primitives.
        adapter_id: Exact selected embedding endpoint-adapter identity.
        invocation: Frozen prepared embedding invocation.

    Returns:
        EmbeddingAdapterResult: Normalized vectors and transport metadata.

    Notes:
        Selection contains no provider-name branching or fallback. Shared retry,
        rate gating, and metering remain outside this single-attempt boundary.
    """

    handler = _EMBEDDING_HANDLERS.get(str(adapter_id or "").strip())
    if handler is None:
        raise NotImplementedError(
            f"Endpoint adapter {adapter_id!r} has no embedding implementation"
        )
    return await handler(host, invocation)

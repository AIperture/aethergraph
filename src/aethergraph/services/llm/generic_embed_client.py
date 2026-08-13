# aethergraph/services/llm/embedding_client.py
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Any

import httpx

from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.services.llm.adapters.embedding import (
    EmbeddingAdapterInvocation,
    invoke_embedding_adapter,
)
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    ProviderRateGate,
    ProviderRetryExecutor,
    ProviderRetrySettings,
)
from aethergraph.services.llm.registry import provider_default_base_url, resolve_endpoint_adapter


@dataclass
class GenericEmbeddingClient(EmbeddingClientProtocol):
    """
    Provider-agnostic embedding client.

    provider: one of {"openai","azure","anthropic","google","deepseek","openrouter","lmstudio","ollama","openai_compatible","dummy"}

    Configuration (env defaults, but can be passed directly):

    - OPENAI_API_KEY / OPENAI_BASE_URL
    - AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT
    - ANTHROPIC_API_KEY
    - GOOGLE_API_KEY
    - OPENROUTER_API_KEY
    - LMSTUDIO_BASE_URL (default http://localhost:1234/v1)
    - OLLAMA_BASE_URL   (default http://localhost:11434/v1)
    """

    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    azure_deployment: str | None = None
    timeout: float = 60.0
    retry_settings: ProviderRetrySettings | None = None
    rate_limit_group: str | None = None
    rate_gate: ProviderRateGate | None = None

    # metering (optional, can be None)
    metering: MeteringService | None = None
    endpoint_id: str | None = None

    def __post_init__(self) -> None:
        self.provider = (
            self.provider or os.getenv("EMBED_PROVIDER") or os.getenv("LLM_PROVIDER") or "openai"
        ).lower()  # type: ignore[assignment]
        self.model = (
            self.model
            or os.getenv("EMBED_MODEL")
            or os.getenv("LLM_EMBED_MODEL")
            or "text-embedding-3-small"
        )

        self.api_key = resolve_provider_credential(
            provider_id=self.provider,
            direct=self.api_key,
            secret_ref=None,
            secrets=None,
        ).value
        self.base_url = self.base_url or provider_default_base_url(self.provider) or ""

        if self.endpoint_id is not None:
            self.endpoint_id = resolve_endpoint_adapter(
                self.provider,
                "embeddings",
                endpoint_id=self.endpoint_id,
            ).adapter_id

        # Azure deployment (for /deployments/{name}/embeddings)
        if self.provider == "azure" and self.azure_deployment is None:
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self._provider_retry = ProviderRetryExecutor(
            self.retry_settings,
            rate_gate=self.rate_gate,
            base_url=self.base_url,
            credential=self.api_key,
        )
        self._client: httpx.AsyncClient | None = None

    # ------------ client management -----------------

    async def _ensure_client(self) -> None:
        """
        Ensure we have an httpx.AsyncClient bound to the *current* event loop.

        IMPORTANT: We do NOT try to aclose() a client created on a different loop,
        because httpx/anyio expects it to be closed on the same loop it was created on.
        """
        loop = asyncio.get_running_loop()

        if self._client is None:
            # first-time init
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._bound_loop = loop
            return

        if self._bound_loop is not loop:
            # We're now in a different loop -> do not reuse the old client.
            # We also do NOT call aclose() here, because that tends to explode
            # if the old loop is already closed.
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._bound_loop = loop

    # ------------ public API ------------------------

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **kw: Any,
    ) -> list[list[float]]:
        """Embed one ordered batch through the configured exact endpoint.

        Intro:
            Validates provider-neutral input, resolves one endpoint before each
            lifecycle execution, and retains retry, rate gating, and best-effort
            metering outside the physical adapter.

        Examples:
            Embed one text:
                ```python
                client = GenericEmbeddingClient(provider="dummy", model="test")
                vectors = await client.embed(["hello"])
                ```

            Override the configured model:
                ```python
                client = GenericEmbeddingClient(provider="dummy", model="default")
                vectors = await client.embed(["hello", "world"], model="test")
                ```

        Args:
            self: Configured provider-neutral embedding client.
            texts: Ordered text inputs. An empty sequence returns immediately.
            model: Optional per-call model override.
            **kw: Metering context plus bounded provider request options such as
                `extra_body` and `azure_api_version`.

        Returns:
            list[list[float]]: Ordered embedding vectors matching the input count.

        Notes:
            Anthropic and DeepSeek embeddings remain explicitly unsupported.
            Physical adapters perform one attempt; this facade owns retries.
        """
        await self._ensure_client()
        assert self._client is not None

        if not isinstance(texts, Sequence) or any(not isinstance(t, str) for t in texts):
            raise TypeError("embed(texts) expects Sequence[str]")
        if len(texts) == 0:
            return []

        # Resolve model (override > configured)
        model = model or self.model or "text-embedding-3-small"

        if self.provider == "anthropic":
            raise NotImplementedError("Embeddings not supported for anthropic")
        if self.provider == "deepseek":
            raise NotImplementedError("Embeddings not supported for deepseek")
        adapter = resolve_endpoint_adapter(
            self.provider,
            "embeddings",
            endpoint_id=self.endpoint_id,
        )
        invocation = EmbeddingAdapterInvocation(
            texts=tuple(texts),
            model=model,
            azure_deployment=self.azure_deployment,
            azure_api_version=kw.get("azure_api_version"),
            extra_body=kw.get("extra_body") or {},
        )

        async def _attempt() -> ProviderCallResult[list[list[float]]]:
            return await invoke_embedding_adapter(
                self,
                adapter_id=adapter.adapter_id,
                invocation=invocation,
            )

        provider_result = await self._provider_retry.execute(
            _attempt,
            provider=self.provider,
            model=model,
            operation="embedding",
            rate_limit_group=self.rate_limit_group,
        )
        embs = provider_result.value

        # ---- metering hook (best effort) ----
        metering = self.metering or current_metering()
        if metering is not None:
            ctx = current_meter_context.get()
            try:
                # TODO: compute token estimates or bytes; for now just count inputs
                await metering.record_embedding(
                    scope=kw.get("scope"),
                    user_id=kw.get("user_id", ctx.get("user_id")),
                    org_id=kw.get("org_id", ctx.get("org_id")),
                    run_id=kw.get("run_id", ctx.get("run_id")),
                    graph_id=kw.get("graph_id", ctx.get("graph_id")),
                    client_id=kw.get("client_id"),
                    app_id=kw.get("app_id"),
                    session_id=kw.get("session_id"),
                    provider=self.provider,
                    model=model,
                    num_texts=len(texts),
                    # tokens=estimated_tokens,
                )
            except Exception:
                # best-effort; never break main path
                import logging

                logger = logging.getLogger(__name__)
                logger.exception("Error recording embedding metering")
                pass

        return embs

    async def embed_one(
        self,
        text: str,
        *,
        model: str | None = None,
        **kw: Any,
    ) -> list[float]:
        """Embed one text through the configured exact endpoint.

        Intro:
            Delegates to the batch lifecycle so validation, adapter selection,
            retry, rate gating, and metering remain centralized.

        Examples:
            Embed one text:
                ```python
                client = GenericEmbeddingClient(provider="dummy", model="test")
                vector = await client.embed_one("hello")
                ```

            Override the configured model:
                ```python
                client = GenericEmbeddingClient(provider="dummy", model="default")
                vector = await client.embed_one("hello", model="test")
                ```

        Args:
            self: Configured provider-neutral embedding client.
            text: Single text input.
            model: Optional per-call model override.
            **kw: Metering context and bounded provider request options forwarded
                to `embed`.

        Returns:
            list[float]: The single normalized embedding vector.

        Notes:
            This method intentionally shares the batch lifecycle and does not
            introduce a separate provider path.
        """

        res = await self.embed([text], model=model, **kw)
        return res[0]

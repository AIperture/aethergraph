# aethergraph/services/llm/embedding_client.py
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
import os
from typing import Any

import httpx

from aethergraph.config.config import EmbeddingUsageQuotaSettings
from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.core.runtime.runtime_metering import current_metering
from aethergraph.services.llm.adapters.embedding import (
    EmbeddingAdapterInvocation,
    invoke_embedding_adapter,
)
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.http_lifecycle import (
    _close_http_clients,
    _ensure_loop_http_client,
)
from aethergraph.services.llm.operation_quota import embedding_quota_ledger
from aethergraph.services.llm.operation_runtime import (
    OperationTraceProjection,
    execute_model_operation,
    model_operation_dimensions,
)
from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    ProviderRateGate,
    ProviderRetryExecutor,
    ProviderRetrySettings,
)
from aethergraph.services.llm.registry import provider_default_base_url, resolve_endpoint_adapter
from aethergraph.services.llm.types import EmbeddingResult, EmbeddingUsage
from aethergraph.services.llm.usage_metering import _record_embedding_metering


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
    operation_quota_cfg: EmbeddingUsageQuotaSettings | None = None
    default_dimensions: int | None = None
    profile_name: str | None = None

    def __post_init__(self) -> None:
        if self.default_dimensions is not None and self.default_dimensions < 1:
            raise ValueError("default embedding dimensions must be positive")
        resolved_provider = (
            self.provider or os.getenv("EMBED_PROVIDER") or os.getenv("LLM_PROVIDER") or "openai"
        ).lower()
        resolved_model = (
            self.model
            or os.getenv("EMBED_MODEL")
            or os.getenv("LLM_EMBED_MODEL")
            or "text-embedding-3-small"
        )
        self._apply_connection_state(
            provider=resolved_provider,
            model=resolved_model,
            base_url=self.base_url,
            api_key=self.api_key,
            azure_deployment=self.azure_deployment,
            timeout=self.timeout,
            retry_settings=self.retry_settings,
            rate_limit_group=self.rate_limit_group,
            endpoint_id=self.endpoint_id,
            rate_gate=self.rate_gate,
            retire_current=False,
        )
        self._client: httpx.AsyncClient | None = None
        self._retired_http_clients: list[httpx.AsyncClient] = []
        self._bound_loop = None
        self._operation_quota = embedding_quota_ledger(self.operation_quota_cfg)

    def _apply_connection_state(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None,
        api_key: str | None,
        azure_deployment: str | None,
        timeout: float,
        retry_settings: ProviderRetrySettings | None,
        rate_limit_group: str | None,
        endpoint_id: str | None,
        rate_gate: ProviderRateGate | None,
        retire_current: bool,
    ) -> None:
        resolved_provider = str(provider or "").strip().lower()
        try:
            resolved_endpoint = resolve_endpoint_adapter(
                resolved_provider,
                "embeddings",
                endpoint_id=endpoint_id,
            ).adapter_id
        except ValueError:
            if endpoint_id is not None:
                raise
            resolved_endpoint = None
        resolved_api_key = resolve_provider_credential(
            provider_id=resolved_provider,
            direct=api_key,
            secret_ref=None,
            secrets=None,
        ).value
        resolved_base_url = base_url or provider_default_base_url(resolved_provider) or ""
        resolved_deployment = azure_deployment or (
            os.getenv("AZURE_OPENAI_DEPLOYMENT") if resolved_provider == "azure" else None
        )
        retry_executor = ProviderRetryExecutor(
            retry_settings,
            rate_gate=rate_gate,
            base_url=resolved_base_url,
            credential=resolved_api_key,
        )
        current_client = getattr(self, "_client", None)
        self.provider = resolved_provider
        self.model = str(model or "").strip()
        self.endpoint_id = resolved_endpoint
        self.base_url = resolved_base_url
        self.api_key = resolved_api_key
        self.azure_deployment = resolved_deployment
        self.timeout = float(timeout)
        self.retry_settings = retry_settings
        self.rate_limit_group = rate_limit_group
        self.rate_gate = rate_gate
        self._provider_retry = retry_executor
        if retire_current and current_client is not None:
            self._retired_http_clients.append(current_client)
            self._client = None
            self._bound_loop = None

    def reconfigure_connection(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None,
        api_key: str | None,
        azure_deployment: str | None,
        timeout: float,
        retry_settings: ProviderRetrySettings | None = None,
        rate_limit_group: str | None = None,
        endpoint_id: str | None = None,
    ) -> None:
        """Replace the complete embedding connection while preserving identity.

        Intro:
            Validates one canonical endpoint and rebuilds retry/rate connection
            state before swapping the live binding used by future calls.

        Examples:
            Switch an OpenAI model:
                ```python
                client.reconfigure_connection(
                    provider="openai",
                    model="text-embedding-3-large",
                    base_url=None,
                    api_key=None,
                    azure_deployment=None,
                    timeout=60.0,
                )
                ```

            Pin an Azure embedding endpoint:
                ```python
                client.reconfigure_connection(
                    provider="azure",
                    model="embedding-model",
                    endpoint_id="azure_embeddings",
                    base_url="https://example.openai.azure.com",
                    api_key="secret",
                    azure_deployment="embedding-prod",
                    timeout=90.0,
                )
                ```

        Args:
            self: Configured embedding client retained by dependent services.
            provider: Registered provider identity.
            model: Provider embedding model identity.
            base_url: Optional provider API base URL override.
            api_key: Optional already-resolved provider credential.
            azure_deployment: Optional Azure deployment identity.
            timeout: HTTP request timeout in seconds.
            retry_settings: Optional bounded provider retry policy.
            rate_limit_group: Optional shared provider quota bucket.
            endpoint_id: Optional exact embedding endpoint adapter.

        Returns:
            None: Replaces connection-derived state atomically for future calls.

        Notes:
            Retired transports remain reachable until `aclose()` so in-flight
            operations are not forcibly interrupted.
        """

        self._apply_connection_state(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            azure_deployment=azure_deployment,
            timeout=timeout,
            retry_settings=retry_settings,
            rate_limit_group=rate_limit_group,
            endpoint_id=endpoint_id,
            rate_gate=self._provider_retry.rate_gate,
            retire_current=True,
        )

    # ------------ client management -----------------

    async def _ensure_client(self) -> None:
        """
        Ensure we have an httpx.AsyncClient bound to the *current* event loop.

        IMPORTANT: We do NOT try to aclose() a client created on a different loop,
        because httpx/anyio expects it to be closed on the same loop it was created on.
        """
        self._client, self._bound_loop, retired = _ensure_loop_http_client(
            self._client,
            self._bound_loop,
            timeout=self.timeout,
        )
        if retired is not None:
            self._retired_http_clients.append(retired)

    async def aclose(self) -> None:
        """Close active and safely retired embedding HTTP clients.

        Intro:
            Owns cleanup for transports retained across event-loop changes and
            atomic connection hot reloads.

        Examples:
            Close one embedding client:
                ```python
                await client.aclose()
                ```

            Close through the embedding service:
                ```python
                await service.aclose()
                ```

        Args:
            self: Embedding client owning transport resources.

        Returns:
            None: Closes every distinct reachable HTTP transport.

        Notes:
            Cross-loop close failures are logged and do not prevent remaining
            transports from being processed.
        """

        clients = [self._client, *self._retired_http_clients]
        self._retired_http_clients = []
        await _close_http_clients(
            clients,
            logger=logging.getLogger(__name__),
            warning_key="embedding_http_client_close_failed",
        )

    # ------------ public API ------------------------

    async def embed_result(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        **kw: Any,
    ) -> EmbeddingResult:
        """Embed one ordered batch and retain typed provider usage.

        Intro:
            Validates provider-neutral input, resolves one endpoint before each
            lifecycle execution, and retains provider usage for operation-specific
            metering before the vector-only facade projects it away.

        Examples:
            Embed one text:
                ```python
                client = GenericEmbeddingClient(provider="dummy", model="test")
                result = await client.embed_result(["hello"])
                ```

            Request a smaller output vector:
                ```python
                result = await client.embed_result(["hello"], dimensions=256)
                ```

        Args:
            self: Configured provider-neutral embedding client.
            texts: Ordered text inputs. An empty sequence returns immediately.
            model: Optional per-call model override.
            dimensions: Optional output-vector dimensionality overriding the
                canonical profile default.
            **kw: Metering context plus bounded provider request options such as
                `extra_body` and `azure_api_version`.

        Returns:
            EmbeddingResult: Ordered vectors and typed provider usage.

        Notes:
            Anthropic and DeepSeek embeddings remain explicitly unsupported.
            Physical adapters perform one attempt; this facade owns retries.
        """
        if not isinstance(texts, Sequence) or any(not isinstance(t, str) for t in texts):
            raise TypeError("embed(texts) expects Sequence[str]")
        if len(texts) == 0:
            return EmbeddingResult(vectors=[], usage=EmbeddingUsage.from_provider_usage(None))

        # Resolve model (override > configured)
        model = model or self.model or "text-embedding-3-small"
        requested_dimensions = dimensions if dimensions is not None else self.default_dimensions
        if requested_dimensions is not None and requested_dimensions < 1:
            raise ValueError("embedding dimensions must be positive")
        extra_body = dict(kw.get("extra_body") or {})
        if requested_dimensions is not None and "dimensions" in extra_body:
            raise ValueError(
                "embedding dimensions must use the canonical dimensions argument, not both "
                "dimensions and extra_body['dimensions']"
            )

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
            dimensions=requested_dimensions,
            azure_deployment=self.azure_deployment,
            azure_api_version=kw.get("azure_api_version"),
            extra_body=extra_body,
        )

        async def _attempt() -> ProviderCallResult[EmbeddingResult]:
            return await invoke_embedding_adapter(
                self,
                adapter_id=adapter.adapter_id,
                invocation=invocation,
            )

        meter_dimensions = model_operation_dimensions(
            profile_name=self.profile_name,
            overrides={key: kw.get(key) for key in ("user_id", "org_id", "run_id", "graph_id")},
        )

        async def _account(result: EmbeddingResult, latency_ms: int) -> None:
            await _record_embedding_metering(
                self.metering or current_metering(),
                provider=self.provider,
                model=model,
                usage=result.usage,
                num_texts=len(texts),
                latency_ms=latency_ms,
                dimensions=meter_dimensions,
                logger=logging.getLogger(__name__),
            )

        return await execute_model_operation(
            self,
            model=model,
            provider_operation="embedding",
            requested_quota={"calls": 1, "texts": len(invocation.texts)},
            attempt=_attempt,
            actual_quota=lambda result: {
                "calls": 1,
                "texts": len(invocation.texts),
                "input_tokens": result.usage.input_tokens,
            },
            usage_payload=lambda result: result.usage.to_dict(),
            account_usage=_account,
            trace=OperationTraceProjection(
                service="embedding",
                operation="embed",
                request={
                    "provider": self.provider,
                    "model": model,
                    "endpoint_id": adapter.adapter_id,
                    "num_texts": len(texts),
                    "output_dimensions": requested_dimensions,
                },
                tags=("model", "embedding"),
                response=lambda result: {
                    "num_vectors": len(result.vectors),
                    "usage": result.usage.to_dict(),
                },
                metrics=lambda result: {
                    "num_texts": len(texts),
                    "num_vectors": len(result.vectors),
                    "input_tokens": result.usage.input_tokens,
                },
            ),
            dimensions=meter_dimensions,
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int | None = None,
        **kw: Any,
    ) -> list[list[float]]:
        """Embed one ordered batch through the compatibility vector facade.

        Intro:
            Delegates to `embed_result()` so provider usage is retained and
            metered once before returning the historical vector-only value.

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
            dimensions: Optional output-vector dimensionality overriding the
                canonical profile default.
            **kw: Metering context plus bounded provider request options.

        Returns:
            list[list[float]]: Ordered embedding vectors matching the input count.

        Notes:
            This public compatibility projection does not discard usage until
            after the operation-specific meter has consumed it.
        """

        return (
            await self.embed_result(
                texts,
                model=model,
                dimensions=dimensions,
                **kw,
            )
        ).vectors

    async def embed_one(
        self,
        text: str,
        *,
        model: str | None = None,
        dimensions: int | None = None,
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
            dimensions: Optional output-vector dimensionality overriding the
                canonical profile default.
            **kw: Metering context and bounded provider request options forwarded
                to `embed`.

        Returns:
            list[float]: The single normalized embedding vector.

        Notes:
            This method intentionally shares the batch lifecycle and does not
            introduce a separate provider path.
        """

        res = await self.embed(
            [text],
            model=model,
            dimensions=dimensions,
            **kw,
        )
        return res[0]

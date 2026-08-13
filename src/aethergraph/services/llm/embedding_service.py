# aethergraph/services/llm/embedding_service.py
from __future__ import annotations

from collections.abc import Mapping, Sequence

from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.services.llm.generic_embed_client import GenericEmbeddingClient


class EmbeddingService:
    def __init__(self, clients: Mapping[str, EmbeddingClientProtocol]):
        self._clients = dict(clients)

    def get(self, name: str = "default") -> EmbeddingClientProtocol:
        """Return one exact configured embedding client.

        Intro:
            Resolves a stable named client without creating a fallback profile.

        Examples:
            Read the default client:
                ```python
                client = service.get()
                ```

            Read a named client:
                ```python
                client = service.get("search")
                ```

        Args:
            self: Container-owned embedding service.
            name: Exact configured profile name.

        Returns:
            EmbeddingClientProtocol: Stable configured client identity.

        Notes:
            Missing profiles raise `KeyError`; no profile fallback is performed.
        """

        return self._clients[name]

    async def embed(
        self,
        texts: Sequence[str],
        *,
        profile: str = "default",
        model: str | None = None,
        **kw,
    ) -> list[list[float]]:
        """Embed one text batch through a named profile.

        Intro:
            Delegates to the selected client so its canonical endpoint, retry,
            rate gate, transport, and metering lifecycle remain authoritative.

        Examples:
            Embed with the default profile:
                ```python
                vectors = await service.embed(["hello"])
                ```

            Embed with a named profile and model override:
                ```python
                vectors = await service.embed(
                    ["north", "south"],
                    profile="search",
                    model="embedding-model",
                )
                ```

        Args:
            self: Container-owned embedding service.
            texts: Ordered text inputs.
            profile: Exact configured profile name.
            model: Optional per-call model override.
            **kw: Metering context and bounded adapter request options.

        Returns:
            list[list[float]]: Ordered embedding vectors.

        Notes:
            This service does not add retries, fallback, or provider dispatch.
        """

        client = self.get(profile)
        return await client.embed(texts, model=model, **kw)

    async def aclose(self) -> None:
        """Close transport resources for all configured embedding clients.

        Intro:
            Visits each distinct client once and invokes its optional asynchronous
            close boundary.

        Examples:
            Close at application shutdown:
                ```python
                await service.aclose()
                ```

            Close an empty service safely:
                ```python
                await EmbeddingService({}).aclose()
                ```

        Args:
            self: Container-owned embedding service.

        Returns:
            None: Completes after every close-capable client is processed.

        Notes:
            Client-specific cross-loop close behavior remains client-owned.
        """

        seen: set[int] = set()
        for client in self._clients.values():
            if id(client) in seen:
                continue
            seen.add(id(client))
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()

    # --- Runtime profile helpers ---------------------------------
    def configure_profile(
        self,
        name: str = "default",
        *,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float | None = None,
    ) -> EmbeddingClientProtocol:
        """Create or atomically reconfigure an in-memory embedding profile.

        Intro:
            New profiles inherit shared infrastructure controls from the default
            client. Existing profiles preserve object identity while replacing
            all connection-derived endpoint, retry, credential, and transport
            state together.

        Examples:
            Create a named local profile:
                ```python
                client = service.configure_profile(
                    name="local",
                    provider="ollama",
                    model="nomic-embed-text",
                )
                ```

            Switch an existing profile to Azure:
                ```python
                client = service.configure_profile(
                    name="default",
                    provider="azure",
                    model="embedding-model",
                    base_url="https://example.openai.azure.com",
                    api_key="secret",
                    azure_deployment="embedding-prod",
                    timeout=90.0,
                )
                ```

        Args:
            self: Container-owned embedding service.
            name: Profile name to create or update.
            provider: Optional registered provider identity.
            model: Optional embedding model identity.
            base_url: Optional provider API base URL override.
            api_key: Optional in-memory provider credential override.
            azure_deployment: Optional Azure deployment identity.
            timeout: Optional HTTP request timeout in seconds.

        Returns:
            EmbeddingClientProtocol: New or identity-preserved configured client.

        Notes:
            This method does not persist settings. Provider changes with no base
            URL or key re-resolve the new provider's registry/environment values.
        """
        if name not in self._clients:
            template = self._clients.get("default")
            client = GenericEmbeddingClient(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key,
                azure_deployment=azure_deployment,
                timeout=timeout or 60.0,
                retry_settings=getattr(template, "retry_settings", None),
                rate_limit_group=getattr(template, "rate_limit_group", None),
                rate_gate=getattr(getattr(template, "_provider_retry", None), "rate_gate", None),
                metering=getattr(template, "metering", None),
                operation_quota_cfg=getattr(template, "operation_quota_cfg", None),
                default_dimensions=getattr(template, "default_dimensions", None),
            )
            self._clients[name] = client
            return client

        c = self._clients[name]
        if not isinstance(c, GenericEmbeddingClient):
            raise TypeError("Configured embedding client does not support profile reconfiguration")
        provider_changed = provider is not None and provider != c.provider
        c.reconfigure_connection(
            provider=provider or c.provider,
            model=model or c.model,
            base_url=(None if provider_changed else c.base_url) if base_url is None else base_url,
            api_key=(None if provider_changed else c.api_key) if api_key is None else api_key,
            azure_deployment=(
                (None if provider_changed else c.azure_deployment)
                if azure_deployment is None
                else azure_deployment
            ),
            timeout=timeout if timeout is not None else c.timeout,
            retry_settings=c.retry_settings,
            rate_limit_group=c.rate_limit_group,
            endpoint_id=None if provider_changed else c.endpoint_id,
        )
        return c

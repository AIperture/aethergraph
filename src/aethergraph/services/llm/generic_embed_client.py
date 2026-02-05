# aethergraph/services/llm/embedding_client.py
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import os
from typing import Any

import httpx

from aethergraph.contracts.services.llm import EmbeddingClientProtocol
from aethergraph.services.llm.generic_client import _Retry
from aethergraph.services.metering.eventlog_metering import MeteringService


@dataclass
class GenericEmbeddingClient(EmbeddingClientProtocol):
    """
    Provider-agnostic embedding client.

    provider: one of {"openai","azure","anthropic","google","openrouter","lmstudio","ollama","dummy"}

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

    # metering (optional, can be None)
    metering: MeteringService | None = None

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

        # Pick an API key from provider-specific envs (or explicit api_key)
        if self.api_key is None:
            self.api_key = (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("AZURE_OPENAI_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
            )

        # Base URL defaults per provider
        if self.base_url is None:
            self.base_url = {
                "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "azure": os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
                "anthropic": "https://api.anthropic.com",
                "google": "https://generativelanguage.googleapis.com",
                "openrouter": "https://openrouter.ai/api/v1",
                "lmstudio": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                "dummy": "http://localhost:8745",  # for tests
            }[self.provider]

        # Azure deployment (for /deployments/{name}/embeddings)
        if self.provider == "azure" and self.azure_deployment is None:
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self._retry = _Retry()
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
        """
        Provider-agnostic batch embedding.
        """
        await self._ensure_client()
        assert self._client is not None

        if not isinstance(texts, Sequence) or any(not isinstance(t, str) for t in texts):
            raise TypeError("embed(texts) expects Sequence[str]")
        if len(texts) == 0:
            return []

        # Resolve model (override > configured)
        model = model or self.model or "text-embedding-3-small"

        # Dispatch by provider
        if self.provider in {"openai", "openrouter", "lmstudio", "ollama"}:
            embs = await self._embed_openai_like(texts, model=model, **kw)
        elif self.provider == "azure":
            embs = await self._embed_azure(texts, model=model, **kw)
        elif self.provider == "google":
            embs = await self._embed_google(texts, model=model, **kw)
        elif self.provider == "anthropic":
            raise NotImplementedError("Embeddings not supported for anthropic")
        elif self.provider == "dummy":
            embs = await self._embed_dummy(texts, model=model, **kw)
        else:  # pragma: no cover
            raise NotImplementedError(f"Unknown embedding provider: {self.provider}")

        # ---- metering hook (placeholder) ----
        if self.metering is not None:
            try:
                # TODO: compute token estimates or bytes; for now just count inputs
                await self.metering.record_embedding(
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
        res = await self.embed([text], model=model, **kw)
        return res[0]

    # ------------ provider-specific helpers ------------------------

    def _headers_openai_like(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _embed_openai_like(
        self,
        texts: Sequence[str],
        *,
        model: str,
        **kw: Any,
    ) -> list[list[float]]:
        assert self._client is not None
        url = f"{self.base_url}/embeddings"
        headers = self._headers_openai_like()
        extra_body: dict[str, Any] = kw.get("extra_body") or {}

        body: dict[str, Any] = {
            "model": model,
            "input": list(texts),
        }
        body.update(extra_body)

        def parse(data: dict[str, Any]) -> list[list[float]]:
            items = data.get("data", []) or []
            embs = [d.get("embedding") for d in items]
            if len(embs) != len(texts) or any(e is None for e in embs):
                raise RuntimeError(
                    f"Embeddings response shape mismatch: got {len(embs)} items for {len(texts)} inputs"
                )
            return embs  # type: ignore[return-value]

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Embeddings request failed ({e.response.status_code}): {e.response.text}"
                ) from e
            return parse(r.json())

        return await self._retry.run(_call)

    async def _embed_azure(
        self,
        texts: Sequence[str],
        *,
        model: str,
        **kw: Any,
    ) -> list[list[float]]:
        if not self.azure_deployment:
            raise RuntimeError(
                "Azure embeddings requires AZURE_OPENAI_DEPLOYMENT (azure_deployment)"
            )

        assert self._client is not None

        azure_api_version = kw.get("azure_api_version") or "2024-08-01-preview"
        extra_body: dict[str, Any] = kw.get("extra_body") or {}

        url = (
            f"{self.base_url}/openai/deployments/"
            f"{self.azure_deployment}/embeddings?api-version={azure_api_version}"
        )
        headers = {"api-key": self.api_key or "", "Content-Type": "application/json"}
        body: dict[str, Any] = {"input": list(texts)}
        # Some Azure flavors accept model/dimensions; keep flexible
        if model:
            body["model"] = model
        body.update(extra_body)

        def parse(data: dict[str, Any]) -> list[list[float]]:
            items = data.get("data", []) or []
            embs = [d.get("embedding") for d in items]
            if len(embs) != len(texts) or any(e is None for e in embs):
                raise RuntimeError(
                    f"Azure embeddings response shape mismatch: got {len(embs)} items for {len(texts)} inputs"
                )
            return embs  # type: ignore[return-value]

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Embeddings request failed ({e.response.status_code}): {e.response.text}"
                ) from e
            return parse(r.json())

        return await self._retry.run(_call)

    async def _embed_google(
        self,
        texts: Sequence[str],
        *,
        model: str,
        **kw: Any,
    ) -> list[list[float]]:
        assert self._client is not None
        base = self.base_url.rstrip("/")
        api_key = self.api_key or os.getenv("GOOGLE_API_KEY") or ""
        headers = {"Content-Type": "application/json"}

        # v1 and v1beta endpoints
        batch_url_v1 = f"{base}/v1/models/{model}:batchEmbedContents?key={api_key}"
        embed_url_v1 = f"{base}/v1/models/{model}:embedContent?key={api_key}"
        batch_url_v1beta = f"{base}/v1beta/models/{model}:batchEmbedContents?key={api_key}"
        embed_url_v1beta = f"{base}/v1beta/models/{model}:embedContent?key={api_key}"

        def parse_single(data: dict[str, Any]) -> list[float]:
            return (data.get("embedding") or {}).get("values") or []

        def parse_batch(data: dict[str, Any]) -> list[list[float]]:
            embs: list[list[float]] = []
            for e in data.get("embeddings") or []:
                embs.append((e or {}).get("values") or [])
            if len(embs) != len(texts):
                raise RuntimeError(
                    f"Gemini batch embeddings mismatch: got {len(embs)} for {len(texts)}"
                )
            return embs

        async def try_batch(url: str) -> list[list[float]] | None:
            body = {"requests": [{"content": {"parts": [{"text": t}]}} for t in texts]}
            r = await self._client.post(url, headers=headers, json=body)
            if r.status_code in (400, 404):
                return None
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Gemini batchEmbedContents failed ({e.response.status_code}): {e.response.text}"
                ) from e
            return parse_batch(r.json())

        async def call_single(url: str) -> list[list[float]]:
            out: list[list[float]] = []
            for t in texts:
                r = await self._client.post(
                    url, headers=headers, json={"content": {"parts": [{"text": t}]}}
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(
                        f"Gemini embedContent failed ({e.response.status_code}): {e.response.text}"
                    ) from e
                out.append(parse_single(r.json()))
            if len(out) != len(texts):
                raise RuntimeError(f"Gemini embeddings mismatch: got {len(out)} for {len(texts)}")
            return out

        async def _call():
            # try v1 batch, then v1beta batch, then single
            res = await try_batch(batch_url_v1)
            if res is not None:
                return res
            res = await try_batch(batch_url_v1beta)
            if res is not None:
                return res
            try:
                return await call_single(embed_url_v1)
            except RuntimeError:
                return await call_single(embed_url_v1beta)

        return await self._retry.run(_call)

    async def _embed_dummy(
        self,
        texts: Sequence[str],
        *,
        model: str,
        **kw: Any,
    ) -> list[list[float]]:
        """
        Dummy provider for tests: returns [len(text)] as a 1D "embedding".
        """
        return [[float(len(t))] for t in texts]

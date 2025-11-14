from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from aethergraph.contracts.services.llm import LLMClientProtocol


# ---- Helpers --------------------------------------------------------------
class _Retry:
    def __init__(self, tries=4, base=0.5, cap=8.0):
        self.tries, self.base, self.cap = tries, base, cap

    async def run(self, fn, *a, **k):
        exc = None
        for i in range(self.tries):
            try:
                return await fn(*a, **k)
            except (httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPStatusError) as e:
                exc = e
                await asyncio.sleep(min(self.cap, self.base * (2**i)))
        raise exc


def _first_text(choices):
    """Extract text and usage from OpenAI-style choices list."""
    if not choices:
        return "", {}
    c = choices[0]
    text = (c.get("message", {}) or {}).get("content") or c.get("text") or ""
    usage = {}
    return text, usage


# ---- Generic client -------------------------------------------------------
class GenericLLMClient(LLMClientProtocol):
    """
    provider: one of {"openai","azure","anthropic","google","openrouter","lmstudio","ollama"}
    Configuration (read from env by default, but you can pass in):
      - OPENAI_API_KEY / OPENAI_BASE_URL
      - AZURE_OPENAI_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT
      - ANTHROPIC_API_KEY
      - GOOGLE_API_KEY
      - OPENROUTER_API_KEY
      - LMSTUDIO_BASE_URL (defaults http://localhost:1234/v1)
      - OLLAMA_BASE_URL   (defaults http://localhost:11434/v1)
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float = 60.0,
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.embed_model = embed_model or os.getenv("EMBED_MODEL") or "text-embedding-3-small"
        self._retry = _Retry()
        self._client = httpx.AsyncClient(timeout=timeout)
        self._bound_loop = None

        # Resolve creds/base
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )

        self.base_url = (
            base_url
            or {
                "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "azure": os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
                "anthropic": "https://api.anthropic.com",
                "google": "https://generativelanguage.googleapis.com",
                "openrouter": "https://openrouter.ai/api/v1",
                "lmstudio": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            }[self.provider]
        )
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

    async def _ensure_client(self):
        """Ensure the httpx client is bound to the current event loop.
        This allows safe usage across multiple async contexts.
        """
        loop = asyncio.get_running_loop()
        if self._client is None or self._bound_loop != loop:
            # close old client if any
            if self._client is not None:
                try:
                    await self._client.aclose()
                except Exception:
                    logger = logging.getLogger("aethergraph.services.llm.generic_client")
                    logger.warning("llm_client_close_failed")
            self._client = httpx.AsyncClient(timeout=self._client.timeout)
            self._bound_loop = loop

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        model = kw.get("model", self.model)

        if self.provider != "openai":
            # Make sure _chat_by_provider ALSO returns (str, usage),
            # or wraps provider-specific structures into text.
            return await self._chat_by_provider(messages, **kw)

        body: dict[str, Any] = {
            "model": model,
            "input": messages,
        }
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        temperature = kw.get("temperature")
        top_p = kw.get("top_p")
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p

        async def _call():
            r = await self._client.post(
                f"{self.base_url}/responses",
                headers=self._headers_openai_like(),
                json=body,
            )

            try:
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"OpenAI Responses API error: {e.response.text}") from e

            data = r.json()
            output = data.get("output")
            txt = ""

            # NEW: handle list-of-messages shape
            if isinstance(output, list) and output:
                first = output[0]
                if isinstance(first, dict) and first.get("type") == "message":
                    parts = first.get("content") or []
                    chunks: list[str] = []
                    for p in parts:
                        if "text" in p:
                            chunks.append(p["text"])
                    txt = "".join(chunks)

            elif isinstance(output, dict) and output.get("type") == "message":
                msg = output.get("message") or output
                parts = msg.get("content") or []
                chunks: list[str] = []
                for p in parts:
                    if "text" in p:
                        chunks.append(p["text"])
                txt = "".join(chunks)

            elif isinstance(output, str):
                txt = output

            else:
                txt = str(output) if output is not None else ""

            usage = data.get("usage", {}) or {}
            return txt, usage

        return await self._retry.run(_call)

    # ---------------- Chat ----------------
    async def _chat_by_provider(
        self, messages: list[dict[str, Any]], **kw
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)
        model = kw.get("model", self.model)

        if self.provider in {"openrouter", "lmstudio", "ollama"}:

            async def _call():
                body = {
                    "model": model,
                    "messages": messages,
                }

                r = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers_openai_like(),
                    json=body,
                )

                try:
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    raise RuntimeError(f"OpenAI Responses API error: {e.response.text}") from e
                data = r.json()
                txt, _ = _first_text(data.get("choices", []))
                return txt, data.get("usage", {}) or {}

            return await self._retry.run(_call)

        if self.provider == "azure":
            if not (self.base_url and self.azure_deployment):
                raise RuntimeError(
                    "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
                )

            async def _call():
                r = await self._client.post(
                    f"{self.base_url}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2024-08-01-preview",
                    headers={"api-key": self.api_key, "Content-Type": "application/json"},
                    json={"messages": messages, "temperature": temperature, "top_p": top_p},
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    raise RuntimeError(f"OpenAI Responses API error: {e.response.text}") from e

                data = r.json()
                txt, _ = _first_text(data.get("choices", []))
                return txt, data.get("usage", {}) or {}

            return await self._retry.run(_call)

        if self.provider == "anthropic":
            # Convert OpenAI-style messages -> Anthropic Messages API format
            # 1) Collect system messages (as strings)
            sys_msgs = [m["content"] for m in messages if m["role"] == "system"]

            # 2) Convert non-system messages into Anthropic blocks
            conv = []
            for m in messages:
                role = m["role"]
                if role == "system":
                    continue  # handled via `system` field

                # Anthropic only accepts "user" or "assistant"
                anthro_role = "assistant" if role == "assistant" else "user"

                content = m["content"]
                # Wrap string content into text blocks; if caller is already giving blocks, pass them through.
                if isinstance(content, str):
                    content_blocks = [{"type": "text", "text": content}]
                else:
                    # Assume caller knows what they're doing for multimodal content
                    content_blocks = content

                conv.append({"role": anthro_role, "content": content_blocks})

            # 3) Build payload
            payload = {
                "model": model,
                "max_tokens": kw.get("max_tokens", 1024),
                "messages": conv,
                "temperature": temperature,
                "top_p": top_p,
            }

            # ✅ Anthropic  v1/messages now expects `system` to be a list
            if sys_msgs:
                payload["system"] = "\n\n".join(sys_msgs)

            async def _call():
                r = await self._client.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    # keep the nice debugging message
                    raise RuntimeError(f"Anthropic API error: {e.response.text}") from e

                data = r.json()
                # data["content"] is a list of blocks
                blocks = data.get("content") or []
                txt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                return txt, data.get("usage", {}) or {}

            return await self._retry.run(_call)

        if self.provider == "google":
            # Merge system messages into a single preamble
            system = "\n".join([m["content"] for m in messages if m["role"] == "system"])

            # Non-system messages
            turns = [
                {
                    "role": "user" if m["role"] == "user" else "model",
                    "parts": [{"text": m["content"]}],
                }
                for m in messages
                if m["role"] != "system"
            ]

            if system:
                turns.insert(
                    0,
                    {
                        "role": "user",
                        "parts": [{"text": f"System instructions: {system}"}],
                    },
                )

            async def _call():
                payload = {
                    "contents": turns,
                    "generationConfig": {
                        "temperature": temperature,
                        "topP": top_p,
                    },
                }

                r = await self._client.post(
                    f"{self.base_url}/v1/models/{model}:generateContent?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(
                        f"Gemini generateContent failed ({e.response.status_code}): {e.response.text}"
                    ) from e

                data = r.json()
                cand = (data.get("candidates") or [{}])[0]
                txt = "".join(
                    p.get("text", "") for p in (cand.get("content", {}).get("parts") or [])
                )
                return txt, {}  # usage parsing optional

            return await self._retry.run(_call)

        if self.provider == "openai":
            raise RuntimeError(
                "Internal error: OpenAI provider should use chat() or responses_chat() directly."
            )

        raise NotImplementedError(f"provider {self.provider}")

    # ---------------- Embeddings ----------------
    async def embed(self, texts: list[str], **kw) -> list[list[float]]:
        # model override order: kw > self.embed_model > ENV > default
        await self._ensure_client()

        model = (
            kw.get("model")
            or self.embed_model
            or os.getenv("EMBED_MODEL")
            or "text-embedding-3-small"
        )

        if self.provider in {"openai", "openrouter", "lmstudio", "ollama"}:

            async def _call():
                r = await self._client.post(
                    f"{self.base_url}/embeddings",
                    headers=self._headers_openai_like(),
                    json={"model": model, "input": texts},
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    # Log or re-raise with more context
                    msg = f"Embeddings request failed ({e.response.status_code}): {e.response.text}"
                    raise RuntimeError(msg) from e

                data = r.json()
                return [d["embedding"] for d in data.get("data", [])]

            return await self._retry.run(_call)

        if self.provider == "azure":

            async def _call():
                r = await self._client.post(
                    f"{self.base_url}/openai/deployments/{self.azure_deployment}/embeddings?api-version=2024-08-01-preview",
                    headers={"api-key": self.api_key, "Content-Type": "application/json"},
                    json={"input": texts},
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    # Log or re-raise with more context
                    msg = f"Embeddings request failed ({e.response.status_code}): {e.response.text}"
                    raise RuntimeError(msg) from e

                data = r.json()
                return [d["embedding"] for d in data.get("data", [])]

            return await self._retry.run(_call)

        if self.provider == "google":

            async def _call():
                r = await self._client.post(
                    f"{self.base_url}/v1/models/{model}:embedContent?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json={"content": {"parts": [{"text": "\n".join(texts)}]}},
                )
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise RuntimeError(
                        f"Gemini embedContent failed ({e.response.status_code}): {e.response.text}"
                    ) from e

                data = r.json()
                return [data.get("embedding", {}).get("values", [])]

            return await self._retry.run(_call)

        # Anthropic: no embeddings endpoint
        raise NotImplementedError(f"Embeddings not supported for {self.provider}")

    # ---------------- Internals ----------------
    def _headers_openai_like(self):
        hdr = {"Content-Type": "application/json"}
        if self.provider in {"openai", "openrouter"}:
            hdr["Authorization"] = f"Bearer {self.api_key}"
        return hdr

    async def aclose(self):
        await self._client.aclose()

    def _default_headers_for_raw(self) -> dict[str, str]:
        hdr = {"Content-Type": "application/json"}

        if self.provider in {"openai", "openrouter"}:
            if self.api_key:
                hdr["Authorization"] = f"Bearer {self.api_key}"
            else:
                raise RuntimeError("OpenAI/OpenRouter requires an API key for raw() calls.")

        elif self.provider == "anthropic":
            if self.api_key:
                hdr.update(
                    {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                    }
                )
            else:
                raise RuntimeError("Anthropic requires an API key for raw() calls.")

        elif self.provider == "azure":
            if self.api_key:
                hdr["api-key"] = self.api_key
            else:
                raise RuntimeError("Azure OpenAI requires an API key for raw() calls.")

        # For google, lmstudio, ollama we usually put keys in the URL or
        # they’re local; leave headers minimal unless user overrides.
        return hdr

    async def raw(
        self,
        *,
        method: str = "POST",
        path: str | None = None,
        url: str | None = None,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        return_response: bool = False,
    ) -> Any:
        """
        Low-level escape hatch: send a raw HTTP request using this client’s
        base_url, auth, and retry logic.

        - If `url` is provided, it is used as-is.
        - Otherwise, `path` is joined to `self.base_url`.
        - `json` and `params` are forwarded to httpx.
        - Provider-specific default headers (auth, version, etc.) are applied,
          then overridden by `headers` if provided.

        Returns:
          - r.json() by default
          - or the raw `httpx.Response` if `return_response=True`
        """
        await self._ensure_client()

        if not url and not path:
            raise ValueError("Either `url` or `path` must be provided to raw().")

        if not url:
            url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

        base_headers = self._default_headers_for_raw()
        if headers:
            base_headers.update(headers)

        async def _call():
            r = await self._client.request(
                method=method,
                url=url,
                headers=base_headers,
                json=json,
                params=params,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"{self.provider} raw API error ({e.response.status_code}): {e.response.text}"
                ) from e

            return r if return_response else r.json()

        return await self._retry.run(_call)


# Convenience factory
def llm_from_env() -> GenericLLMClient:
    return GenericLLMClient()

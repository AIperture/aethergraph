from __future__ import annotations

import asyncio
import json
import logging
import os

# from time import time
import time
from typing import Any

import httpx

from aethergraph.config.config import RateLimitSettings
from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    GeneratedImage,
    ImageFormat,
    ImageGenerationResult,
    ImageResponseFormat,
    LLMUnsupportedFeatureError,
)
from aethergraph.services.llm.utils import (
    _azure_images_generations_url,
    _data_url_to_b64_and_mime,
    _ensure_system_json_directive,
    _extract_json_text,
    _guess_mime_from_format,
    _is_data_url,
    _normalize_base_url_no_trailing_slash,
    _normalize_openai_responses_input,
    _to_anthropic_blocks,
    _to_gemini_parts,
    _validate_json_schema,
)


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
        # metering
        metering: MeteringService | None = None,
        # rate limit
        rate_limit_cfg: RateLimitSettings | None = None,
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

        self.metering = metering

        # Rate limit settings
        self._rate_limit_cfg = rate_limit_cfg
        self._per_run_calls: dict[str, int] = {}
        self._per_run_tokens: dict[str, int] = {}

    # ---------------- internal helpers for metering ----------------
    @staticmethod
    def _normalize_usage(usage: dict[str, Any]) -> tuple[int, int]:
        """Normalize usage dict to standard keys: prompt_tokens, completion_tokens."""
        if not usage:
            return 0, 0

        prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
        completion = usage.get("completion_tokens") or usage.get("output_tokens")

        try:
            prompt_i = int(prompt) if prompt is not None else 0
        except (ValueError, TypeError):
            prompt_i = 0
        try:
            completion_i = int(completion) if completion is not None else 0
        except (ValueError, TypeError):
            completion_i = 0

        return prompt_i, completion_i

    def _get_rate_limit_cfg(self) -> RateLimitSettings | None:
        if self._rate_limit_cfg is not None:
            return self._rate_limit_cfg
        # Lazy-load from container if available
        try:
            from aethergraph.core.runtime.runtime_services import (
                current_services,  # local import to avoid cycles
            )

            container = current_services()
            settings = getattr(container, "settings", None)
            if settings is not None and getattr(settings, "rate_limit", None) is not None:
                self._rate_limit_cfg = settings.rate_limit
                return self._rate_limit_cfg
        except Exception:
            pass

    def _enforce_llm_limits_for_run(self, *, usage: dict[str, Any]) -> None:
        cfg = self._get_rate_limit_cfg()
        if cfg is None or not cfg.enabled:
            return

        # get current run_id from context
        ctx = current_meter_context.get()
        run_id = ctx.get("run_id")
        if not run_id:
            # no run_id context; cannot enforce per-run limits
            return

        prompt_tokens, completion_tokens = self._normalize_usage(usage)
        total_tokens = prompt_tokens + completion_tokens

        calls = self._per_run_calls.get(run_id, 0) + 1
        tokens = self._per_run_tokens.get(run_id, 0) + total_tokens

        # store updated counts
        self._per_run_calls[run_id] = calls
        self._per_run_tokens[run_id] = tokens

        if cfg.max_llm_calls_per_run and calls > cfg.max_llm_calls_per_run:
            raise RuntimeError(
                f"LLM call limit exceeded for this run "
                f"({calls} > {cfg.max_llm_calls_per_run}). "
                "Consider simplifying the graph or raising the limit."
            )

        if cfg.max_llm_tokens_per_run and tokens > cfg.max_llm_tokens_per_run:
            raise RuntimeError(
                f"LLM token limit exceeded for this run "
                f"({tokens} > {cfg.max_llm_tokens_per_run}). "
                "Consider simplifying the graph or raising the limit."
            )

    async def _record_llm_usage(
        self,
        *,
        model: str,
        usage: dict[str, Any],
        latency_ms: int | None = None,
    ) -> None:
        self.metering = self.metering or current_metering()
        prompt_tokens, completion_tokens = self._normalize_usage(usage)
        ctx = current_meter_context.get()
        user_id = ctx.get("user_id")
        org_id = ctx.get("org_id")
        run_id = ctx.get("run_id")

        try:
            await self.metering.record_llm(
                user_id=user_id,
                org_id=org_id,
                run_id=run_id,
                model=model,
                provider=self.provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        except Exception as e:
            # Never fail the LLM call due to metering issues
            logger = logging.getLogger("aethergraph.services.llm.generic_client")
            logger.warning(f"llm_metering_failed: {e}")

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
        output_format: ChatOutputFormat = "text",
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "output",
        strict_schema: bool = True,
        validate_json: bool = True,
        fail_on_unsupported: bool = True,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        """
        Send a chat request to the LLM provider and return the response in a normalized format.
        This method handles provider-specific dispatch, output postprocessing,
        rate limiting, and usage metering. It supports structured output via JSON schema
        validation and flexible output formats.

        Examples:
            Basic usage with a list of messages:
            ```python
            response, usage = await context.llm().chat([
                {"role": "user", "content": "Hello, assistant!"}
            ])
            ```

            Requesting structured output with a JSON schema:
            ```python
            response, usage = await context.llm().chat(
                messages=[{"role": "user", "content": "Summarize this text."}],
                output_format="json",
                json_schema={"type": "object", "properties": {"summary": {"type": "string"}}}
            ```

        Args:
            messages: List of message dicts, each with "role" and "content" keys.
            reasoning_effort: Optional string to control model reasoning depth.
            max_output_tokens: Optional maximum number of output tokens.
            output_format: Output format, e.g., "text" or "json".
            json_schema: Optional JSON schema for validating structured output.
            schema_name: Name for the root schema object (default: "output").
            strict_schema: If True, enforce strict schema validation.
            validate_json: If True, validate JSON output against schema.
            fail_on_unsupported: If True, raise error for unsupported features.
            **kw: Additional provider-specific keyword arguments.

        Returns:
            tuple[str, dict[str, int]]: The model response (text or structured output) and usage statistics.

        Raises:
            NotImplementedError: If the provider is not supported.
            RuntimeError: For various errors including invalid JSON output or rate limit violations.
            LLMUnsupportedFeatureError: If a requested feature is unsupported by the provider.

        Notes:
            - This method centralizes handling of different LLM providers, ensuring consistent behavior.
            - Structured output support allows for robust integration with downstream systems.
            - Rate limiting and metering help manage resource usage effectively.
        """
        await self._ensure_client()
        model = kw.get("model", self.model)

        start = time.perf_counter()

        # Provider-specific call (now symmetric)
        text, usage = await self._chat_dispatch(
            messages,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            fail_on_unsupported=fail_on_unsupported,
            **kw,
        )

        # JSON postprocessing/validation is centralized here (consistent behavior)
        text = self._postprocess_structured_output(
            text=text,
            output_format=output_format,
            json_schema=json_schema,
            strict_schema=strict_schema,
            validate_json=validate_json,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Enforce rate limits (existing)
        self._enforce_llm_limits_for_run(usage=usage)

        # Metering (existing)
        await self._record_llm_usage(
            model=model,
            usage=usage,
            latency_ms=latency_ms,
        )

        return text, usage

    async def _chat_dispatch(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
        validate_json: bool,
        fail_on_unsupported: bool,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        # OpenAI is now symmetric too
        if self.provider == "openai":
            return await self._chat_openai_responses(
                messages,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                output_format=output_format,
                json_schema=json_schema,
                schema_name=schema_name,
                strict_schema=strict_schema,
            )

        # Everyone else
        if self.provider in {"openrouter", "lmstudio", "ollama"}:
            return await self._chat_openai_like_chat_completions(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                **kw,
            )

        if self.provider == "azure":
            return await self._chat_azure_chat_completions(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                **kw,
            )

        if self.provider == "anthropic":
            return await self._chat_anthropic_messages(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                **kw,
            )

        if self.provider == "google":
            return await self._chat_gemini_generate_content(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                **kw,
            )

        raise NotImplementedError(f"provider {self.provider}")

    def _postprocess_structured_output(
        self,
        *,
        text: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        strict_schema: bool,
        validate_json: bool,
    ) -> str:
        if output_format not in ("json_object", "json_schema"):
            return text

        if not validate_json:
            return text

        json_text = _extract_json_text(text)
        try:
            obj = json.loads(json_text)
        except Exception as e:
            raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{text}") from e

        if output_format == "json_schema" and json_schema is not None and strict_schema:
            _validate_json_schema(obj, json_schema)

        # Canonical JSON string output (makes downstream robust)
        return json.dumps(obj, ensure_ascii=False)

    async def _chat_openai_responses(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        url = f"{self.base_url}/responses"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # Normalize input so vision works if caller used image_url parts
        input_messages = _normalize_openai_responses_input(messages)

        body: dict[str, Any] = {"model": model, "input": input_messages}
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        # Structured output (Responses API style)
        if output_format == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        elif output_format == "json_schema":
            if json_schema is None:
                raise ValueError("output_format='json_schema' requires json_schema")
            body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": bool(strict_schema),
                }
            }

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"OpenAI Responses API error: {e.response.text}") from e

            data = r.json()
            output = data.get("output")
            txt = ""

            # Your existing parsing logic, but robust for list shape
            if isinstance(output, list) and output:
                # concat all message outputs if multiple
                chunks: list[str] = []
                for item in output:
                    if isinstance(item, dict) and item.get("type") == "message":
                        parts = item.get("content") or []
                        for p in parts:
                            if isinstance(p, dict) and "text" in p:
                                chunks.append(p["text"])
                txt = "".join(chunks)

            elif isinstance(output, dict) and output.get("type") == "message":
                msg = output.get("message") or output
                parts = msg.get("content") or []
                chunks: list[str] = []
                for p in parts:
                    if isinstance(p, dict) and "text" in p:
                        chunks.append(p["text"])
                txt = "".join(chunks)

            elif isinstance(output, str):
                txt = output
            else:
                txt = ""

            usage = data.get("usage", {}) or {}
            return txt, usage

        return await self._retry.run(_call)

    async def _chat_openai_like_chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        """
        Docstring for _chat_openai_like_chat_completions

        :param self: Description
        :param messages: Description
        :type messages: list[dict[str, Any]]
        :param model: Description
        :type model: str
        :param output_format: Description
        :type output_format: ChatOutputFormat
        :param json_schema: Description
        :type json_schema: dict[str, Any] | None
        :param fail_on_unsupported: Description
        :type fail_on_unsupported: bool
        :param kw: Description
        :type kw: Any
        :return: Description
        :rtype: tuple[str, dict[str, int]]

        Call OpenAI-like /chat/completions endpoint.
        """
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        response_format = None

        if output_format == "json_object":
            response_format = {"type": "json_object"}
            msg_for_provider = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            # not truly native in most openai-like providers
            if fail_on_unsupported:
                raise RuntimeError(f"provider {self.provider} does not support native json_schema")
            msg_for_provider = _ensure_system_json_directive(messages, schema=json_schema)

        async def _call():
            body: dict[str, Any] = {
                "model": model,
                "messages": msg_for_provider,
                "temperature": temperature,
                "top_p": top_p,
            }
            if response_format is not None:
                body["response_format"] = response_format

            r = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers_openai_like(),
                json=body,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"OpenAI-like chat/completions error: {e.response.text}") from e

            data = r.json()
            txt, _ = _first_text(data.get("choices", []))  # you already have _first_text in file
            usage = data.get("usage", {}) or {}
            return txt, usage

        return await self._retry.run(_call)

    async def _chat_azure_chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        if not (self.base_url and self.azure_deployment):
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
            )

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        payload: dict[str, Any] = {
            "messages": msg_for_provider,
            "temperature": temperature,
            "top_p": top_p,
        }

        if output_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            if fail_on_unsupported:
                raise RuntimeError(
                    "Azure native json_schema not guaranteed; set fail_on_unsupported=False for best-effort"
                )
            payload["messages"] = _ensure_system_json_directive(messages, schema=json_schema)

        async def _call():
            r = await self._client.post(
                f"{self.base_url}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2024-08-01-preview",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"Azure chat/completions error: {e.response.text}") from e

            data = r.json()
            txt, _ = _first_text(data.get("choices", []))
            usage = data.get("usage", {}) or {}
            return txt, usage

        return await self._retry.run(_call)

    async def _chat_anthropic_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        # System text aggregation
        sys_msgs: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                sys_msgs.append(c if isinstance(c, str) else str(c))

        if output_format in ("json_object", "json_schema"):
            sys_msgs.insert(0, "Return ONLY valid JSON. No markdown, no commentary.")
            if output_format == "json_schema" and json_schema is not None:
                sys_msgs.insert(
                    1,
                    "JSON MUST conform to this schema:\n"
                    + json.dumps(json_schema, ensure_ascii=False),
                )

        # Convert messages to Anthropic format (blocks)
        conv: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            anthro_role = "assistant" if role == "assistant" else "user"
            content_blocks = _to_anthropic_blocks(m.get("content"))
            conv.append({"role": anthro_role, "content": content_blocks})

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": kw.get("max_tokens", 1024),
            "messages": conv,
            "temperature": temperature,
            "top_p": top_p,
        }
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
                body = e.response.text or ""
                if e.response.status_code == 404:
                    # Often model not found, or wrong base URL.
                    hint = (
                        "Anthropic returned 404. Common causes:\n"
                        "1) base_url should be https://api.anthropic.com (no /v1 suffix)\n"
                        "2) model id is invalid / unavailable for your key\n"
                        f"Request URL: {e.request.url}\n"
                    )
                    raise RuntimeError(hint + "Response body:\n" + body) from e

                raise RuntimeError(f"Anthropic API error ({e.response.status_code}): {body}") from e

            data = r.json()
            blocks = data.get("content") or []
            txt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            usage = data.get("usage", {}) or {}
            return txt, usage

        return await self._retry.run(_call)

    async def _chat_gemini_generate_content(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        # Merge system messages into preamble
        system_parts: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                system_parts.append(c if isinstance(c, str) else str(c))
        system = "\n".join(system_parts)

        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "user" if m.get("role") == "user" else "model"
            parts = _to_gemini_parts(m.get("content"))
            turns.append({"role": role, "parts": parts})

        if system:
            turns.insert(0, {"role": "user", "parts": [{"text": f"System instructions: {system}"}]})

        async def _call():
            gen_cfg: dict[str, Any] = {"temperature": temperature, "topP": top_p}

            # Gemini native structured outputs
            if output_format == "json_object":
                gen_cfg["responseMimeType"] = "application/json"
            elif output_format == "json_schema":
                if json_schema is None:
                    raise ValueError("output_format='json_schema' requires json_schema")
                gen_cfg["responseMimeType"] = "application/json"
                gen_cfg["responseJsonSchema"] = json_schema

            payload = {"contents": turns, "generationConfig": gen_cfg}

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
            txt = "".join(p.get("text", "") for p in (cand.get("content", {}).get("parts") or []))

            um = data.get("usageMetadata") or {}
            usage = {
                "input_tokens": int(um.get("promptTokenCount", 0) or 0),
                "output_tokens": int(um.get("candidatesTokenCount", 0) or 0),
            }
            return txt, usage

        return await self._retry.run(_call)

    # ---------------- Image Generation ----------------

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,  # e.g. "1024x1024"
        quality: str | None = None,  # OpenAI: "high|medium|low|auto" or dall-e: "hd|standard"
        style: str | None = None,  # dall-e-3: "vivid|natural"
        output_format: ImageFormat | None = None,  # OpenAI GPT image models: png|jpeg|webp
        response_format: ImageResponseFormat | None = None,  # dall-e: url|b64_json (OpenAI/azure)
        background: str | None = None,  # OpenAI GPT image models: "transparent|opaque|auto"
        # Optional image inputs for providers that can do edit-style generation via "prompt + image(s)"
        input_images: list[str] | None = None,  # data: URLs (base64) for now
        # Provider-specific knobs
        azure_api_version: str | None = None,
        **kw: Any,
    ) -> ImageGenerationResult:
        """
        Generate images from a text prompt using the configured LLM provider.

        This method supports provider-agnostic image generation, including OpenAI, Azure, and Google Gemini.
        It automatically handles rate limiting, usage metering, and provider-specific options.

        Examples:
            Basic usage with a prompt:
            ```python
            result = await context.llm().generate_image("A cat riding a bicycle")
            ```

            Requesting multiple images with custom size and style:
            ```python
            result = await context.llm().generate_image(
                "A futuristic cityscape",
                n=3,
                size="1024x1024",
                style="vivid"
            )
            ```

            Supplying input images for edit-style generation (Gemini):
            ```python
            result = await context.llm().generate_image(
                "Make this image brighter",
                input_images=[my_data_url]
            )
            ```

        Args:
            prompt: The text prompt describing the desired image(s).
            model: Optional model name to override the default.
            n: Number of images to generate (default: 1).
            size: Image size, e.g., "1024x1024".
            quality: Image quality setting (provider-specific).
            style: Artistic style (provider-specific).
            output_format: Desired image format, e.g., "png", "jpeg".
            response_format: Response format, e.g., "url" or "b64_json".
            background: Background setting, e.g., "transparent".
            input_images: List of input images (as data URLs) for edit-style generation.
            azure_api_version: Azure-specific API version override.
            **kw: Additional provider-specific keyword arguments.

        Returns:
            ImageGenerationResult: An object containing generated images, usage statistics, and raw response data.

        Raises:
            LLMUnsupportedFeatureError: If the provider does not support image generation.
            RuntimeError: For provider-specific errors or invalid configuration.

        Notes:
            - This method is accessed via `context.llm().generate_image(...)`.
            - Usage metering and rate limits are enforced automatically. However, token usage is typically not reported for image generation.
            - The returned `ImageGenerationResult` includes both images and metadata.
        """
        await self._ensure_client()
        model = model or self.model

        start = time.perf_counter()

        result = await self._image_dispatch(
            prompt,
            model=model,
            n=n,
            size=size,
            quality=quality,
            style=style,
            output_format=output_format,
            response_format=response_format,
            background=background,
            input_images=input_images,
            azure_api_version=azure_api_version,
            **kw,
        )

        # Rate limits: count as a call; tokens are typically not reported for images
        self._enforce_llm_limits_for_run(usage=result.usage or {})

        latency_ms = int((time.perf_counter() - start) * 1000)
        await self._record_llm_usage(model=model, usage=result.usage or {}, latency_ms=latency_ms)

        return result

    async def _image_dispatch(
        self,
        prompt: str,
        *,
        model: str,
        n: int,
        size: str | None,
        quality: str | None,
        style: str | None,
        output_format: ImageFormat | None,
        response_format: ImageResponseFormat | None,
        background: str | None,
        input_images: list[str] | None,
        azure_api_version: str | None,
        **kw: Any,
    ) -> ImageGenerationResult:
        if self.provider == "openai":
            return await self._image_openai_generate(
                prompt,
                model=model,
                n=n,
                size=size,
                quality=quality,
                style=style,
                output_format=output_format,
                response_format=response_format,
                background=background,
                **kw,
            )

        if self.provider == "azure":
            return await self._image_azure_generate(
                prompt,
                model=model,
                n=n,
                size=size,
                quality=quality,
                style=style,
                output_format=output_format,
                response_format=response_format,
                background=background,
                azure_api_version=azure_api_version,
                **kw,
            )

        if self.provider == "google":
            return await self._image_gemini_generate(
                prompt,
                model=model,
                input_images=input_images,
                **kw,
            )

        if self.provider == "anthropic":
            raise LLMUnsupportedFeatureError(
                "Anthropic does not support image generation via Claude API (vision is input-only)."
            )

        # openrouter/lmstudio/ollama: no single standard image endpoint
        raise LLMUnsupportedFeatureError(
            f"provider '{self.provider}' does not support generate_image() in this client."
        )

    async def _image_openai_generate(
        self,
        prompt: str,
        *,
        model: str,
        n: int,
        size: str | None,
        quality: str | None,
        style: str | None,
        output_format: ImageFormat | None,
        response_format: ImageResponseFormat | None,
        background: str | None,
        **kw: Any,
    ) -> ImageGenerationResult:
        assert self._client is not None

        url = f"{_normalize_base_url_no_trailing_slash(self.base_url)}/images/generations"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "n": n,
        }
        if size is not None:
            body["size"] = size
        if quality is not None:
            body["quality"] = quality
        if style is not None:
            body["style"] = style
        if output_format is not None:
            body["output_format"] = output_format
        if background is not None:
            body["background"] = background

        # For dall-e models, response_format can be url|b64_json.
        # GPT image models generally return base64 and may ignore response_format. :contentReference[oaicite:4]{index=4}
        if response_format is not None:
            body["response_format"] = response_format

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"OpenAI image generation error: {r.text}") from e

            data = r.json()
            imgs: list[GeneratedImage] = []
            for item in data.get("data", []) or []:
                imgs.append(
                    GeneratedImage(
                        b64=item.get("b64_json"),
                        url=item.get("url"),
                        mime_type=_guess_mime_from_format(output_format or "png")
                        if item.get("b64_json")
                        else None,
                        revised_prompt=item.get("revised_prompt"),
                    )
                )

            # OpenAI images endpoints often don't return token usage; keep empty usage.
            return ImageGenerationResult(images=imgs, usage=data.get("usage", {}) or {}, raw=data)

        return await self._retry.run(_call)

    async def _image_azure_generate(
        self,
        prompt: str,
        *,
        model: str,
        n: int,
        size: str | None,
        quality: str | None,
        style: str | None,
        output_format: ImageFormat | None,
        response_format: ImageResponseFormat | None,
        background: str | None,
        azure_api_version: str | None,
        **kw: Any,
    ) -> ImageGenerationResult:
        assert self._client is not None

        if not self.base_url or not self.azure_deployment:
            raise RuntimeError(
                "Azure generate_image requires base_url=<resource endpoint> and azure_deployment=<deployment name>"
            )

        api_version = (
            azure_api_version or "2025-04-01-preview"
        )  # doc example for GPT-image-1 series :contentReference[oaicite:6]{index=6}
        url = _azure_images_generations_url(self.base_url, self.azure_deployment, api_version)

        headers = {"api-key": self.api_key, "Content-Type": "application/json"}

        body: dict[str, Any] = {"prompt": prompt, "n": n}

        # For GPT-image-1 series Azure expects "model" in body (per docs). :contentReference[oaicite:7]{index=7}
        if model:
            body["model"] = model

        if size is not None:
            body["size"] = size
        if quality is not None:
            body["quality"] = quality
        if style is not None:
            body["style"] = style

        # Azure docs: GPT-image-1 series returns base64; DALL-E supports url/b64_json. :contentReference[oaicite:8]{index=8}
        if response_format is not None:
            body["response_format"] = response_format
        if output_format is not None:
            # Azure uses output_format like PNG/JPEG for some image models; you can pass through as-is.
            body["output_format"] = output_format.upper()
        if background is not None:
            body["background"] = background

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Azure image generation error: {r.text}") from e

            data = r.json()
            imgs: list[GeneratedImage] = []
            for item in data.get("data", []) or []:
                imgs.append(
                    GeneratedImage(
                        b64=item.get("b64_json"),
                        url=item.get("url"),
                        mime_type=_guess_mime_from_format((output_format or "png").lower())
                        if item.get("b64_json")
                        else None,
                        revised_prompt=item.get("revised_prompt"),
                    )
                )

            return ImageGenerationResult(images=imgs, usage=data.get("usage", {}) or {}, raw=data)

        return await self._retry.run(_call)

    async def _image_gemini_generate(
        self,
        prompt: str,
        *,
        model: str,
        input_images: list[str] | None,
        **kw: Any,
    ) -> ImageGenerationResult:
        assert self._client is not None

        # Gemini REST endpoint uses generativelanguage.googleapis.com and API key header. :contentReference[oaicite:10]{index=10}
        # Your self.base_url should already be something like: https://generativelanguage.googleapis.com
        base = (
            _normalize_base_url_no_trailing_slash(self.base_url)
            or "https://generativelanguage.googleapis.com"
        )
        url = f"{base}/v1beta/models/{model}:generateContent"

        parts: list[dict[str, Any]] = []
        if input_images:
            for img in input_images:
                if not _is_data_url(img):
                    raise ValueError("Gemini input_images must be data: URLs (base64) for now.")
                b64, mime = _data_url_to_b64_and_mime(img)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        parts.append({"text": prompt})

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
        }
        # Optional: ImageConfig etc. could be added here later per Gemini docs. :contentReference[oaicite:11]{index=11}

        async def _call():
            r = await self._client.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Gemini image generation error: {r.text}") from e

            data = r.json()
            cand = (data.get("candidates") or [{}])[0]
            content = cand.get("content") or {}
            out_parts = content.get("parts") or []

            imgs: list[GeneratedImage] = []
            for p in out_parts:
                inline = p.get("inlineData") or p.get("inline_data")
                if inline and inline.get("data"):
                    mime = inline.get("mimeType") or inline.get("mime_type")
                    imgs.append(GeneratedImage(b64=inline["data"], mime_type=mime))

            # Usage shape varies; keep best-effort.
            um = data.get("usageMetadata") or {}
            usage = {
                "input_tokens": int(um.get("promptTokenCount", 0) or 0),
                "output_tokens": int(um.get("candidatesTokenCount", 0) or 0),
            }

            return ImageGenerationResult(images=imgs, usage=usage, raw=data)

        return await self._retry.run(_call)

    # ---------------- Embeddings ----------------
    async def embed_deprecated(self, texts: list[str], **kw) -> list[list[float]]:
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

    async def embed(self, texts: list[str], **kw) -> list[list[float]]:
        """
        Generate vector embeddings for a batch of texts using the configured LLM provider.

        This method provides a provider-agnostic interface for embedding text, automatically
        handling model selection, batching, and provider-specific API quirks. It ensures the
        output shape matches the input and raises informative errors for configuration issues.

        Examples:
            Basic usage with a list of texts:
            ```python
            embeddings = await context.llm().embed([
                "The quick brown fox.",
                "Jumped over the lazy dog."
            ])
            ```

            Specifying a custom embedding model:
            ```python
            embeddings = await context.llm().embed(
                ["Hello world!"],
                model="text-embedding-3-large"
            )
            ```

        Args:
            texts: List of input strings to embed.
            model: Optional model name to override the default embedding model.
            azure_api_version: Optional Azure API version override.
            extra_body: Optional dict of extra fields to pass to the provider.
            **kw: Additional provider-specific keyword arguments.

        Returns:
            list[list[float]]: List of embedding vectors, one per input text.

        Raises:
            TypeError: If `texts` is not a list of strings.
            RuntimeError: For provider/model/configuration errors or shape mismatches.
            NotImplementedError: If embeddings are not supported for the provider.

        Notes:
            - For Google Gemini, uses batch embedding if available, otherwise falls back to per-item embedding.
            - For Azure, requires `azure_deployment` to be set.
            - The returned list always matches the length of `texts`.
        """
        await self._ensure_client()
        assert self._client is not None

        # ---- validate input ----
        if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
            raise TypeError("embed(texts) expects list[str]")
        if len(texts) == 0:
            return []

        # ---- resolve model ----
        # model override order: kw > self.embed_model > ENV > default
        model = (
            kw.get("model")
            or self.embed_model
            or os.getenv("EMBED_MODEL")
            or "text-embedding-3-small"
        )

        # ---- capability + config checks ----
        if self.provider == "anthropic":
            raise NotImplementedError("Embeddings not supported for anthropic")

        if self.provider == "azure" and not self.azure_deployment:
            raise RuntimeError(
                "Azure embeddings requires AZURE_OPENAI_DEPLOYMENT (azure_deployment)"
            )

        # Optional knobs
        azure_api_version = kw.get("azure_api_version") or "2024-08-01-preview"
        # For OpenAI-like, some providers support extra fields like dimensions/user; pass-through if present
        extra_body = kw.get("extra_body") or {}

        # ---- build request spec (within one function) ----
        # spec = (url, headers, json_body, parser_fn)
        if self.provider in {"openai", "openrouter", "lmstudio", "ollama"}:
            url = f"{self.base_url}/embeddings"
            headers = self._headers_openai_like()
            body: dict[str, object] = {"model": model, "input": texts}
            if isinstance(extra_body, dict):
                body.update(extra_body)

            def parse(data: dict) -> list[list[float]]:
                items = data.get("data", []) or []
                embs = [d.get("embedding") for d in items]
                # Ensure shape consistency
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

        if self.provider == "azure":
            # Azure embeddings are typically per-deployment; model sometimes optional/ignored
            url = f"{self.base_url}/openai/deployments/{self.azure_deployment}/embeddings?api-version={azure_api_version}"
            headers = {"api-key": self.api_key, "Content-Type": "application/json"}
            body: dict[str, object] = {"input": texts}
            # Some Azure variants also accept "model" or dimensions; keep pass-through flexible
            if model:
                body["model"] = model
            if isinstance(extra_body, dict):
                body.update(extra_body)

            def parse(data: dict) -> list[list[float]]:
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

        if self.provider == "google":
            # Goal: return one embedding per input.
            # Preferred: batchEmbedContents if supported by your endpoint/model.
            # If it 404s/400s, fallback to per-item embedContent.
            base = self.base_url.rstrip("/")
            # Newer APIs often live under v1beta; your current code uses v1. Keep v1 but fallback to v1beta if needed.
            batch_url_v1 = f"{base}/v1/models/{model}:batchEmbedContents?key={self.api_key}"
            embed_url_v1 = f"{base}/v1/models/{model}:embedContent?key={self.api_key}"
            batch_url_v1beta = f"{base}/v1beta/models/{model}:batchEmbedContents?key={self.api_key}"
            embed_url_v1beta = f"{base}/v1beta/models/{model}:embedContent?key={self.api_key}"

            headers = {"Content-Type": "application/json"}

            def parse_single(data: dict) -> list[float]:
                return (data.get("embedding") or {}).get("values") or []

            def parse_batch(data: dict) -> list[list[float]]:
                # Typical shape: {"embeddings":[{"values":[...]} , ...]}
                embs = []
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
                if r.status_code in (404, 400):
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
                    raise RuntimeError(
                        f"Gemini embeddings mismatch: got {len(out)} for {len(texts)}"
                    )
                return out

            async def _call():
                # Try v1 batch, then v1beta batch, then fallback to v1 single, then v1beta single
                res = await try_batch(batch_url_v1)
                if res is not None:
                    return res
                res = await try_batch(batch_url_v1beta)
                if res is not None:
                    return res

                # fallback loop
                try:
                    return await call_single(embed_url_v1)
                except RuntimeError:
                    return await call_single(embed_url_v1beta)

            return await self._retry.run(_call)

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
        Send a low-level HTTP request using the configured LLM provider’s client.

        This method provides direct access to the underlying HTTP transport, automatically
        applying provider-specific authentication, base URL resolution, and retry logic.
        It is intended for advanced use cases where you need to call custom endpoints
        or experiment with provider APIs not covered by higher-level methods.

        Examples:
            Basic usage with a relative path:
            ```python
            result = await context.llm().raw(
                method="POST",
                path="/custom/endpoint",
                json={"foo": "bar"}
            )
            ```

            Sending a GET request to an absolute URL:
            ```python
            response = await context.llm().raw(
                method="GET",
                url="https://api.openai.com/v1/models",
                return_response=True
            )
            ```

            Overriding headers and query parameters:
            ```python
            result = await context.llm().raw(
                path="/v1/special",
                headers={"X-Custom": "123"},
                params={"q": "search"}
            )
            ```

        Args:
            method: HTTP method to use (e.g., "POST", "GET").
            path: Relative path to append to the provider’s base URL.
            url: Absolute URL to call (overrides `path` and `base_url`).
            json: JSON-serializable body to send with the request.
            params: Dictionary of query parameters.
            headers: Dictionary of HTTP headers to override defaults.
            return_response: If True, return the raw `httpx.Response` object;
                otherwise, return the parsed JSON response.

        Returns:
            Any: The parsed JSON response by default, or the raw `httpx.Response`
            if `return_response=True`.

        Raises:
            ValueError: If neither `url` nor `path` is provided.
            RuntimeError: For HTTP errors or provider-specific failures.

        Notes:
            - This method is accessed via `context.llm().raw(...)`.
            - Provider authentication and retry logic are handled automatically.
            - Use with caution; malformed requests may result in provider errors.
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

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import json
import logging
import os
import time
from typing import Any

import httpx

from aethergraph.config.config import RateLimitSettings
from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.services.llm._anthropic_mixin import _AnthropicMixin
from aethergraph.services.llm._azure_mixin import _AzureMixin
from aethergraph.services.llm._gemini_mixin import _GeminiMixin
from aethergraph.services.llm._openai_like_mixin import _OpenAILikeMixin

# Provider mixins (chat, streaming, image generation)
from aethergraph.services.llm._openai_mixin import _OpenAIMixin
from aethergraph.services.llm.observability import (
    CaptureMode,
    LLMObservationRecord,
    LLMObservationSink,
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    ImageFormat,
    ImageGenerationResult,
    ImageResponseFormat,
    LLMUnsupportedFeatureError,
)
from aethergraph.services.llm.utils import (
    _extract_json_text,
    _strip_schema_enforced_json_fence,
    _validate_json_schema,
)

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]
_UNSET = object()


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


# ---- Generic client -------------------------------------------------------
class GenericLLMClient(
    _OpenAIMixin,
    _AnthropicMixin,
    _AzureMixin,
    _GeminiMixin,
    _OpenAILikeMixin,
    LLMClientProtocol,
):
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
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        azure_deployment: str | None = None,
        timeout: float = 60.0,
        # metering
        metering: MeteringService | None = None,
        # rate limit
        rate_limit_cfg: RateLimitSettings | None = None,
        # thinking / reasoning
        thinking_budget: int | None = None,
        reasoning_summary: str | None = None,
        # observability
        observation_sink: LLMObservationSink | None = None,
        observation_capture_mode: CaptureMode = "full",
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
        self.model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        self.embed_model = None  # will be deprecated in favor of a separate EmbeddingsClient
        self._retry = _Retry()
        self._client = httpx.AsyncClient(timeout=timeout)
        self._bound_loop = None
        self._timeout = timeout

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
                "dummy": "http://localhost:8745",  # for testing with a dummy server
            }[self.provider]
        )
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self.metering = metering

        # Rate limit settings
        self._rate_limit_cfg = rate_limit_cfg
        self._per_run_calls: dict[str, int] = {}
        self._per_run_tokens: dict[str, int] = {}

        # Thinking / reasoning config
        self.thinking_budget = thinking_budget
        self.reasoning_summary = reasoning_summary
        self.observation_sink = observation_sink
        self.observation_capture_mode = observation_capture_mode

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

    def _current_dimensions(self) -> dict[str, Any]:
        ctx = current_meter_context.get()
        return {
            "user_id": ctx.get("user_id"),
            "org_id": ctx.get("org_id"),
            "run_id": ctx.get("run_id"),
            "graph_id": ctx.get("graph_id"),
            "session_id": ctx.get("session_id"),
            "app_id": ctx.get("app_id"),
            "agent_id": ctx.get("agent_id"),
            "node_id": ctx.get("node_id"),
            "trace_id": ctx.get("trace_id"),
            "span_id": ctx.get("span_id"),
        }

    async def _record_llm_usage(
        self,
        *,
        model: str,
        usage: dict[str, Any],
        latency_ms: int | None = None,
    ) -> None:
        self.metering = self.metering or current_metering()
        prompt_tokens, completion_tokens = self._normalize_usage(usage)
        dims = self._current_dimensions()

        try:
            await self.metering.record_llm(
                user_id=dims.get("user_id"),
                org_id=dims.get("org_id"),
                run_id=dims.get("run_id"),
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

    def _build_observation_record(
        self,
        *,
        call_type: str,
        model: str,
        messages: list[dict[str, Any]],
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
        validate_json: bool,
        extra_params: dict[str, Any],
        trace_payload: dict[str, Any] | None,
    ) -> LLMObservationRecord:
        return LLMObservationRecord.new(
            call_type=call_type,
            provider=self.provider,
            model=model,
            dimensions=self._current_dimensions(),
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            extra_params=extra_params,
            trace_payload=trace_payload,
        )

    async def _emit_observation(self, record: LLMObservationRecord) -> None:
        if self.observation_sink is None:
            return
        try:
            await self.observation_sink.emit(
                record,
                capture_mode=self.observation_capture_mode,
            )
        except Exception as exc:
            logger = logging.getLogger("aethergraph.services.llm.generic_client")
            logger.warning(f"llm_observability_failed: {exc}")

    async def _ensure_client(self):
        loop = asyncio.get_running_loop()

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._bound_loop = loop
            return

        if self._bound_loop is not loop:
            # Don't attempt to close the old client here; it belongs to the old loop.
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._bound_loop = loop

    # ================================================================
    # chat() — non-streaming
    # ================================================================
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
                Common cross-provider options include:
                - model: override default model name.
                - tools: OpenAI-style tools / functions description.
                - tool_choice: tool selection strategy (e.g., "auto", "none", or provider-specific dict).

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
        model = kw.pop("model", self.model)
        trace_payload = kw.pop("trace_payload", None)
        observation_record = self._build_observation_record(
            call_type="chat",
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            extra_params=kw,
            trace_payload=trace_payload,
        )

        start = time.perf_counter()
        try:
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
            observation_record.raw_text = text
            observation_record.usage = usage or {}
            observation_record.latency_ms = latency_ms

            # Enforce rate limits (existing)
            self._enforce_llm_limits_for_run(usage=usage)

            # Metering (existing)
            await self._record_llm_usage(
                model=model,
                usage=usage,
                latency_ms=latency_ms,
            )
            await self._emit_observation(observation_record)
            return text, usage
        except Exception as exc:
            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            observation_record.error_type = type(exc).__name__
            observation_record.error_message = str(exc)
            await self._emit_observation(observation_record)
            raise

    # ================================================================
    # chat_stream() — streaming with thinking/reasoning support
    # ================================================================
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        thinking_budget: int | None | object = _UNSET,
        reasoning_summary: str | None | object = _UNSET,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat = "text",
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "output",
        strict_schema: bool = True,
        validate_json: bool = True,
        fail_on_unsupported: bool = True,
        on_delta: DeltaCallback | None = None,
        on_thinking_delta: ThinkingDeltaCallback | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        """
        Stream a chat request to the LLM provider and return the accumulated response.

        This method handles provider-specific streaming paths, falling back to non-streaming
        chat() if streaming is not implemented. It supports real-time delta updates via
        a callback function and returns the full response text and usage statistics at the end.

        Examples:
            Basic usage with a list of messages:
            ```python
            response, usage = await context.llm().chat_stream(
            messages=[{"role": "user", "content": "Hello, assistant!"}]
            )
            ```

            Using a delta callback for real-time updates:
            ```python
            async def on_delta(delta):
                print(delta, end="")

            response, usage = await context.llm().chat_stream(
                messages=[{"role": "user", "content": "Tell me a joke."}],
                on_delta=on_delta
            )
            ```

        Args:
            messages: List of message dicts, each with "role" and "content" keys.
            reasoning_effort: Optional string to control model reasoning depth.
            thinking_budget: Anthropic extended thinking budget_tokens. Uses profile default
                when omitted; pass None (or <=0) to disable for this call.
            reasoning_summary: OpenAI reasoning summary mode ('auto'/'concise'). Uses profile
                default when omitted; pass None to disable for this call.
            max_output_tokens: Optional maximum number of output tokens.
            output_format: Output format, e.g., "text" or "json".
            json_schema: Optional JSON schema for validating structured output.
            schema_name: Name for the root schema object (default: "output").
            strict_schema: If True, enforce strict schema validation.
            validate_json: If True, validate JSON output against schema.
            fail_on_unsupported: If True, raise error for unsupported features.
            on_delta: Optional callback function to handle real-time text deltas.
            on_thinking_delta: Optional callback for thinking/reasoning token deltas.
            **kw: Additional provider-specific keyword arguments.

        Returns:
            tuple[str, dict[str, int]]: The accumulated response text and usage statistics.

        Raises:
            NotImplementedError: If the provider is not supported.
            RuntimeError: For various errors including invalid JSON output or rate limit violations.
            LLMUnsupportedFeatureError: If a requested feature is unsupported by the provider.

        Notes:
            - This method centralizes handling of streaming and non-streaming paths for LLM providers.
            - The `on_delta` callback allows for real-time updates, making it suitable for interactive applications.
            - The `on_thinking_delta` callback streams thinking/reasoning tokens (OpenAI reasoning summaries, Anthropic extended thinking).
            - Rate limiting and usage metering are applied consistently across providers.
        """

        await self._ensure_client()
        model = kw.pop("model", self.model)
        trace_payload = kw.pop("trace_payload", None)
        observation_record = self._build_observation_record(
            call_type="chat_stream",
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            extra_params=kw,
            trace_payload=trace_payload,
        )
        start = time.perf_counter()

        # Resolve thinking config: omitted -> profile default, explicit value -> per-call override.
        _thinking_budget = self.thinking_budget if thinking_budget is _UNSET else thinking_budget
        _reasoning_summary = (
            self.reasoning_summary if reasoning_summary is _UNSET else reasoning_summary
        )
        if isinstance(_thinking_budget, int) and _thinking_budget <= 0:
            _thinking_budget = None

        try:
            if self.provider == "openai":
                text, usage = await self._chat_openai_responses_stream(
                    messages,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    reasoning_summary=_reasoning_summary,
                    max_output_tokens=max_output_tokens,
                    output_format=output_format,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    strict_schema=strict_schema,
                    fail_on_unsupported=fail_on_unsupported,
                    on_delta=on_delta,
                    on_thinking_delta=on_thinking_delta,
                    **kw,
                )
            elif self.provider == "anthropic":
                text, usage = await self._chat_anthropic_messages_stream(
                    messages,
                    model=model,
                    thinking_budget=_thinking_budget,
                    max_output_tokens=max_output_tokens,
                    output_format=output_format,
                    json_schema=json_schema,
                    fail_on_unsupported=fail_on_unsupported,
                    on_delta=on_delta,
                    on_thinking_delta=on_thinking_delta,
                    **kw,
                )
            else:
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
                if on_delta is not None and text:
                    await on_delta(text)

            # Postprocess (JSON modes etc.)
            text = self._postprocess_structured_output(
                text=text,
                output_format=output_format,
                json_schema=json_schema,
                strict_schema=strict_schema,
                validate_json=validate_json,
            )

            latency_ms = int((time.perf_counter() - start) * 1000)
            observation_record.raw_text = text
            observation_record.usage = usage or {}
            observation_record.latency_ms = latency_ms

            # Rate limits + metering as usual
            self._enforce_llm_limits_for_run(usage=usage)
            await self._record_llm_usage(model=model, usage=usage, latency_ms=latency_ms)
            await self._emit_observation(observation_record)
            return text, usage
        except Exception as exc:
            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            observation_record.error_type = type(exc).__name__
            observation_record.error_message = str(exc)
            await self._emit_observation(observation_record)
            raise

    # ================================================================
    # Dispatch + postprocessing
    # ================================================================
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
        # Extract cross-provider extras if any
        tools = kw.pop("tools", None)
        tool_choice = kw.pop("tool_choice", None)

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
                tools=tools,
                tool_choice=tool_choice,
                **kw,
            )

        # Everyone else
        if self.provider in {"openrouter", "lmstudio", "ollama"}:
            return await self._chat_openai_like_chat_completions(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
                tool_choice=tool_choice,
                **kw,
            )

        if self.provider == "azure":
            return await self._chat_azure_chat_completions(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
                tool_choice=tool_choice,
                **kw,
            )

        if self.provider == "anthropic":
            return await self._chat_anthropic_messages(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
                **kw,
            )

        if self.provider == "google":
            return await self._chat_gemini_generate_content(
                messages,
                model=model,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
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
        if output_format not in ("json", "json_object", "json_schema"):
            return text

        if not validate_json:
            return text

        candidate = (
            _strip_schema_enforced_json_fence(text) if output_format == "json_schema" else text
        )
        json_text = _extract_json_text(candidate)
        try:
            obj = json.loads(json_text)
        except Exception as e:
            raise RuntimeError(f"Model did not return valid JSON. Raw output:\n{text}") from e

        if output_format == "json_schema" and json_schema is not None and strict_schema:
            _validate_json_schema(obj, json_schema)

        # Canonical JSON string output (makes downstream robust)
        return json.dumps(obj, ensure_ascii=False)

    # ================================================================
    # Image Generation
    # ================================================================
    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int = 1,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        output_format: ImageFormat | None = None,
        response_format: ImageResponseFormat | None = None,
        background: str | None = None,
        input_images: list[str] | None = None,
        azure_api_version: str | None = None,
        **kw: Any,
    ) -> ImageGenerationResult:
        """
        Generate images from a text prompt using the configured LLM provider.

        This method supports provider-agnostic image generation, including OpenAI, Azure, and Google Gemini.
        It automatically handles rate limiting, usage metering, and provider-specific options.

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

    # ================================================================
    # Embeddings (deprecated — use EmbeddingClient instead)
    # ================================================================
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

    async def embed_deprecated_use_embedding_client_instead(
        self, texts: list[str], **kw
    ) -> list[list[float]]:
        """
        Generate vector embeddings for a batch of texts using the configured LLM provider.
        Deprecated: use the dedicated EmbeddingClient instead.

        Args:
            texts: List of input strings to embed.
            **kw: Additional provider-specific keyword arguments.

        Returns:
            list[list[float]]: List of embedding vectors, one per text.
        """
        await self._ensure_client()
        assert self._client is not None

        # ---- validate input ----
        if not isinstance(texts, list) or any(not isinstance(t, str) for t in texts):
            raise TypeError("embed(texts) expects list[str]")
        if len(texts) == 0:
            return []

        # ---- resolve model ----
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
        extra_body = kw.get("extra_body") or {}

        # ---- build request spec (within one function) ----
        if self.provider in {"openai", "openrouter", "lmstudio", "ollama"}:
            url = f"{self.base_url}/embeddings"
            headers = self._headers_openai_like()
            body: dict[str, object] = {"model": model, "input": texts}
            if isinstance(extra_body, dict):
                body.update(extra_body)

            def parse(data: dict) -> list[list[float]]:
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

        if self.provider == "azure":
            url = f"{self.base_url}/openai/deployments/{self.azure_deployment}/embeddings?api-version={azure_api_version}"
            headers = {"api-key": self.api_key, "Content-Type": "application/json"}
            body: dict[str, object] = {"input": texts}
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
            base = self.base_url.rstrip("/")
            batch_url_v1 = f"{base}/v1/models/{model}:batchEmbedContents?key={self.api_key}"
            embed_url_v1 = f"{base}/v1/models/{model}:embedContent?key={self.api_key}"
            batch_url_v1beta = f"{base}/v1beta/models/{model}:batchEmbedContents?key={self.api_key}"
            embed_url_v1beta = f"{base}/v1beta/models/{model}:embedContent?key={self.api_key}"

            headers = {"Content-Type": "application/json"}

            def parse_single(data: dict) -> list[float]:
                return (data.get("embedding") or {}).get("values") or []

            def parse_batch(data: dict) -> list[list[float]]:
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

        raise NotImplementedError(f"Embeddings not supported for {self.provider}")

    # ================================================================
    # Internals
    # ================================================================
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
        Send a low-level HTTP request using the configured LLM provider's client.

        Args:
            method: HTTP method to use (e.g., "POST", "GET").
            path: Relative path to append to the provider's base URL.
            url: Absolute URL to call (overrides `path` and `base_url`).
            json: JSON-serializable body to send with the request.
            params: Dictionary of query parameters.
            headers: Dictionary of HTTP headers to override defaults.
            return_response: If True, return the raw `httpx.Response` object.

        Returns:
            Any: The parsed JSON response by default, or the raw `httpx.Response`
            if `return_response=True`.
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

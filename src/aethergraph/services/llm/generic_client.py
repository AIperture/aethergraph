from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
import copy
import json
import logging
import os
import time
from typing import Any
import warnings

import httpx

from aethergraph.config.config import LLMUsageQuotaSettings
from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.contracts.services.metering import MeteringService
from aethergraph.core.runtime.runtime_metering import current_meter_context, current_metering
from aethergraph.core.schema_validation import first_schema_issue
from aethergraph.services.llm._anthropic_mixin import _AnthropicMixin
from aethergraph.services.llm._azure_mixin import _AzureMixin
from aethergraph.services.llm._gemini_mixin import _GeminiMixin
from aethergraph.services.llm._openai_like_mixin import _OpenAILikeMixin

# Provider mixins (chat, streaming, image generation)
from aethergraph.services.llm._openai_mixin import _OpenAIMixin
from aethergraph.services.llm.correlation import begin_llm_call_correlation
from aethergraph.services.llm.observability import (
    CaptureMode,
    LLMObservationRecord,
    LLMObservationSink,
)
from aethergraph.services.llm.prompt_cache import (
    PreparedPromptCache,
    prepare_prompt_cache,
)
from aethergraph.services.llm.structured_output import (
    PreparedStructuredOutput,
    StructuredOutputPolicy,
    _schema_fingerprint,
    prepare_structured_output,
    resolve_structured_output_capabilities,
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    ImageFormat,
    ImageGenerationResult,
    ImageResponseFormat,
    LLMContextWindowExceededError,
    LLMRequestEstimate,
    LLMRunQuotaExceededError,
    LLMRunQuotaWouldExceedError,
    LLMStructuredOutputCapabilityError,
    LLMStructuredOutputParseError,
    LLMStructuredOutputProviderRequestError,
    LLMStructuredOutputRefusalError,
    LLMStructuredOutputResponseError,
    LLMStructuredOutputTruncationError,
    LLMStructuredOutputValidationError,
    LLMUnsupportedFeatureError,
    PromptCacheRequest,
    StructuredOutputRequest,
)
from aethergraph.services.llm.usage import (
    normalize_llm_usage,
    normalized_usage_metrics,
)
from aethergraph.services.llm.utils import (
    _ensure_system_json_directive,
    _extract_json_text,
    _strip_schema_enforced_json_fence,
)
from aethergraph.services.tracing import resolve_tracer

DeltaCallback = Callable[[str], Awaitable[None]]
ThinkingDeltaCallback = Callable[[str], Awaitable[None]]
_UNSET = object()


def _merge_request_fields(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key == "generationConfig"
            and isinstance(value, dict)
            and isinstance(result.get(key), dict)
        ):
            result[key] = _merge_request_fields(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _record_structured_output_failure(
    request_args: dict[str, Any],
    exc: Exception,
) -> None:
    validation_outcome = "not_run"
    response_state = "runtime_error"
    if isinstance(exc, LLMStructuredOutputProviderRequestError):
        response_state = "provider_request_rejected"
    elif isinstance(exc, LLMStructuredOutputRefusalError):
        response_state = "refused"
    elif isinstance(exc, LLMStructuredOutputTruncationError):
        response_state = "truncated"
    elif isinstance(exc, LLMStructuredOutputParseError):
        validation_outcome = "parse_failed"
        response_state = exc.response_state
    elif isinstance(exc, LLMStructuredOutputValidationError):
        validation_outcome = "failed"
        response_state = exc.response_state
    request_args["structured_output_validation_outcome"] = validation_outcome
    request_args["structured_output_response_state"] = response_state
    if isinstance(exc, LLMStructuredOutputResponseError):
        request_args["structured_output_error"] = exc.to_dict()


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
        # infrastructure usage quota
        usage_quota_cfg: LLMUsageQuotaSettings | None = None,
        # thinking / reasoning
        reasoning_effort: str | None = None,
        thinking_mode: str | None = None,
        thinking_budget: int | None = None,
        reasoning_summary: str | None = None,
        compatibility_policy: str = "compat",
        structured_output_policy: StructuredOutputPolicy = "best_available",
        context_window_tokens: int | None = None,
        # observability
        observation_sink: LLMObservationSink | None = None,
        observation_capture_mode: CaptureMode = "manifest",
        # identity
        profile_name: str | None = None,
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
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
        )

        self.base_url = (
            base_url
            or {
                "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "azure": os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/"),
                "anthropic": "https://api.anthropic.com",
                "google": "https://generativelanguage.googleapis.com",
                "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "openrouter": "https://openrouter.ai/api/v1",
                "lmstudio": os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
                "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                "dummy": "http://localhost:8745",  # for testing with a dummy server
            }[self.provider]
        )
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        self.metering = metering

        self._usage_quota_cfg = usage_quota_cfg

        # Thinking / reasoning config
        self.reasoning_effort = reasoning_effort
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.reasoning_summary = reasoning_summary
        self.compatibility_policy = compatibility_policy or "compat"
        self.structured_output_policy = structured_output_policy
        self.context_window_tokens = (
            int(context_window_tokens) if context_window_tokens is not None else None
        )
        self.observation_sink = observation_sink
        self.observation_capture_mode = observation_capture_mode
        self.profile_name = profile_name
        self._logger = logging.getLogger("aethergraph.services.llm")

    def _normalize_output_format(self, output_format: ChatOutputFormat) -> ChatOutputFormat:
        if output_format == "json":
            self._logger.warning("output_format='json' is deprecated; use 'json_object' instead.")
            return "json_object"
        return output_format

    def _resolve_fail_on_unsupported(self, fail_on_unsupported: bool | None) -> bool:
        if fail_on_unsupported is not None:
            return fail_on_unsupported
        return self.compatibility_policy == "strict"

    def _normalize_structured_output(
        self,
        *,
        output_format: ChatOutputFormat,
        structured_output: StructuredOutputRequest | None,
        json_schema: dict[str, Any] | None | object,
        schema_name: str | object,
        strict_schema: bool | object,
        validate_json: bool | object,
        fail_on_unsupported: bool | None | object,
    ) -> tuple[
        ChatOutputFormat,
        dict[str, Any] | None,
        str,
        bool,
        bool,
        bool | None,
        tuple[str, ...],
    ]:
        """
        Normalize new and deprecated structured-output call forms.

        Examples:
            Normalize the provider-neutral request:
                ```python
                values = client._normalize_structured_output(
                    output_format="text",
                    structured_output=StructuredOutputRequest(
                        "Answer", {"type": "object"}
                    ),
                    json_schema=_UNSET,
                    schema_name=_UNSET,
                    strict_schema=_UNSET,
                    validate_json=_UNSET,
                    fail_on_unsupported=_UNSET,
                )
                assert values[0] == "json_schema"
                ```

            Reject mixed new and deprecated parameters:
                ```python
                try:
                    client._normalize_structured_output(
                        output_format="text",
                        structured_output=StructuredOutputRequest(
                            "Answer", {"type": "object"}
                        ),
                        json_schema={"type": "object"},
                        schema_name=_UNSET,
                        strict_schema=_UNSET,
                        validate_json=_UNSET,
                        fail_on_unsupported=_UNSET,
                    )
                except ValueError:
                    pass
                ```

        Args:
            output_format: Requested legacy text or JSON output mode.
            structured_output: New provider-neutral schema request.
            json_schema: Deprecated schema argument or the internal unset marker.
            schema_name: Deprecated schema-name argument or the unset marker.
            strict_schema: Deprecated strict-validation flag or the unset marker.
            validate_json: Deprecated local-JSON flag or the unset marker.
            fail_on_unsupported: Deprecated provider-failure flag or the unset marker.

        Returns:
            tuple: Normalized internal output fields and deprecated parameter names.

        Notes:
            Deprecated fields remain accepted through AetherGraph `0.1.x` and
            are scheduled for removal in `0.2.0`.
        """

        supplied = tuple(
            name
            for name, value in (
                ("json_schema", json_schema),
                ("schema_name", schema_name),
                ("strict_schema", strict_schema),
                ("validate_json", validate_json),
                ("fail_on_unsupported", fail_on_unsupported),
            )
            if value is not _UNSET
        )
        if structured_output is not None:
            if supplied:
                names = ", ".join(supplied)
                raise ValueError(
                    "structured_output cannot be combined with deprecated "
                    f"structured-output parameters: {names}"
                )
            if output_format != "text":
                raise ValueError(
                    "structured_output determines the output format and cannot "
                    "be combined with a non-text output_format"
                )
            return (
                "json_schema",
                copy.deepcopy(structured_output.schema),
                structured_output.name,
                True,
                True,
                None,
                (),
            )

        if supplied:
            warnings.warn(
                "json_schema, schema_name, strict_schema, validate_json, and "
                "fail_on_unsupported are deprecated; pass structured_output="
                "StructuredOutputRequest(...) instead. Legacy parameters remain "
                "supported through AetherGraph 0.1.x and will be removed in 0.2.0.",
                DeprecationWarning,
                stacklevel=3,
            )
        return (
            output_format,
            None if json_schema is _UNSET else json_schema,
            "output" if schema_name is _UNSET else str(schema_name),
            True if strict_schema is _UNSET else bool(strict_schema),
            True if validate_json is _UNSET else bool(validate_json),
            None if fail_on_unsupported is _UNSET else fail_on_unsupported,
            supplied,
        )

    @staticmethod
    def _map_deepseek_reasoning_effort(reasoning_effort: str) -> str:
        mapping = {
            "low": "high",
            "medium": "high",
            "high": "high",
            "xhigh": "max",
            "max": "max",
        }
        return mapping.get(str(reasoning_effort).lower(), str(reasoning_effort).lower())

    def _deepseek_thinking_body(self, **kw: Any) -> dict[str, Any]:
        thinking = kw.get("thinking")
        thinking_mode = kw.get("thinking_mode") or self.thinking_mode or "auto"
        if thinking is None:
            if thinking_mode == "off":
                thinking = {"type": "disabled"}
            elif thinking_mode in {"auto", "on"}:
                thinking = {"type": "enabled"}
        if isinstance(thinking, dict):
            return {"thinking": thinking}
        return {}

    def _resolve_reasoning_effort(self, reasoning_effort: str | None) -> str | None:
        return reasoning_effort if reasoning_effort is not None else self.reasoning_effort

    @staticmethod
    def _prune_none(value: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in value.items() if v is not None and v != {} and v != []}

    def _build_request_args(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
        validate_json: bool,
        deprecated_parameters: tuple[str, ...],
        prepared_structured_output: PreparedStructuredOutput | None,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        args = {
            "provider": self.provider,
            "model": model,
            "profile_name": self.profile_name,
            "compatibility_policy": self.compatibility_policy,
            "structured_output_policy": (
                prepared_structured_output.policy
                if prepared_structured_output is not None
                else self.structured_output_policy
            ),
            "reasoning_effort": reasoning_effort,
            "thinking_mode": extra_params.get("thinking_mode", self.thinking_mode),
            "thinking_budget": extra_params.get("thinking_budget", self.thinking_budget),
            "reasoning_summary": extra_params.get("reasoning_summary", self.reasoning_summary),
            "max_output_tokens": max_output_tokens,
            "output_format": output_format,
            "validate_json": validate_json
            if output_format in ("json_object", "json_schema")
            else None,
            "strict_schema": strict_schema if output_format == "json_schema" else None,
            "schema_name": schema_name
            if output_format == "json_schema" and schema_name != "output"
            else None,
            "json_schema_present": bool(json_schema) if output_format == "json_schema" else None,
            "deprecated_parameters": list(deprecated_parameters) or None,
            "structured_output_effective_mode": (
                prepared_structured_output.mode if prepared_structured_output is not None else None
            ),
            "structured_output_capability_source": (
                prepared_structured_output.capabilities.source
                if prepared_structured_output is not None
                else None
            ),
            "structured_output_projection_diagnostics": (
                [
                    {
                        "code": item.code,
                        "path": item.path,
                        "message": item.message,
                    }
                    for item in prepared_structured_output.diagnostics
                ]
                if prepared_structured_output is not None and prepared_structured_output.diagnostics
                else None
            ),
            "structured_output_canonical_schema_fingerprint": (
                prepared_structured_output.canonical_schema_fingerprint
                if prepared_structured_output is not None
                else None
            ),
            "structured_output_provider_schema_fingerprint": (
                prepared_structured_output.provider_schema_fingerprint
                if prepared_structured_output is not None
                else None
            ),
            "structured_output_validation_outcome": (
                "pending" if prepared_structured_output is not None else None
            ),
            "structured_output_response_state": (
                "pending" if prepared_structured_output is not None else None
            ),
            "temperature": extra_params.get("temperature"),
            "top_p": extra_params.get("top_p"),
            "tool_choice": extra_params.get("tool_choice"),
            "tools_count": len(extra_params.get("tools") or [])
            if extra_params.get("tools")
            else None,
        }
        return self._prune_none(args)

    def _build_provider_request_args(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        schema_name: str,
        strict_schema: bool,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        if self.provider == "openai":
            return self._prune_none(
                {
                    "reasoning": self._prune_none(
                        {
                            "effort": reasoning_effort,
                            "summary": extra_params.get(
                                "reasoning_summary", self.reasoning_summary
                            ),
                        }
                    ),
                    "text": self._prune_none(
                        {
                            "format": self._prune_none(
                                {
                                    "type": "json_object"
                                    if output_format == "json_object"
                                    else (
                                        "json_schema" if output_format == "json_schema" else None
                                    ),
                                    "name": schema_name if output_format == "json_schema" else None,
                                    "strict": strict_schema
                                    if output_format == "json_schema"
                                    else None,
                                    "schema_present": bool(json_schema)
                                    if output_format == "json_schema"
                                    else None,
                                }
                            )
                        }
                    ),
                    "max_output_tokens": max_output_tokens,
                }
            )
        if self.provider == "anthropic":
            thinking_mode = extra_params.get("thinking_mode", self.thinking_mode)
            thinking: dict[str, Any] | None = None
            if thinking_mode == "off":
                thinking = None
            elif reasoning_effort is not None:
                thinking = {"type": "adaptive", "effort": reasoning_effort}
            elif thinking_mode == "on":
                thinking = {
                    "type": "enabled",
                    "budget_tokens": extra_params.get("thinking_budget", self.thinking_budget),
                }
            return self._prune_none(
                {
                    "thinking": thinking,
                    "output_config": {"format": {"type": "json_schema", "name": schema_name}}
                    if output_format == "json_schema"
                    else None,
                    "max_tokens": max_output_tokens,
                }
            )
        if self.provider == "google":
            return self._prune_none(
                {
                    "generationConfig": self._prune_none(
                        {
                            "thinkingConfig": self._gemini_thinking_config(
                                model=model,
                                reasoning_effort=reasoning_effort,
                                thinking_mode=extra_params.get("thinking_mode", self.thinking_mode),
                            ),
                            "responseMimeType": "application/json"
                            if output_format in ("json_object", "json_schema")
                            else None,
                            "responseJsonSchemaPresent": bool(json_schema)
                            if output_format == "json_schema"
                            else None,
                            "maxOutputTokens": max_output_tokens,
                        }
                    )
                }
            )
        if self.provider == "deepseek":
            return self._prune_none(
                {
                    "reasoning_effort": self._map_deepseek_reasoning_effort(reasoning_effort)
                    if reasoning_effort is not None
                    else None,
                    "thinking": self._deepseek_thinking_body(**extra_params).get("thinking"),
                    "response_format": {"type": "json_object"}
                    if output_format == "json_object"
                    else None,
                    "max_tokens": max_output_tokens,
                }
            )
        if self.provider in {"openrouter", "lmstudio", "ollama"}:
            if self.provider == "lmstudio":
                response_format: dict[str, Any] | None = None
                if output_format == "json_object":
                    response_format = {"type": "text"}
                elif output_format == "json_schema" and json_schema is not None:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema_present": True,
                            "strict": strict_schema,
                        },
                    }
                return self._prune_none(
                    {
                        "response_format": response_format,
                        "max_tokens": max_output_tokens,
                    }
                )
            return self._prune_none(
                {
                    "response_format": {"type": "json_object"}
                    if output_format == "json_object"
                    else None,
                    "max_tokens": max_output_tokens,
                }
            )
        if self.provider == "azure":
            return self._prune_none(
                {
                    "response_format": {"type": "json_object"}
                    if output_format == "json_object"
                    else None,
                    "schema_present": bool(json_schema) if output_format == "json_schema" else None,
                    "max_tokens": max_output_tokens,
                }
            )
        return {}

    def _build_compatibility_notes(
        self,
        *,
        output_format: ChatOutputFormat,
        request_args: dict[str, Any],
        provider_request_args: dict[str, Any],
    ) -> list[str]:
        notes: list[str] = []
        if output_format == "json_object" and self.provider == "deepseek":
            notes.append("DeepSeek JSON output also requires prompt-side JSON instructions.")
        if output_format == "json_object" and self.provider == "lmstudio":
            notes.append(
                "LM Studio json_object uses response_format.type='text' plus prompt-side JSON instructions and local validation."
            )
        if output_format == "json_schema" and self.provider == "lmstudio":
            notes.append("LM Studio json_schema is requested natively and validated locally.")
        if output_format == "json_schema" and self.provider in {
            "deepseek",
            "openrouter",
            "ollama",
            "azure",
        }:
            notes.append(
                "json_schema is validated locally; native provider enforcement may be partial or unavailable."
            )
        if request_args.get("thinking_mode") == "off" and self.provider == "openai":
            notes.append(
                "OpenAI does not expose a direct thinking-mode knob; thinking_mode is advisory only."
            )
        if request_args.get("thinking_budget") and self.provider not in {"anthropic", "google"}:
            notes.append("thinking_budget is ignored by this provider.")
        if not provider_request_args:
            notes.append("No provider-specific request args were captured for this call.")
        return notes

    @staticmethod
    def _gemini_thinking_config(
        *, model: str, reasoning_effort: str | None, thinking_mode: str | None = None
    ) -> dict[str, Any] | None:
        if thinking_mode == "off":
            if model.startswith("gemini-3"):
                return {"thinkingLevel": "minimal"}
            return {"thinkingBudget": 0}
        if reasoning_effort is None:
            if thinking_mode == "on" and model.startswith("gemini-3"):
                return {"thinkingLevel": "high"}
            return None
        effort = str(reasoning_effort).lower()
        if model.startswith("gemini-3"):
            level = {
                "low": "low",
                "medium": "low",
                "high": "high",
                "xhigh": "high",
            }.get(effort)
            return {"thinkingLevel": level} if level else None
        budget = {
            "low": 0,
            "medium": 1024,
            "high": 8192,
            "xhigh": 16384,
        }.get(effort)
        return {"thinkingBudget": budget} if budget is not None else None

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

    def _get_usage_quota_cfg(self) -> LLMUsageQuotaSettings | None:
        if self._usage_quota_cfg is not None:
            return self._usage_quota_cfg
        try:
            from aethergraph.core.runtime.runtime_services import (
                current_services,
            )

            container = current_services()
            settings = getattr(container, "settings", None)
            if settings is not None and getattr(settings, "llm_usage_quota", None) is not None:
                self._usage_quota_cfg = settings.llm_usage_quota
                return self._usage_quota_cfg
        except Exception:
            pass
        return None

    @staticmethod
    def _quota_state() -> tuple[str, dict[str, int]] | None:
        ctx = current_meter_context.get()
        run_id = ctx.get("run_id")
        if not run_id:
            return None
        state = ctx.setdefault(
            "_llm_usage_quota_state",
            {"calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        return str(run_id), state

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        raw = str(text or "")
        if not raw:
            return 0
        return max(1, (len(raw) + 3) // 4)

    def _estimate_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for message in list(messages or []):
            total += 8  # role/message framing overhead
            role = message.get("role")
            if role is not None:
                total += self._estimate_text_tokens(str(role))
            content = message.get("content")
            if content is None:
                continue
            if isinstance(content, str):
                total += self._estimate_text_tokens(content)
            else:
                try:
                    total += self._estimate_text_tokens(
                        json.dumps(content, ensure_ascii=False, default=str)
                    )
                except Exception:
                    total += self._estimate_text_tokens(str(content))
        return total

    def estimate_chat_request(
        self,
        messages: list[dict[str, Any]],
        *,
        max_output_tokens: int | None,
        structured_output: StructuredOutputRequest | None = None,
        json_schema: dict[str, Any] | None = None,
        tools: Any = None,
        model: str | None = None,
    ) -> LLMRequestEstimate:
        """Estimate the current chat request without accumulated run usage.

        The estimate uses an explicit approximation and includes canonical
        structured-output and Tool payloads when supplied.

        Examples:
            Estimate a plain request:
                ```python
                estimate = client.estimate_chat_request(
                    [{"role": "user", "content": "Hello"}],
                    max_output_tokens=256,
                )
                assert estimate.reserved_output_tokens == 256
                ```

            Include one structured response schema:
                ```python
                estimate = client.estimate_chat_request(
                    messages,
                    max_output_tokens=512,
                    structured_output=request,
                )
                assert estimate.estimated_input_tokens > 0
                ```

        Args:
            messages: Current provider-neutral chat messages.
            max_output_tokens: Maximum output tokens reserved for this call.
            structured_output: Canonical structured-output request, if any.
            json_schema: Prepared or legacy canonical JSON Schema, if any.
            tools: Provider-neutral Tool declaration payload, if any.
            model: Optional per-call model override.

        Returns:
            LLMRequestEstimate: Current-request estimate and configured context
            capacity.

        Notes:
            The current implementation is deliberately labelled
            ``approximate_chars_div_4``. It is suitable for admission warnings,
            not billing.
        """

        estimated_input_tokens = self._estimate_messages_tokens(messages)
        schema = structured_output.schema if structured_output is not None else json_schema
        if schema is not None:
            estimated_input_tokens += self._estimate_text_tokens(
                json.dumps(schema, ensure_ascii=False, sort_keys=True, default=str)
            )
        if tools is not None:
            estimated_input_tokens += self._estimate_text_tokens(
                json.dumps(tools, ensure_ascii=False, sort_keys=True, default=str)
            )
        reserved_output_tokens = max(0, int(max_output_tokens or 0))
        return LLMRequestEstimate(
            model=str(model or self.model),
            estimated_input_tokens=estimated_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            estimated_total_tokens=estimated_input_tokens + reserved_output_tokens,
            context_window_tokens=self.context_window_tokens,
            source="approximate_chars_div_4",
        )

    def _preflight_llm_request(self, estimate: LLMRequestEstimate) -> None:
        if (
            estimate.context_window_tokens is not None
            and estimate.estimated_total_tokens > estimate.context_window_tokens
        ):
            raise LLMContextWindowExceededError(
                model=estimate.model,
                estimated_input_tokens=estimate.estimated_input_tokens,
                reserved_output_tokens=estimate.reserved_output_tokens,
                estimated_total_tokens=estimate.estimated_total_tokens,
                limit=estimate.context_window_tokens,
                estimate_source=estimate.source,
            )

        cfg = self._get_usage_quota_cfg()
        quota_state = self._quota_state()
        if cfg is None or quota_state is None:
            return
        run_id, state = quota_state
        checks = (
            ("llm_calls", state["calls"], 1, cfg.max_calls_per_run),
            (
                "input_tokens",
                state["input_tokens"],
                estimate.estimated_input_tokens,
                cfg.max_input_tokens_per_run,
            ),
            (
                "output_tokens",
                state["output_tokens"],
                estimate.reserved_output_tokens,
                cfg.max_output_tokens_per_run,
            ),
            (
                "total_tokens",
                state["input_tokens"] + state["output_tokens"],
                estimate.estimated_total_tokens,
                cfg.max_total_tokens_per_run,
            ),
        )
        for quota, consumed, requested, configured_limit in checks:
            if configured_limit is None:
                continue
            limit = int(configured_limit)
            projected = consumed + requested
            if projected > limit:
                raise LLMRunQuotaWouldExceedError(
                    run_id=run_id,
                    quota=quota,
                    consumed=consumed,
                    requested=requested,
                    projected=projected,
                    limit=limit,
                    phase="would be exceeded before provider dispatch",
                )

    def _record_llm_quota_usage(
        self,
        *,
        usage: dict[str, Any],
    ) -> LLMRunQuotaExceededError | None:
        cfg = self._get_usage_quota_cfg()
        if cfg is None:
            return None
        quota_state = self._quota_state()
        if quota_state is None:
            return None
        run_id, state = quota_state
        normalized = normalize_llm_usage(usage)
        input_tokens = int(normalized["input_tokens"])
        output_tokens = int(normalized["output_tokens"])
        state["calls"] += 1
        state["input_tokens"] += input_tokens
        state["output_tokens"] += output_tokens

        checks = (
            ("llm_calls", state["calls"], 1, cfg.max_calls_per_run),
            (
                "input_tokens",
                state["input_tokens"],
                input_tokens,
                cfg.max_input_tokens_per_run,
            ),
            (
                "output_tokens",
                state["output_tokens"],
                output_tokens,
                cfg.max_output_tokens_per_run,
            ),
            (
                "total_tokens",
                state["input_tokens"] + state["output_tokens"],
                input_tokens + output_tokens,
                cfg.max_total_tokens_per_run,
            ),
        )
        for quota, projected, requested, configured_limit in checks:
            if configured_limit is not None and projected > int(configured_limit):
                return LLMRunQuotaExceededError(
                    run_id=run_id,
                    quota=quota,
                    consumed=projected - requested,
                    requested=requested,
                    projected=projected,
                    limit=int(configured_limit),
                    phase="was exceeded by actual provider usage",
                    usage=normalized,
                )
        return None

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
        normalized_metrics = normalized_usage_metrics(normalize_llm_usage(usage))
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
                cache_read_tokens=normalized_metrics["cache_read_tokens"],
                cache_write_tokens=normalized_metrics["cache_write_tokens"],
                uncached_input_tokens=normalized_metrics["uncached_input_tokens"],
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
        request_args: dict[str, Any] | None,
        provider_request_args: dict[str, Any] | None,
        compatibility_notes: list[str] | None,
        trace_payload: dict[str, Any] | None,
        call_name: str | None = None,
    ) -> LLMObservationRecord:
        record = LLMObservationRecord.new(
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
            request_args=request_args,
            provider_request_args=provider_request_args,
            compatibility_notes=compatibility_notes,
            trace_payload=trace_payload,
            profile_name=self.profile_name,
            call_name=call_name,
        )
        begin_llm_call_correlation(record.llm_call_id)
        return record

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
        structured_output: StructuredOutputRequest | None = None,
        prompt_cache: PromptCacheRequest | None = None,
        json_schema: dict[str, Any] | None | object = _UNSET,
        schema_name: str | object = _UNSET,
        strict_schema: bool | object = _UNSET,
        validate_json: bool | object = _UNSET,
        fail_on_unsupported: bool | None | object = _UNSET,
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

            Request structured output with canonical JSON Schema:
            ```python
            response, usage = await context.llm().chat(
                messages=[{"role": "user", "content": "Summarize this text."}],
                structured_output=StructuredOutputRequest(
                    name="Summary",
                    schema={
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                    },
                ),
            )
            ```

            Cache a stable header and append-only transcript boundary:
            ```python
            response, usage = await context.llm().chat(
                messages,
                prompt_cache=PromptCacheRequest(
                    stable_message_indexes=(0, 4),
                    prefix_family="research-agent.v2",
                ),
            )
            ```

        Args:
            messages: List of message dicts, each with "role" and "content" keys.
            reasoning_effort: Optional string to control model reasoning depth.
            max_output_tokens: Optional maximum number of output tokens.
            output_format: Output format, e.g., "text" or "json".
            structured_output: Provider-neutral canonical schema request.
            prompt_cache: Provider-neutral stable-prefix cache request.
            json_schema: Deprecated schema argument; removed in `0.2.0`.
            schema_name: Deprecated root schema name; removed in `0.2.0`.
            strict_schema: Deprecated strict-validation flag; removed in `0.2.0`.
            validate_json: Deprecated local-validation flag; removed in `0.2.0`.
            fail_on_unsupported: Deprecated provider-failure flag; removed in `0.2.0`.
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
            - Deprecated structured-output parameters remain operational through
              `0.1.x`, emit `DeprecationWarning`, and cannot be mixed with
              `structured_output`.
            - Rate limiting and metering help manage resource usage effectively.
        """
        (
            output_format,
            json_schema,
            schema_name,
            strict_schema,
            validate_json,
            fail_on_unsupported,
            deprecated_parameters,
        ) = self._normalize_structured_output(
            output_format=output_format,
            structured_output=structured_output,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            fail_on_unsupported=fail_on_unsupported,
        )
        await self._ensure_client()
        output_format = self._normalize_output_format(output_format)
        reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)
        if "thinking_mode" not in kw and self.thinking_mode is not None:
            kw["thinking_mode"] = self.thinking_mode
        if "thinking_budget" not in kw and self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        model = kw.pop("model", self.model)
        trace_payload = kw.pop("trace_payload", None)
        call_name = kw.pop("call_name", None)
        canonical_json_schema = json_schema
        canonical_strict_validation = strict_schema
        prepared_structured_output: PreparedStructuredOutput | None = None
        prepared_prompt_cache: PreparedPromptCache | None = None
        if output_format == "json_schema" and json_schema is not None:
            effective_policy = self.structured_output_policy
            if "fail_on_unsupported" in deprecated_parameters:
                effective_policy = "native_required" if fail_on_unsupported else "best_available"
            try:
                prepared_structured_output = prepare_structured_output(
                    StructuredOutputRequest(name=schema_name, schema=json_schema),
                    provider=self.provider,
                    model=model,
                    policy=effective_policy,
                    allow_native_strict=strict_schema,
                )
            except LLMStructuredOutputCapabilityError as exc:
                capabilities = resolve_structured_output_capabilities(
                    self.provider,
                    model,
                )
                request_args = self._build_request_args(
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    output_format=output_format,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    strict_schema=strict_schema,
                    validate_json=validate_json,
                    deprecated_parameters=deprecated_parameters,
                    prepared_structured_output=None,
                    extra_params=kw,
                )
                request_args.update(
                    {
                        "structured_output_policy": effective_policy,
                        "structured_output_effective_mode": "unavailable",
                        "structured_output_capability_source": capabilities.source,
                        "structured_output_canonical_schema_fingerprint": (
                            _schema_fingerprint(json_schema)
                        ),
                        "structured_output_validation_outcome": "not_run",
                        "structured_output_response_state": "capability_rejected",
                        "structured_output_projection_diagnostics": [
                            {
                                "code": "capability_policy_unsatisfied",
                                "path": "$",
                                "message": exc.detail,
                            }
                        ],
                    }
                )
                compatibility_notes = [
                    exc.detail,
                    *(
                        [
                            "Deprecated structured-output parameters used: "
                            + ", ".join(deprecated_parameters)
                            + ". Removal is scheduled for AetherGraph 0.2.0."
                        ]
                        if deprecated_parameters
                        else []
                    ),
                ]
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
                    request_args=request_args,
                    provider_request_args={},
                    compatibility_notes=compatibility_notes,
                    trace_payload=trace_payload,
                    call_name=call_name,
                )
                observation_record.error_type = type(exc).__name__
                observation_record.error_message = str(exc)
                await self._emit_observation(observation_record)
                raise
            schema_name = prepared_structured_output.provider_schema_name
            strict_schema = prepared_structured_output.provider_strict
            json_schema = prepared_structured_output.provider_schema
            if prepared_structured_output.mode in {"native_strict", "native_schema"}:
                output_format = "json_schema"
            elif prepared_structured_output.mode == "json_object":
                output_format = "json_object"
            else:
                output_format = "json_object"
            if prepared_structured_output.prompt_guidance:
                messages = _ensure_system_json_directive(
                    messages,
                    schema=canonical_json_schema,
                )
        provider_messages = messages
        if prompt_cache is not None:
            prepared_prompt_cache = prepare_prompt_cache(
                prompt_cache,
                messages,
                provider=self.provider,
                model=model,
                scope_dimensions=self._current_dimensions(),
            )
            provider_messages = list(prepared_prompt_cache.messages)
        fail_on_unsupported = self._resolve_fail_on_unsupported(fail_on_unsupported)
        request_args = self._build_request_args(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            deprecated_parameters=deprecated_parameters,
            prepared_structured_output=prepared_structured_output,
            extra_params=kw,
        )
        provider_request_args = self._build_provider_request_args(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            extra_params=kw,
        )
        if prepared_structured_output is not None:
            provider_request_args = _merge_request_fields(
                provider_request_args,
                prepared_structured_output.provider_request_fields,
            )
        if prepared_prompt_cache is not None:
            request_args["prompt_cache"] = copy.deepcopy(prepared_prompt_cache.observation)
            provider_request_args["prompt_cache"] = copy.deepcopy(prepared_prompt_cache.observation)
        request_estimate = self.estimate_chat_request(
            messages,
            max_output_tokens=max_output_tokens,
            json_schema=canonical_json_schema,
            tools=kw.get("tools"),
            model=model,
        )
        request_args.update(
            {
                "estimated_input_tokens": request_estimate.estimated_input_tokens,
                "reserved_output_tokens": request_estimate.reserved_output_tokens,
                "estimated_request_tokens": request_estimate.estimated_total_tokens,
                "request_estimate_source": request_estimate.source,
                "model_context_window_tokens": request_estimate.context_window_tokens,
            }
        )
        compatibility_notes = self._build_compatibility_notes(
            output_format=output_format,
            request_args=request_args,
            provider_request_args=provider_request_args,
        )
        if deprecated_parameters:
            compatibility_notes.append(
                "Deprecated structured-output parameters used: "
                + ", ".join(deprecated_parameters)
                + ". Removal is scheduled for AetherGraph 0.2.0."
            )
        if prepared_structured_output is not None:
            compatibility_notes.extend(
                item.message for item in prepared_structured_output.diagnostics
            )
        observation_record = self._build_observation_record(
            call_type="chat",
            model=model,
            messages=messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=canonical_json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            extra_params=kw,
            request_args=request_args,
            provider_request_args=provider_request_args,
            compatibility_notes=compatibility_notes,
            trace_payload=trace_payload,
            call_name=call_name,
        )
        tags = ["llm", "chat"]
        if call_name:
            tags.append(call_name)
        tracer = resolve_tracer()
        span = await tracer.start_span(
            service="llm",
            operation="chat",
            request={
                "provider": self.provider,
                "model": model,
                "messages": messages,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
                "output_format": output_format,
                "trace_payload": trace_payload,
                "call_name": call_name,
            },
            tags=tags,
            metadata=self._current_dimensions(),
        )

        start = time.perf_counter()
        normalized_usage: dict[str, int] = {}
        try:
            self._preflight_llm_request(request_estimate)
            # Provider-specific call (now symmetric)
            provider_text, usage = await self._chat_dispatch(
                provider_messages,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                output_format=output_format,
                json_schema=json_schema,
                schema_name=schema_name,
                strict_schema=strict_schema,
                validate_json=validate_json,
                fail_on_unsupported=fail_on_unsupported,
                structured_output_fields=(
                    prepared_structured_output.provider_request_fields
                    if prepared_structured_output is not None
                    else None
                ),
                prompt_cache_fields=(
                    prepared_prompt_cache.provider_request_fields
                    if prepared_prompt_cache is not None
                    else None
                ),
                **kw,
            )

            observation_record.raw_text = str(provider_text or "")
            observation_record.usage = usage or {}
            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            normalized_usage = normalize_llm_usage(usage)

            quota_error = self._record_llm_quota_usage(usage=usage)
            await self._record_llm_usage(
                model=model,
                usage=usage,
                latency_ms=observation_record.latency_ms,
            )
            if quota_error is not None:
                raise quota_error

            # Canonical parsing/validation happens only after response evidence
            # and provider usage have been retained and accounted.
            text = self._postprocess_structured_output(
                text=observation_record.raw_text,
                output_format=output_format,
                json_schema=canonical_json_schema,
                strict_schema=canonical_strict_validation,
                validate_json=validate_json,
            )
            if prepared_structured_output is not None:
                request_args["structured_output_validation_outcome"] = "passed"
                request_args["structured_output_response_state"] = "completed"

            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            await self._emit_observation(observation_record)
            await span.finish(
                response={
                    "text": text,
                    "usage": usage,
                    "normalized_usage": normalized_usage,
                },
                metadata=self._current_dimensions(),
                metrics={
                    **(usage or {}),
                    **normalized_usage_metrics(normalized_usage),
                    "latency_ms": observation_record.latency_ms,
                },
            )
            return text, usage
        except Exception as exc:
            if prepared_structured_output is not None:
                _record_structured_output_failure(request_args, exc)
            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            observation_record.error_type = type(exc).__name__
            observation_record.error_message = str(exc)
            await self._emit_observation(observation_record)
            await span.fail(
                exc,
                metadata=self._current_dimensions(),
                metrics={
                    **normalized_usage_metrics(normalized_usage),
                    "latency_ms": observation_record.latency_ms or 0,
                },
            )
            raise
        finally:
            if not getattr(span, "finished", True):
                with contextlib.suppress(Exception):
                    await span.fail(
                        RuntimeError("LLM call interrupted before completion"),
                        metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
                    )

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
        structured_output: StructuredOutputRequest | None = None,
        json_schema: dict[str, Any] | None | object = _UNSET,
        schema_name: str | object = _UNSET,
        strict_schema: bool | object = _UNSET,
        validate_json: bool | object = _UNSET,
        fail_on_unsupported: bool | None | object = _UNSET,
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
            structured_output: Provider-neutral schema request; streaming rejects it.
            json_schema: Deprecated schema argument; removed in `0.2.0`.
            schema_name: Deprecated root schema name; removed in `0.2.0`.
            strict_schema: Deprecated strict-validation flag; removed in `0.2.0`.
            validate_json: Deprecated local-validation flag; removed in `0.2.0`.
            fail_on_unsupported: Deprecated provider-failure flag; removed in `0.2.0`.
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

        (
            output_format,
            json_schema,
            schema_name,
            strict_schema,
            validate_json,
            fail_on_unsupported,
            deprecated_parameters,
        ) = self._normalize_structured_output(
            output_format=output_format,
            structured_output=structured_output,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            fail_on_unsupported=fail_on_unsupported,
        )
        await self._ensure_client()
        output_format = self._normalize_output_format(output_format)
        fail_on_unsupported = self._resolve_fail_on_unsupported(fail_on_unsupported)
        reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)
        if "thinking_mode" not in kw and self.thinking_mode is not None:
            kw["thinking_mode"] = self.thinking_mode
        if output_format != "text":
            raise LLMUnsupportedFeatureError(
                self.provider,
                self.model,
                "streaming structured output",
                "chat_stream() is text-only by contract in this client",
            )
        model = kw.pop("model", self.model)
        trace_payload = kw.pop("trace_payload", None)
        call_name = kw.pop("call_name", None)
        request_args = self._build_request_args(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            deprecated_parameters=deprecated_parameters,
            prepared_structured_output=None,
            extra_params=kw,
        )
        provider_request_args = self._build_provider_request_args(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            extra_params=kw,
        )
        request_estimate = self.estimate_chat_request(
            messages,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema if isinstance(json_schema, dict) else None,
            tools=kw.get("tools"),
            model=model,
        )
        request_args.update(
            {
                "estimated_input_tokens": request_estimate.estimated_input_tokens,
                "reserved_output_tokens": request_estimate.reserved_output_tokens,
                "estimated_request_tokens": request_estimate.estimated_total_tokens,
                "request_estimate_source": request_estimate.source,
                "model_context_window_tokens": request_estimate.context_window_tokens,
            }
        )
        compatibility_notes = self._build_compatibility_notes(
            output_format=output_format,
            request_args=request_args,
            provider_request_args=provider_request_args,
        )
        if deprecated_parameters:
            compatibility_notes.append(
                "Deprecated structured-output parameters used: "
                + ", ".join(deprecated_parameters)
                + ". Removal is scheduled for AetherGraph 0.2.0."
            )
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
            request_args=request_args,
            provider_request_args=provider_request_args,
            compatibility_notes=compatibility_notes,
            trace_payload=trace_payload,
            call_name=call_name,
        )
        tags = ["llm", "chat_stream"]
        if call_name:
            tags.append(call_name)
        tracer = resolve_tracer()
        span = await tracer.start_span(
            service="llm",
            operation="chat_stream",
            request={
                "provider": self.provider,
                "model": model,
                "messages": messages,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
                "output_format": output_format,
                "trace_payload": trace_payload,
                "call_name": call_name,
            },
            tags=tags,
            metadata=self._current_dimensions(),
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
            self._preflight_llm_request(request_estimate)
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
                    reasoning_effort=reasoning_effort,
                    **kw,
                )
            elif self.provider == "deepseek":
                text, usage = await self._chat_openai_like_chat_completions_stream(
                    messages,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                    on_delta=on_delta,
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

            latency_ms = int((time.perf_counter() - start) * 1000)
            normalized_usage = normalize_llm_usage(usage)
            observation_record.raw_text = text
            observation_record.usage = usage or {}
            observation_record.latency_ms = latency_ms

            quota_error = self._record_llm_quota_usage(usage=usage)
            await self._record_llm_usage(model=model, usage=usage, latency_ms=latency_ms)
            if quota_error is not None:
                raise quota_error
            await self._emit_observation(observation_record)
            await span.finish(
                response={
                    "text": text,
                    "usage": usage,
                    "normalized_usage": normalized_usage,
                },
                metadata=self._current_dimensions(),
                metrics={
                    **(usage or {}),
                    **normalized_usage_metrics(normalized_usage),
                    "latency_ms": latency_ms,
                },
            )
            return text, usage
        except Exception as exc:
            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            observation_record.error_type = type(exc).__name__
            observation_record.error_message = str(exc)
            await self._emit_observation(observation_record)
            await span.fail(
                exc,
                metadata=self._current_dimensions(),
                metrics={"latency_ms": observation_record.latency_ms or 0},
            )
            raise
        finally:
            if not getattr(span, "finished", True):
                with contextlib.suppress(Exception):
                    await span.fail(
                        RuntimeError("LLM stream interrupted before completion"),
                        metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
                    )

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
        structured_output_fields: dict[str, Any] | None = None,
        prompt_cache_fields: dict[str, Any] | None = None,
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
                structured_output_fields=structured_output_fields,
                prompt_cache_fields=prompt_cache_fields,
                **kw,
            )

        # Everyone else
        if self.provider in {"deepseek", "openrouter", "lmstudio", "ollama"}:
            return await self._chat_openai_like_chat_completions(
                messages,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
                tool_choice=tool_choice,
                schema_name=schema_name,
                strict_schema=strict_schema,
                structured_output_fields=structured_output_fields,
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
                structured_output_fields=structured_output_fields,
                **kw,
            )

        if self.provider == "anthropic":
            return await self._chat_anthropic_messages(
                messages,
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                thinking_budget=kw.pop("thinking_budget", None),
                thinking_mode=kw.get("thinking_mode"),
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
                schema_name=schema_name,
                structured_output_fields=structured_output_fields,
                **kw,
            )

        if self.provider == "google":
            return await self._chat_gemini_generate_content(
                messages,
                model=model,
                reasoning_effort=reasoning_effort,
                thinking_mode=kw.get("thinking_mode"),
                max_output_tokens=max_output_tokens,
                output_format=output_format,
                json_schema=json_schema,
                fail_on_unsupported=fail_on_unsupported,
                tools=tools,
                structured_output_fields=structured_output_fields,
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

        candidate = (
            _strip_schema_enforced_json_fence(text) if output_format == "json_schema" else text
        )
        json_text, was_truncated, remainder = _extract_json_text(candidate)
        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise LLMStructuredOutputParseError(
                code="invalid_json",
                summary=(
                    "Model output was not valid JSON " f"(line {exc.lineno}, column {exc.colno})."
                ),
                path="$",
                validator="json",
                canonical_schema_fingerprint=(
                    _schema_fingerprint(json_schema) if json_schema is not None else ""
                ),
                response_state="invalid_json",
            ) from exc

        if was_truncated and remainder:
            try:
                remainder_obj = json.loads(remainder)
            except Exception:
                remainder_obj = None
            if remainder_obj == obj:
                logging.getLogger(__name__).warning(
                    "Model stuttered: returned identical JSON object twice. Deduplicating silently."
                )
            else:
                raise LLMStructuredOutputParseError(
                    code="multiple_json_values",
                    summary=("Model returned multiple JSON values; exactly one is required."),
                    path="$",
                    validator="json",
                    canonical_schema_fingerprint=(
                        _schema_fingerprint(json_schema) if json_schema is not None else ""
                    ),
                    response_state="invalid_json",
                )

        if json_schema is not None and strict_schema:
            issue = first_schema_issue(obj, json_schema, path="$")
            if issue is not None:
                raise LLMStructuredOutputValidationError(
                    code="schema_invalid",
                    summary=(
                        "Model output failed canonical JSON Schema validation: "
                        f"{issue.path}: {issue.message}"
                    ),
                    path=issue.path,
                    schema_path=issue.schema_path,
                    validator=issue.validator,
                    invalid_value=issue.invalid_value,
                    expected=issue.expected,
                    canonical_schema_fingerprint=_schema_fingerprint(json_schema),
                    response_state="schema_invalid",
                )

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
        tracer = resolve_tracer()
        span = await tracer.start_span(
            service="llm",
            operation="generate_image",
            request={
                "provider": self.provider,
                "model": model,
                "prompt": prompt,
                "n": n,
                "size": size,
            },
            tags=["llm", "image"],
            metadata=self._current_dimensions(),
        )

        start = time.perf_counter()

        try:
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

            latency_ms = int((time.perf_counter() - start) * 1000)
            quota_error = self._record_llm_quota_usage(usage=result.usage or {})
            await self._record_llm_usage(
                model=model, usage=result.usage or {}, latency_ms=latency_ms
            )
            if quota_error is not None:
                raise quota_error
            await span.finish(
                response={"usage": result.usage or {}, "images_count": len(result.images or [])},
                metadata=self._current_dimensions(),
                metrics={**(result.usage or {}), "latency_ms": latency_ms},
            )
            return result
        except Exception as exc:
            await span.fail(
                exc,
                metadata=self._current_dimensions(),
                metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
            )
            raise

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

        if self.provider == "deepseek":
            raise NotImplementedError("Embeddings not supported for deepseek in this client")

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
        if self.provider == "deepseek":
            raise NotImplementedError("Embeddings not supported for deepseek")

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
        if self.provider in {"openai", "openrouter", "deepseek"}:
            hdr["Authorization"] = f"Bearer {self.api_key}"
        return hdr

    async def aclose(self):
        await self._client.aclose()

    def _default_headers_for_raw(self) -> dict[str, str]:
        hdr = {"Content-Type": "application/json"}

        if self.provider in {"openai", "openrouter", "deepseek"}:
            if self.api_key:
                hdr["Authorization"] = f"Bearer {self.api_key}"
            else:
                raise RuntimeError(
                    "OpenAI-compatible providers require an API key for raw() calls."
                )

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

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import contextlib
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import logging
import os
import threading
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
from aethergraph.services.llm.adapters import ChatAdapterInvocation, invoke_chat_adapter
from aethergraph.services.llm.compat.endpoint_selection import resolve_legacy_chat_adapter
from aethergraph.services.llm.contracts import ModelRequest
from aethergraph.services.llm.correlation import begin_llm_call_correlation
from aethergraph.services.llm.credentials import resolve_provider_credential
from aethergraph.services.llm.observability import (
    CaptureMode,
    LLMObservationRecord,
    LLMObservationSink,
)
from aethergraph.services.llm.profiles import PromptCachePolicy
from aethergraph.services.llm.prompt_cache import (
    PreparedPromptCache,
    prepare_prompt_cache,
)
from aethergraph.services.llm.provider_transport import (
    LLMProviderRequestError,
    ProviderCallResult,
    ProviderRateGate,
    ProviderRetryExecutor,
    ProviderRetrySettings,
    checked_response_metadata,
)
from aethergraph.services.llm.registry import provider_default_base_url, resolve_endpoint_adapter
from aethergraph.services.llm.request_preparation import prepare_model_request
from aethergraph.services.llm.request_validation import (
    LLMRequestCompatibilityError,
    validate_model_request,
)
from aethergraph.services.llm.streaming import (
    ModelEvent,
    ModelReasoningDelta,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelUsageUpdate,
)
from aethergraph.services.llm.structured_output import (
    PreparedStructuredOutput,
    StructuredOutputPolicy,
    _schema_fingerprint,
    prepare_structured_output,
    resolve_structured_output_capabilities,
)
from aethergraph.services.llm.tool_calling import (
    AssistantOutput,
    LLMToolCallCapabilityError,
    ModelResponse,
    ToolCallRequest,
    ToolCallResponse,
    assistant_output_identity,
    tool_call_request_fingerprint,
    tool_call_surface_fingerprint,
)
from aethergraph.services.llm.tool_discovery import (
    ToolDiscoveryCapabilities,
    ToolDiscoveryModeCapability,
    ToolTransportCheckpoint,
    resolve_tool_discovery_capabilities,
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
    LLMStructuredOutputRefusalError,
    LLMStructuredOutputResponseError,
    LLMStructuredOutputTruncationError,
    LLMStructuredOutputValidationError,
    LLMUnsupportedFeatureError,
    PromptCacheRequest,
    StructuredOutputRequest,
)
from aethergraph.services.llm.usage import (
    ModelUsage,
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
UsageUpdateCallback = Callable[[dict[str, int]], Awaitable[None]]
_UNSET = object()
_QUOTA_LOCK_CREATION_GUARD = threading.Lock()
_RLOCK_TYPE = type(threading.RLock())


@dataclass
class _LLMQuotaReservation:
    """Track one active per-run estimate reserved before provider dispatch."""

    run_id: str
    state: dict[str, Any]
    lock: threading.RLock
    calls: int
    input_tokens: int
    output_tokens: int
    active: bool = True


@dataclass(frozen=True)
class _LLMConnectionState:
    provider: str
    model: str
    endpoint_id: str | None
    base_url: str
    api_key: str | None
    azure_deployment: str | None
    timeout: float
    rate_limit_group: str | None
    client: httpx.AsyncClient
    provider_retry: ProviderRetryExecutor
    tool_discovery_capabilities: ToolDiscoveryCapabilities | None


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
    if isinstance(exc, LLMProviderRequestError):
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
    provider: one of {"openai","azure","anthropic","google","deepseek","openrouter","lmstudio","ollama","openai_compatible"}
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
        endpoint_id: str | None = None,
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
        prompt_cache_policy: PromptCachePolicy = "auto",
        context_window_tokens: int | None = None,
        retry_settings: ProviderRetrySettings | None = None,
        rate_limit_group: str | None = None,
        rate_gate: ProviderRateGate | None = None,
        # observability
        observation_sink: LLMObservationSink | None = None,
        observation_capture_mode: CaptureMode = "manifest",
        # identity
        profile_name: str | None = None,
    ):
        """Create a provider-neutral LLM client with profile-level runtime policies.

        Intro:
            An explicit endpoint is validated and pinned before any request is
            inspected. Omitting it preserves the temporary `0.1.x` legacy
            provider-routing boundary while stored profiles are migrated.

        Examples:
            Create an OpenAI client with default compatibility policies:
            ```python
            client = GenericLLMClient(provider="openai", model="gpt-5-mini")
            ```

            Require callers to identify a stable prompt prefix for every request:
            ```python
            client = GenericLLMClient(
                provider="anthropic",
                model="claude-sonnet-4-5",
                prompt_cache_policy="required",
            )
            ```

        Args:
            provider: Provider identifier, or the configured environment default.
            model: Default provider model identifier.
            base_url: Optional provider API base URL override.
            api_key: Optional provider credential override.
            endpoint_id: Optional exact registered Chat endpoint adapter.
            azure_deployment: Optional Azure OpenAI deployment name.
            timeout: HTTP request timeout in seconds.
            metering: Optional service for recording normalized usage.
            usage_quota_cfg: Optional per-run usage quota configuration.
            reasoning_effort: Default model reasoning-effort setting.
            thinking_mode: Default provider thinking-mode setting.
            thinking_budget: Default provider thinking-token budget.
            reasoning_summary: Default provider reasoning-summary mode.
            compatibility_policy: Provider compatibility strictness.
            structured_output_policy: Structured-output capability requirement.
            prompt_cache_policy: Stable-prefix cache requirement policy.
            context_window_tokens: Optional context-window override.
            retry_settings: Provider transport retry configuration.
            rate_limit_group: Optional shared provider rate-limit group.
            rate_gate: Optional shared provider rate gate.
            observation_sink: Optional LLM request observation sink.
            observation_capture_mode: Observation payload capture mode.
            profile_name: Optional configured profile identity.

        Returns:
            None.

        Notes:
            The client owns provider transport resources and should be closed after use.
            Explicit endpoint bindings never switch protocol according to request
            features or provider failures.
        """
        self._client: httpx.AsyncClient | None = None
        self._retired_http_clients: list[httpx.AsyncClient] = []
        self._bound_loop = None
        self._apply_connection_state(
            self._build_connection_state(
                provider=provider,
                model=model,
                endpoint_id=endpoint_id,
                base_url=base_url,
                api_key=api_key,
                azure_deployment=azure_deployment,
                timeout=timeout,
                retry_settings=retry_settings,
                rate_limit_group=rate_limit_group,
                rate_gate=rate_gate,
            ),
            retire_current=False,
        )

        self.metering = metering

        self._usage_quota_cfg = usage_quota_cfg

        # Thinking / reasoning config
        self.reasoning_effort = reasoning_effort
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.reasoning_summary = reasoning_summary
        self.compatibility_policy = compatibility_policy or "compat"
        self.structured_output_policy = structured_output_policy
        if prompt_cache_policy not in {"disabled", "auto", "required"}:
            raise ValueError("prompt_cache_policy must be disabled, auto, or required")
        self.prompt_cache_policy = prompt_cache_policy
        self.context_window_tokens = (
            int(context_window_tokens) if context_window_tokens is not None else None
        )
        self.observation_sink = observation_sink
        self.observation_capture_mode = observation_capture_mode
        self.profile_name = profile_name
        self._tool_transport_checkpoints: dict[str, ToolTransportCheckpoint] = {}
        self._latest_tool_checkpoint_refs: dict[tuple[str, str, str], str] = {}
        self._logger = logging.getLogger("aethergraph.services.llm")

    def _resolve_chat_adapter(self, *, has_tool_request: bool):
        """Resolve the exact adapter for one current Chat invocation shape.

        Intro:
            Keeps explicit bindings immutable and confines the endpoint-less Azure
            v3 compatibility decision to one named boundary resolver.

        Examples:
            Resolve direct Chat:
                ```python
                adapter = client._resolve_chat_adapter(has_tool_request=False)
                ```

            Resolve native Tool traffic:
                ```python
                adapter = client._resolve_chat_adapter(has_tool_request=True)
                ```

        Args:
            has_tool_request: Whether the invocation carries a native Tool contract.

        Returns:
            EndpointAdapterDescriptor: Exact registered Chat adapter.

        Notes:
            No provider transport or capability fallback occurs here.
        """

        return resolve_legacy_chat_adapter(
            self.provider,
            self.endpoint_id,
            has_tool_request=has_tool_request,
        )

    @staticmethod
    def _build_connection_state(
        *,
        provider: str | None,
        model: str | None,
        endpoint_id: str | None,
        base_url: str | None,
        api_key: str | None,
        azure_deployment: str | None,
        timeout: float,
        retry_settings: ProviderRetrySettings | None,
        rate_limit_group: str | None,
        rate_gate: ProviderRateGate | None,
    ) -> _LLMConnectionState:
        resolved_provider = (provider or os.getenv("LLM_PROVIDER") or "openai").lower()
        resolved_model = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        resolved_endpoint = (
            resolve_endpoint_adapter(
                resolved_provider,
                "chat",
                endpoint_id=endpoint_id,
            ).adapter_id
            if endpoint_id is not None
            else None
        )
        resolved_api_key = resolve_provider_credential(
            provider_id=resolved_provider,
            direct=api_key,
            secret_ref=None,
            secrets=None,
        ).value
        resolved_base_url = base_url or provider_default_base_url(resolved_provider) or ""
        endpoint_family = resolve_legacy_chat_adapter(
            resolved_provider,
            resolved_endpoint,
            has_tool_request=True,
        ).protocol_family
        tool_discovery_capabilities = resolve_tool_discovery_capabilities(
            resolved_provider,
            resolved_model,
            endpoint_family,
        )
        client = httpx.AsyncClient(timeout=timeout)
        provider_retry = ProviderRetryExecutor(
            retry_settings,
            rate_gate=rate_gate,
            base_url=resolved_base_url,
            credential=resolved_api_key,
        )
        return _LLMConnectionState(
            provider=resolved_provider,
            model=resolved_model,
            endpoint_id=resolved_endpoint,
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            azure_deployment=azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            timeout=timeout,
            rate_limit_group=rate_limit_group,
            client=client,
            provider_retry=provider_retry,
            tool_discovery_capabilities=tool_discovery_capabilities,
        )

    def _apply_connection_state(
        self,
        state: _LLMConnectionState,
        *,
        retire_current: bool,
    ) -> None:
        current_client = self._client
        self.provider = state.provider
        self.model = state.model
        self.endpoint_id = state.endpoint_id
        self.base_url = state.base_url
        self.api_key = state.api_key
        self.azure_deployment = state.azure_deployment
        self._timeout = state.timeout
        self.rate_limit_group = state.rate_limit_group
        self._client = state.client
        self._provider_retry = state.provider_retry
        self._tool_discovery_capabilities = state.tool_discovery_capabilities
        self._bound_loop = None
        if retire_current and current_client is not None:
            self._retired_http_clients.append(current_client)

    def reconfigure_connection(
        self,
        *,
        provider: str,
        model: str,
        endpoint_id: str | None,
        base_url: str | None,
        api_key: str | None,
        azure_deployment: str | None,
        timeout: float,
        retry_settings: ProviderRetrySettings | None = None,
        rate_limit_group: str | None = None,
    ) -> None:
        """Replace the complete provider connection while preserving client identity.

        Intro:
            Settings hot reload uses this boundary so services holding the client
            continue to see one object while all connection-derived state changes
            together.

        Examples:
            Pin an OpenAI Responses binding:
                ```python
                client.reconfigure_connection(
                    provider="openai",
                    model="gpt-5-mini",
                    endpoint_id="openai_responses",
                    base_url=None,
                    api_key=None,
                    azure_deployment=None,
                    timeout=60.0,
                )
                ```

            Switch one Azure deployment atomically:
                ```python
                client.reconfigure_connection(
                    provider="azure",
                    model="deployment-b",
                    endpoint_id="azure_chat_completions",
                    base_url="https://example.openai.azure.com",
                    api_key="secret",
                    azure_deployment="deployment-b",
                    timeout=90.0,
                )
                ```

        Args:
            provider: Registered provider identity.
            model: Provider model or deployment identity.
            endpoint_id: Optional exact registered Chat endpoint adapter.
            base_url: Optional provider API base URL override.
            api_key: Optional already-resolved provider credential.
            azure_deployment: Optional Azure OpenAI deployment name.
            timeout: HTTP request timeout in seconds.
            retry_settings: Optional bounded provider retry policy.
            rate_limit_group: Optional shared provider quota bucket.

        Returns:
            None: Replaces connection state on this client.

        Notes:
            Validation and replacement-client construction finish before the live
            binding changes. Retired HTTP clients remain available to in-flight
            calls and are closed when this client is closed.
        """
        state = self._build_connection_state(
            provider=provider,
            model=model,
            endpoint_id=endpoint_id,
            base_url=base_url,
            api_key=api_key,
            azure_deployment=azure_deployment,
            timeout=timeout,
            retry_settings=retry_settings,
            rate_limit_group=rate_limit_group,
            rate_gate=self._provider_retry.rate_gate,
        )
        self._apply_connection_state(state, retire_current=True)
        self._tool_transport_checkpoints.clear()
        self._latest_tool_checkpoint_refs.clear()

    def pin_tool_transport_checkpoint(
        self,
        checkpoint: ToolTransportCheckpoint,
    ) -> str:
        """Pin the latest cumulative provider checkpoint for one semantic turn.

        The client validates the exact provider/model binding, replaces an older
        revision for the same semantic turn, and returns an opaque reference that
        Engine may retain without copying provider payloads into Agent state.

        Examples:
            Pin an inline checkpoint:
                ```python
                reference = client.pin_tool_transport_checkpoint(checkpoint)
                assert reference.startswith("tool-checkpoint:")
                ```

            Reuse an idempotent checkpoint:
                ```python
                first = client.pin_tool_transport_checkpoint(checkpoint)
                second = client.pin_tool_transport_checkpoint(checkpoint)
                assert first == second
                ```

        Args:
            checkpoint: Validated cumulative checkpoint returned by the provider.

        Returns:
            str: Opaque bounded reference to the sole latest checkpoint.

        Notes:
            A checkpoint with `durable_ref` uses that reference directly. Inline
            checkpoints remain process-local until a provider adapter supplies a
            durable reference in a later integration layer.
        """

        self._validate_tool_transport_checkpoint(
            checkpoint,
            model=checkpoint.model,
        )
        identity = (checkpoint.provider, checkpoint.model, checkpoint.turn_id)
        prior_ref = self._latest_tool_checkpoint_refs.get(identity)
        if prior_ref is not None:
            prior = self._tool_transport_checkpoints.get(prior_ref)
            if prior == checkpoint:
                return prior_ref
            if prior is not None and checkpoint.revision <= prior.revision:
                raise ValueError("Tool transport checkpoint revision must advance monotonically")
        reference = checkpoint.durable_ref or self._inline_checkpoint_reference(checkpoint)
        if prior_ref is not None and prior_ref != reference:
            self._tool_transport_checkpoints.pop(prior_ref, None)
        self._tool_transport_checkpoints[reference] = checkpoint
        self._latest_tool_checkpoint_refs[identity] = reference
        return reference

    def resolve_tool_transport_checkpoint(
        self,
        reference: str,
        *,
        turn_id: str,
    ) -> ToolTransportCheckpoint:
        """Resolve one exact same-turn checkpoint reference for provider replay.

        Resolution validates that the reference is still the latest authority for
        this client's provider, model, and semantic turn before returning private
        checkpoint data to the provider adapter.

        Examples:
            Resolve the current checkpoint:
                ```python
                reference = client.pin_tool_transport_checkpoint(checkpoint)
                restored = client.resolve_tool_transport_checkpoint(
                    reference,
                    turn_id=checkpoint.turn_id,
                )
                assert restored == checkpoint
                ```

            Reject a prior-turn replay:
                ```python
                client.resolve_tool_transport_checkpoint(
                    reference,
                    turn_id="different-turn",
                )
                ```

        Args:
            reference: Opaque reference returned by checkpoint pinning.
            turn_id: Exact current Engine semantic turn identity.

        Returns:
            ToolTransportCheckpoint: Latest validated private replay checkpoint.

        Notes:
            Missing or superseded references fail closed. No fallback checkpoint is
            selected from another turn, provider, or model.
        """

        normalized_ref = str(reference or "").strip()
        normalized_turn = str(turn_id or "").strip()
        if not normalized_ref or not normalized_turn:
            raise ValueError("Checkpoint reference and turn_id must be non-empty")
        checkpoint = self._tool_transport_checkpoints.get(normalized_ref)
        if checkpoint is None:
            raise KeyError(f"Tool transport checkpoint is not pinned: {normalized_ref}")
        self._validate_tool_transport_checkpoint(checkpoint, model=self.model)
        if checkpoint.turn_id != normalized_turn:
            raise ValueError("Tool transport checkpoint belongs to a different turn")
        identity = (checkpoint.provider, checkpoint.model, checkpoint.turn_id)
        if self._latest_tool_checkpoint_refs.get(identity) != normalized_ref:
            raise ValueError("Tool transport checkpoint reference is superseded")
        return checkpoint

    def release_tool_transport_checkpoint(self, reference: str) -> None:
        """Release one pinned checkpoint at the semantic-turn boundary.

        The operation removes both the opaque reference and its latest-turn index.
        Releasing an already absent reference is idempotent.

        Examples:
            Release a current checkpoint:
                ```python
                reference = client.pin_tool_transport_checkpoint(checkpoint)
                client.release_tool_transport_checkpoint(reference)
                ```

            Repeat a release safely:
                ```python
                client.release_tool_transport_checkpoint(reference)
                client.release_tool_transport_checkpoint(reference)
                ```

        Args:
            reference: Opaque checkpoint reference to release.

        Returns:
            None: Completes after removing any matching in-memory pin.

        Notes:
            Durable provider storage cleanup is adapter-owned. This method only
            releases AetherGraph's active replay authority.
        """

        normalized_ref = str(reference or "").strip()
        if not normalized_ref:
            return
        checkpoint = self._tool_transport_checkpoints.pop(normalized_ref, None)
        if checkpoint is None:
            return
        identity = (checkpoint.provider, checkpoint.model, checkpoint.turn_id)
        if self._latest_tool_checkpoint_refs.get(identity) == normalized_ref:
            self._latest_tool_checkpoint_refs.pop(identity, None)

    def _validate_tool_transport_checkpoint(
        self,
        checkpoint: ToolTransportCheckpoint,
        *,
        model: str,
    ) -> None:
        if not isinstance(checkpoint, ToolTransportCheckpoint):
            raise TypeError("checkpoint must be ToolTransportCheckpoint")
        actual = (checkpoint.provider, checkpoint.model)
        expected = (self.provider, str(model))
        if actual != expected:
            raise ValueError("Tool transport checkpoint binding does not match this LLM client")

    @staticmethod
    def _inline_checkpoint_reference(checkpoint: ToolTransportCheckpoint) -> str:
        identity = "|".join(
            (
                checkpoint.provider,
                checkpoint.model,
                checkpoint.turn_id,
                checkpoint.checkpoint_id,
                str(checkpoint.revision),
                checkpoint.integrity_digest,
            )
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"tool-checkpoint:{digest}"

    def bind_tool_discovery_capabilities(
        self,
        capabilities: ToolDiscoveryCapabilities,
    ) -> None:
        """
        Bind one exact discovery capability record to this client.

        Binding validates the client's default provider, model, and concrete
        Tool-call endpoint family. Per-call model overrides are checked again
        when a discovery request is made.

        Examples:
            Bind projected discovery to an OpenAI Responses model:
                ```python
                from aethergraph.services.llm import (
                    ToolDiscoveryCapabilities,
                    ToolDiscoveryModeCapability,
                )
                from aethergraph.services.llm.generic_client import GenericLLMClient

                client = GenericLLMClient("openai", "example-model")
                client.bind_tool_discovery_capabilities(
                    ToolDiscoveryCapabilities(
                        provider="openai",
                        model="example-model",
                        endpoint_family="responses",
                        supported_modes=(
                            ToolDiscoveryModeCapability("engine_projected"),
                        ),
                    )
                )
                ```

            Reject a record for a different endpoint family:
                ```python
                from aethergraph.services.llm import (
                    LLMToolCallCapabilityError,
                    ToolDiscoveryCapabilities,
                    ToolDiscoveryModeCapability,
                )
                from aethergraph.services.llm.generic_client import GenericLLMClient

                client = GenericLLMClient("openai", "example-model")
                try:
                    client.bind_tool_discovery_capabilities(
                        ToolDiscoveryCapabilities(
                            provider="openai",
                            model="example-model",
                            endpoint_family="chat.completions",
                            supported_modes=(
                                ToolDiscoveryModeCapability("engine_projected"),
                            ),
                        )
                    )
                except LLMToolCallCapabilityError:
                    pass
                ```

        Args:
            capabilities: Exact provider, model, endpoint, and mode record.

        Returns:
            None: Stores the validated immutable capability record.

        Notes:
            No capabilities are inferred from provider name. A discovery
            request fails closed until its exact record has been bound.
        """

        if not isinstance(capabilities, ToolDiscoveryCapabilities):
            raise TypeError("tool discovery capabilities must be ToolDiscoveryCapabilities")
        self._validate_tool_discovery_binding(
            capabilities=capabilities,
            model=self.model,
        )
        self._tool_discovery_capabilities = capabilities

    def _validate_tool_discovery_binding(
        self,
        *,
        model: str,
        capabilities: ToolDiscoveryCapabilities | None = None,
        request: ToolCallRequest | None = None,
    ) -> ToolDiscoveryModeCapability | None:
        """
        Validate discovery against the exact active provider binding.

        The check is transport-free and returns the selected per-mode record
        when a request is present. Binding-only validation is used when an
        adapter installs its capability record.

        Examples:
            Validate an exact binding without a request:
                ```python
                from aethergraph.services.llm import (
                    ToolDiscoveryCapabilities,
                    ToolDiscoveryModeCapability,
                )
                from aethergraph.services.llm.generic_client import GenericLLMClient

                client = GenericLLMClient("openai", "example-model")
                record = ToolDiscoveryCapabilities(
                    provider="openai",
                    model="example-model",
                    endpoint_family="responses",
                    supported_modes=(
                        ToolDiscoveryModeCapability("engine_projected"),
                    ),
                )
                assert client._validate_tool_discovery_binding(
                    capabilities=record,
                    model="example-model",
                ) is None
                ```

            Reject discovery when no record is bound:
                ```python
                from aethergraph.services.llm import (
                    LLMToolCallCapabilityError,
                    ToolCallRequest,
                    ToolDefinition,
                    ToolDiscoveryRequest,
                )
                from aethergraph.services.llm.generic_client import GenericLLMClient

                client = GenericLLMClient("openai", "example-model")
                request = ToolCallRequest(
                    tools=(ToolDefinition("finish", "Finish.", {"type": "object"}),),
                    discovery=ToolDiscoveryRequest("engine_projected"),
                    turn_id="turn-1",
                )
                try:
                    client._validate_tool_discovery_binding(
                        model="example-model",
                        request=request,
                    )
                except LLMToolCallCapabilityError:
                    pass
                ```

        Args:
            model: Exact model or deployment selected for the request.
            capabilities: Optional record being installed instead of the bound record.
            request: Optional Tool request whose discovery mode must be supported.

        Returns:
            ToolDiscoveryModeCapability | None: Selected mode record, or None
                for binding-only validation and requests without discovery.

        Notes:
            This method never substitutes a model, endpoint, discovery mode,
            or result bound.
        """

        discovery = request.discovery if request is not None else None
        if request is not None and discovery is None:
            return None
        record = capabilities or self._tool_discovery_capabilities
        feature = (
            f"tool_discovery.{discovery.mode}"
            if discovery is not None
            else "tool_discovery_binding"
        )
        endpoint_family = self._resolve_chat_adapter(has_tool_request=True).protocol_family
        if not endpoint_family:
            raise LLMToolCallCapabilityError(
                provider=self.provider,
                model=model,
                feature=feature,
                detail="The active client has no Tool-call endpoint family.",
            )
        if record is None:
            raise LLMToolCallCapabilityError(
                provider=self.provider,
                model=model,
                feature=feature,
                detail=(
                    "No exact discovery capability record is bound for endpoint "
                    f"'{endpoint_family}'."
                ),
            )
        actual_binding = (self.provider, str(model), endpoint_family)
        declared_binding = (record.provider, record.model, record.endpoint_family)
        if declared_binding != actual_binding:
            raise LLMToolCallCapabilityError(
                provider=self.provider,
                model=model,
                feature=feature,
                detail=(
                    "Capability record binding "
                    f"{declared_binding!r} does not match active binding "
                    f"{actual_binding!r}."
                ),
            )
        if discovery is None:
            return None
        mode_capability = next(
            (item for item in record.supported_modes if item.mode == discovery.mode),
            None,
        )
        if mode_capability is None:
            raise LLMToolCallCapabilityError(
                provider=self.provider,
                model=model,
                feature=feature,
                detail="The selected discovery mode is not declared for this binding.",
            )
        if not mode_capability.supports(discovery):
            raise LLMToolCallCapabilityError(
                provider=self.provider,
                model=model,
                feature=feature,
                detail=(
                    f"Requested max_results={discovery.max_results} cannot be "
                    f"honored by {mode_capability.result_limit_behavior} limit "
                    f"{mode_capability.max_results}."
                ),
            )
        return mode_capability

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
                structured_output.validation_owner == "aethergraph",
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
                stacklevel=4,
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
            "validate_json": (
                validate_json if output_format in ("json_object", "json_schema") else None
            ),
            "strict_schema": strict_schema if output_format == "json_schema" else None,
            "schema_name": (
                schema_name if output_format == "json_schema" and schema_name != "output" else None
            ),
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
            "tools_count": (
                len(extra_params.get("tools") or []) if extra_params.get("tools") else None
            ),
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
                                    "type": (
                                        "json_object"
                                        if output_format == "json_object"
                                        else (
                                            "json_schema"
                                            if output_format == "json_schema"
                                            else None
                                        )
                                    ),
                                    "name": schema_name if output_format == "json_schema" else None,
                                    "strict": (
                                        strict_schema if output_format == "json_schema" else None
                                    ),
                                    "schema_present": (
                                        bool(json_schema)
                                        if output_format == "json_schema"
                                        else None
                                    ),
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
                    "output_config": (
                        {"format": {"type": "json_schema", "name": schema_name}}
                        if output_format == "json_schema"
                        else None
                    ),
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
                            "responseMimeType": (
                                "application/json"
                                if output_format in ("json_object", "json_schema")
                                else None
                            ),
                            "responseJsonSchemaPresent": (
                                bool(json_schema) if output_format == "json_schema" else None
                            ),
                            "maxOutputTokens": max_output_tokens,
                        }
                    )
                }
            )
        if self.provider == "deepseek":
            return self._prune_none(
                {
                    "reasoning_effort": (
                        self._map_deepseek_reasoning_effort(reasoning_effort)
                        if reasoning_effort is not None
                        else None
                    ),
                    "thinking": self._deepseek_thinking_body(**extra_params).get("thinking"),
                    "response_format": (
                        {"type": "json_object"} if output_format == "json_object" else None
                    ),
                    "max_tokens": max_output_tokens,
                }
            )
        if self.provider in {"openrouter", "lmstudio", "ollama", "openai_compatible"}:
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
                    "response_format": (
                        {"type": "json_object"} if output_format == "json_object" else None
                    ),
                    "max_tokens": max_output_tokens,
                }
            )
        if self.provider == "azure":
            return self._prune_none(
                {
                    "response_format": (
                        {"type": "json_object"} if output_format == "json_object" else None
                    ),
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
    def _quota_state() -> tuple[str, dict[str, Any]] | None:
        ctx = current_meter_context.get()
        run_id = ctx.get("run_id")
        if not run_id:
            return None
        state = ctx.setdefault(
            "_llm_usage_quota_state",
            {"calls": 0, "input_tokens": 0, "output_tokens": 0},
        )
        return str(run_id), state

    def _initialize_shared_quota_state(self) -> None:
        """Create the nested run ledger before tracing copies the meter context."""
        if self._get_usage_quota_cfg() is not None:
            self._quota_state()

    @staticmethod
    def _quota_lock(state: dict[str, Any]) -> threading.RLock:
        lock = state.get("_reservation_lock")
        if lock is None:
            with _QUOTA_LOCK_CREATION_GUARD:
                lock = state.setdefault("_reservation_lock", threading.RLock())
        if not isinstance(lock, _RLOCK_TYPE):
            raise TypeError("LLM usage quota reservation lock is invalid")
        return lock

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
        tool_request: ToolCallRequest | None = None,
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
            tool_request: Native provider Tool-selection request, if any.
            tools: Legacy provider-neutral Tool declaration payload, if any.
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
        if tool_request is not None:
            estimated_input_tokens += self._estimate_text_tokens(
                json.dumps(
                    [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in tool_request.tools
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
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

    def estimate(self, request: ModelRequest) -> LLMRequestEstimate:
        """Estimate one canonical model request without invoking a provider.

        The estimate uses the same canonical preparation as `generate()` so Tool
        schemas, structured output, and output reservation are counted once.

        Examples:
            Estimate a direct request:
                ```python
                estimate = client.estimate(request)
                assert estimate.estimated_input_tokens > 0
                ```

            Estimate a Tool request:
                ```python
                estimate = client.estimate(tool_request)
                assert estimate.estimated_total_tokens >= estimate.estimated_input_tokens
                ```

        Args:
            request: Immutable canonical generation request.

        Returns:
            LLMRequestEstimate: Provider-neutral approximate request size and
            configured model context limit.

        Notes:
            The current estimator remains explicitly approximate and is not a
            provider billing receipt.
        """

        self._require_compatible_model_request(request)
        messages, tool_request = prepare_model_request(request)
        return self.estimate_chat_request(
            messages,
            max_output_tokens=request.generation.max_output_tokens,
            structured_output=(
                request.response_format
                if isinstance(request.response_format, StructuredOutputRequest)
                else None
            ),
            tool_request=tool_request,
        )

    def _require_compatible_model_request(self, request: ModelRequest) -> None:
        adapter = self._resolve_chat_adapter(
            has_tool_request=bool(
                request.tools or request.native_tool_search or request.continuation
            )
        )
        report = validate_model_request(request, adapter=adapter)
        if not report.valid:
            raise LLMRequestCompatibilityError(report)

    def _preflight_llm_request(
        self,
        estimate: LLMRequestEstimate,
    ) -> _LLMQuotaReservation | None:
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
            return None
        run_id, state = quota_state
        lock = self._quota_lock(state)
        with lock:
            reserved_calls = int(state.get("reserved_calls", 0))
            reserved_input = int(state.get("reserved_input_tokens", 0))
            reserved_output = int(state.get("reserved_output_tokens", 0))
            checks = (
                (
                    "llm_calls",
                    int(state.get("calls", 0)) + reserved_calls,
                    1,
                    cfg.max_calls_per_run,
                ),
                (
                    "input_tokens",
                    int(state.get("input_tokens", 0)) + reserved_input,
                    estimate.estimated_input_tokens,
                    cfg.max_input_tokens_per_run,
                ),
                (
                    "output_tokens",
                    int(state.get("output_tokens", 0)) + reserved_output,
                    estimate.reserved_output_tokens,
                    cfg.max_output_tokens_per_run,
                ),
                (
                    "total_tokens",
                    int(state.get("input_tokens", 0))
                    + int(state.get("output_tokens", 0))
                    + reserved_input
                    + reserved_output,
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
            state["reserved_calls"] = reserved_calls + 1
            state["reserved_input_tokens"] = reserved_input + estimate.estimated_input_tokens
            state["reserved_output_tokens"] = reserved_output + estimate.reserved_output_tokens
        return _LLMQuotaReservation(
            run_id=run_id,
            state=state,
            lock=lock,
            calls=1,
            input_tokens=estimate.estimated_input_tokens,
            output_tokens=estimate.reserved_output_tokens,
        )

    @staticmethod
    def _release_llm_quota_reservation(
        reservation: _LLMQuotaReservation | None,
    ) -> None:
        if reservation is None or not reservation.active:
            return
        with reservation.lock:
            if not reservation.active:
                return
            state = reservation.state
            state["reserved_calls"] = max(
                0,
                int(state.get("reserved_calls", 0)) - reservation.calls,
            )
            state["reserved_input_tokens"] = max(
                0,
                int(state.get("reserved_input_tokens", 0)) - reservation.input_tokens,
            )
            state["reserved_output_tokens"] = max(
                0,
                int(state.get("reserved_output_tokens", 0)) - reservation.output_tokens,
            )
            reservation.active = False

    def _record_llm_quota_usage(
        self,
        *,
        usage: dict[str, Any],
        reservation: _LLMQuotaReservation | None = None,
    ) -> LLMRunQuotaExceededError | None:
        cfg = self._get_usage_quota_cfg()
        if cfg is None:
            self._release_llm_quota_reservation(reservation)
            return None
        quota_state = (
            (reservation.run_id, reservation.state)
            if reservation is not None
            else self._quota_state()
        )
        if quota_state is None:
            return None
        run_id, state = quota_state
        normalized = normalize_llm_usage(usage)
        input_tokens = int(normalized["input_tokens"])
        output_tokens = int(normalized["output_tokens"])
        lock = reservation.lock if reservation is not None else self._quota_lock(state)
        with lock:
            if reservation is not None and reservation.active:
                self._release_llm_quota_reservation(reservation)
            state["calls"] = int(state.get("calls", 0)) + 1
            state["input_tokens"] = int(state.get("input_tokens", 0)) + input_tokens
            state["output_tokens"] = int(state.get("output_tokens", 0)) + output_tokens

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

    async def _account_llm_usage(
        self,
        *,
        model: str,
        usage: dict[str, Any],
        latency_ms: int | None,
        reservation: _LLMQuotaReservation | None = None,
    ) -> dict[str, int]:
        """Reconcile run quota and record metering for one completed provider call."""
        normalized = normalize_llm_usage(usage)
        quota_error = self._record_llm_quota_usage(
            usage=usage,
            reservation=reservation,
        )
        await self._record_llm_usage(
            model=model,
            usage=usage,
            latency_ms=latency_ms,
        )
        if quota_error is not None:
            raise quota_error
        return normalized

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

        if self._bound_loop is None:
            self._bound_loop = loop
            return

        if self._bound_loop is not loop:
            # Don't attempt to close the old client here; it belongs to the old loop.
            self._retired_http_clients.append(self._client)
            self._client = httpx.AsyncClient(timeout=self._timeout)
            self._bound_loop = loop

    # ================================================================
    # generate() — canonical non-streaming request
    # ================================================================
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate one canonical ordered model response.

        The method accepts only provider-neutral request state and returns one
        ordered response-item stream for direct output, discovery, and Tool calls.

        Examples:
            Generate a direct response:
                ```python
                response = await client.generate(request)
                assert response.assistant_outputs
                ```

            Generate a Tool decision:
                ```python
                response = await client.generate(tool_request)
                assert response.calls
                ```

        Args:
            request: Immutable canonical generation request.

        Returns:
            ModelResponse: Ordered output items, finish reason, typed usage, and
            optional opaque continuation.

        Notes:
            Canonical request preparation enters the shared invocation seam
            directly. The public `chat()` facade is not part of this call path.
        """

        self._require_compatible_model_request(request)
        messages, tool_request = prepare_model_request(request)
        generation_params: dict[str, Any] = {}
        if request.generation.temperature is not None:
            generation_params["temperature"] = request.generation.temperature
        if request.generation.reasoning_budget is not None:
            generation_params["thinking_budget"] = request.generation.reasoning_budget
        if request.generation.reasoning_summary is not None:
            generation_params["reasoning_summary"] = request.generation.reasoning_summary
        return await self._invoke_generation_runtime(
            messages,
            call_name=request.call_name,
            reasoning_effort=request.generation.reasoning_effort,
            max_output_tokens=request.generation.max_output_tokens,
            output_format=(
                "text"
                if isinstance(request.response_format, StructuredOutputRequest)
                else request.response_format
            ),
            structured_output=(
                request.response_format
                if isinstance(request.response_format, StructuredOutputRequest)
                else None
            ),
            tool_request=tool_request,
            prompt_cache=request.prompt_cache,
            **generation_params,
        )

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
        tool_request: ToolCallRequest | None = None,
        prompt_cache: PromptCacheRequest | None = None,
        json_schema: dict[str, Any] | None | object = _UNSET,
        schema_name: str | object = _UNSET,
        strict_schema: bool | object = _UNSET,
        validate_json: bool | object = _UNSET,
        fail_on_unsupported: bool | None | object = _UNSET,
        **kw: Any,
    ) -> tuple[str | ToolCallResponse, dict[str, int]]:
        """Project one legacy Chat call over the canonical generation lifecycle.

        Intro:
            Preserves dictionary messages, raw usage, and the conditional
            text-versus-Tool response return while the runtime itself returns only
            canonical `ModelResponse` values.

        Examples:
            Send a basic text request:
                ```python
                response, usage = await context.llm().chat(
                    [{"role": "user", "content": "Hello, assistant!"}]
                )
                ```

            Request a cacheable stable prefix:
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
            messages: Conversation messages with role and content fields.
            reasoning_effort: Optional per-request reasoning-depth override.
            max_output_tokens: Optional response token ceiling.
            output_format: Requested text or JSON response format.
            structured_output: Provider-neutral canonical schema request.
            tool_request: Provider-neutral native tool-selection request.
            prompt_cache: Provider-neutral stable-prefix cache request.
            json_schema: Deprecated schema argument; removed in `0.2.0`.
            schema_name: Deprecated root schema name; removed in `0.2.0`.
            strict_schema: Deprecated strict-validation flag; removed in `0.2.0`.
            validate_json: Deprecated local-validation flag; removed in `0.2.0`.
            fail_on_unsupported: Deprecated provider-failure flag; removed in `0.2.0`.
            **kw: Additional provider-specific request arguments.

        Returns:
            tuple[str | ToolCallResponse, dict[str, int]]: Normalized assistant
                text or canonical Tool response paired with exact raw provider usage.

        Notes:
            Deprecated structured-output parameters remain operational through `0.1.x`,
            emit `DeprecationWarning`, and cannot be mixed with `structured_output`.
        """
        response = await self._invoke_generation_runtime(
            messages,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            structured_output=structured_output,
            tool_request=tool_request,
            prompt_cache=prompt_cache,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            fail_on_unsupported=fail_on_unsupported,
            **kw,
        )
        raw_usage = dict(response.usage.provider_usage_raw)
        if tool_request is not None or response.calls or response.discovery_events:
            return response, raw_usage
        return response.text, raw_usage

    async def _invoke_generation_runtime(
        self,
        messages: list[dict[str, Any]],
        *,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat = "text",
        structured_output: StructuredOutputRequest | None = None,
        tool_request: ToolCallRequest | None = None,
        prompt_cache: PromptCacheRequest | None = None,
        json_schema: dict[str, Any] | None | object = _UNSET,
        schema_name: str | object = _UNSET,
        strict_schema: bool | object = _UNSET,
        validate_json: bool | object = _UNSET,
        fail_on_unsupported: bool | None | object = _UNSET,
        **kw: Any,
    ) -> ModelResponse:
        """Execute the shared non-streaming canonical generation lifecycle.

        Intro:
            Normalizes policy, validates provider-neutral contracts, prepares
            structured output and caching, and invokes exactly one pinned adapter.

        Examples:
            Execute a projected text request:
                ```python
                response = await client._invoke_generation_runtime(messages)
                ```

            Continue a native Tool turn:
                ```python
                response = await client._invoke_generation_runtime(
                    messages,
                    tool_request=continued_request,
                )
                ```

        Args:
            messages: Provider-projected stable conversation messages.
            reasoning_effort: Optional normalized reasoning level.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text, JSON, schema, or raw format.
            structured_output: Optional canonical structured-output request.
            tool_request: Optional canonical native Tool request and continuation.
            prompt_cache: Optional explicit stable-prefix cache request.
            json_schema: Deprecated direct schema argument.
            schema_name: Deprecated direct schema-name argument.
            strict_schema: Deprecated direct schema strictness argument.
            validate_json: Deprecated local JSON validation argument.
            fail_on_unsupported: Deprecated native-format failure argument.
            **kw: Additional bounded generation and observation options.

        Returns:
            ModelResponse: Ordered normalized response with typed usage and exact
                effective adapter metadata.

        Notes:
            Transport checkpoints are valid for ordinary Tool continuation and
            discovery continuation. Provider, model, and turn bindings are always
            validated before adapter I/O; discovery binding is validated only when
            a discovery contract is actually present.
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
        model = kw.pop("model", self.model)
        if self.prompt_cache_policy == "required" and prompt_cache is None:
            raise LLMUnsupportedFeatureError(
                self.provider,
                model,
                "prompt_cache",
                "profile policy requires explicit stable-prefix boundaries",
            )
        discovery_capability: ToolDiscoveryModeCapability | None = None
        if tool_request is not None:
            if not isinstance(tool_request, ToolCallRequest):
                raise TypeError("tool_request must be a ToolCallRequest")
            if structured_output is not None or output_format in {
                "json_object",
                "json_schema",
                "json",
            }:
                raise ValueError("Native Tool calling cannot be combined with structured output")
            if kw.get("tools") is not None or kw.get("tool_choice") is not None:
                raise ValueError("tool_request cannot be combined with legacy tools/tool_choice")
            discovery_capability = self._validate_tool_discovery_binding(
                model=model,
                request=tool_request,
            )
            if tool_request.transport_checkpoint is not None:
                self._validate_tool_transport_checkpoint(
                    tool_request.transport_checkpoint,
                    model=model,
                )
        selected_adapter = self._resolve_chat_adapter(
            has_tool_request=tool_request is not None,
        )
        effective_endpoint_id = selected_adapter.adapter_id
        await self._ensure_client()
        output_format = self._normalize_output_format(output_format)
        reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)
        if "thinking_mode" not in kw and self.thinking_mode is not None:
            kw["thinking_mode"] = self.thinking_mode
        if "thinking_budget" not in kw and self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        trace_payload = kw.pop("trace_payload", None)
        call_name = kw.pop("call_name", None)
        canonical_json_schema = json_schema
        canonical_strict_validation = strict_schema
        canonical_validation_owner = (
            structured_output.validation_owner if structured_output is not None else "aethergraph"
        )
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
                    endpoint_id=effective_endpoint_id,
                )
            except LLMStructuredOutputCapabilityError as exc:
                capabilities = resolve_structured_output_capabilities(
                    self.provider,
                    model,
                    endpoint_id=effective_endpoint_id,
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
                        "structured_output_validation_owner": canonical_validation_owner,
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
                tool_request=tool_request,
                policy=self.prompt_cache_policy,
                endpoint_id=effective_endpoint_id,
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
        request_args["endpoint_id"] = self.endpoint_id or "legacy_compat"
        provider_request_args["endpoint_id"] = self.endpoint_id or "legacy_compat"
        request_args["effective_endpoint_id"] = effective_endpoint_id
        provider_request_args["effective_endpoint_id"] = effective_endpoint_id
        if tool_request is not None:
            tool_request_summary = {
                "choice": tool_request.choice,
                "max_calls": tool_request.max_calls,
                "tool_names": [tool.name for tool in tool_request.tools],
                "tool_count": len(tool_request.tools),
                "active_tool_names": list(tool_request.active_tool_names),
                "active_tool_count": len(tool_request.active_tool_names),
                "tool_catalog_fingerprint": tool_call_request_fingerprint(tool_request)[:16],
                "tool_surface_fingerprint": tool_call_surface_fingerprint(tool_request)[:16],
            }
            if tool_request.discovery is not None and discovery_capability is not None:
                tool_request_summary["discovery"] = {
                    "mode": tool_request.discovery.mode,
                    "max_results": tool_request.discovery.max_results,
                    "endpoint_family": self._resolve_chat_adapter(
                        has_tool_request=True
                    ).protocol_family,
                    "replay_requirement": discovery_capability.replay_requirement,
                    "result_limit_behavior": discovery_capability.result_limit_behavior,
                    "capability_max_results": discovery_capability.max_results,
                    "protocol_version": discovery_capability.protocol_version,
                }
            request_args["native_tool_calling"] = copy.deepcopy(tool_request_summary)
            provider_request_args["native_tool_calling"] = copy.deepcopy(tool_request_summary)
        if prepared_structured_output is not None:
            request_args["structured_output_validation_owner"] = canonical_validation_owner
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
            tool_request=tool_request,
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
        self._initialize_shared_quota_state()
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
        quota_reservation: _LLMQuotaReservation | None = None
        try:
            quota_reservation = self._preflight_llm_request(request_estimate)
            provider_result = await self._provider_retry.execute(
                lambda: self._chat_dispatch(
                    provider_messages,
                    adapter_id=effective_endpoint_id,
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
                    prompt_cache_stable_message_count=(
                        prepared_prompt_cache.stable_message_count
                        if prepared_prompt_cache is not None
                        else None
                    ),
                    tool_request=tool_request,
                    **kw,
                ),
                provider=self.provider,
                model=model,
                operation="chat",
                rate_limit_group=self.rate_limit_group,
            )
            provider_value, usage = provider_result.value
            observation_record.attempts = provider_result.attempts

            observation_record.raw_text = (
                provider_value.observation_text()
                if isinstance(provider_value, ToolCallResponse)
                else str(provider_value or "")
            )
            observation_record.usage = usage or {}
            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            normalized_usage = await self._account_llm_usage(
                model=model,
                usage=usage,
                latency_ms=observation_record.latency_ms,
                reservation=quota_reservation,
            )
            if isinstance(provider_value, ToolCallResponse):
                provider_value = replace(
                    provider_value,
                    provider_metadata={
                        **provider_value.provider_metadata,
                        "provider": self.provider,
                        "model": model,
                        "endpoint_id": self.endpoint_id or "legacy_compat",
                        "effective_endpoint_id": effective_endpoint_id,
                    },
                    usage=ModelUsage.from_provider_usage(usage),
                )

            # Canonical parsing/validation happens only after response evidence
            # and provider usage have been retained and accounted.
            if isinstance(provider_value, ToolCallResponse):
                response = provider_value
            else:
                value = self._postprocess_structured_output(
                    text=observation_record.raw_text,
                    output_format=output_format,
                    json_schema=canonical_json_schema,
                    strict_schema=canonical_strict_validation,
                    validate_json=validate_json,
                )
                response = ModelResponse(
                    items=(
                        AssistantOutput(
                            output_id=assistant_output_identity(
                                provider=self.provider,
                                item_index=0,
                                content_index=0,
                                text=value,
                            ),
                            text=value,
                        ),
                    ),
                    finish_reason="stop",
                    provider_metadata={
                        "provider": self.provider,
                        "model": model,
                        "endpoint_id": self.endpoint_id or "legacy_compat",
                        "effective_endpoint_id": effective_endpoint_id,
                    },
                    usage=ModelUsage.from_provider_usage(usage),
                )
            if prepared_structured_output is not None:
                if canonical_validation_owner == "caller":
                    request_args["structured_output_validation_outcome"] = "delegated"
                    request_args["structured_output_response_state"] = "returned_unvalidated"
                else:
                    request_args["structured_output_validation_outcome"] = "passed"
                    request_args["structured_output_response_state"] = "completed"

            observation_record.latency_ms = int((time.perf_counter() - start) * 1000)
            await self._emit_observation(observation_record)
            await span.finish(
                response={
                    "text": observation_record.raw_text,
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
            return response
        except Exception as exc:
            if isinstance(exc, LLMProviderRequestError):
                observation_record.attempts = exc.attempts
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
            self._release_llm_quota_reservation(quota_reservation)
            if not getattr(span, "finished", True):
                with contextlib.suppress(Exception):
                    await span.fail(
                        RuntimeError("LLM call interrupted before completion"),
                        metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
                    )

    # ================================================================
    # generate_stream() + public chat_stream() compatibility facade
    # ================================================================
    async def generate_stream(self, request: ModelRequest) -> AsyncIterator[ModelEvent]:
        """Generate one typed canonical model event stream.

        Intro:
            Canonical text and reasoning deltas are yielded in provider arrival
            order, followed by one authoritative terminal `ModelResponse` event.

        Examples:
            Consume text deltas:
                ```python
                async for event in client.generate_stream(request):
                    if isinstance(event, ModelTextDelta):
                        print(event.delta, end="")
                ```

            Read terminal usage:
                ```python
                async for event in client.generate_stream(request):
                    if isinstance(event, ModelStreamCompleted):
                        usage = event.response.usage
                ```

        Args:
            request: Immutable canonical text-streaming request.

        Returns:
            AsyncIterator[ModelEvent]: Ordered typed deltas and one terminal
                canonical response event.

        Notes:
            Streaming currently accepts text responses without native Tools or
            prompt-cache boundaries. Unsupported combinations fail before provider
            I/O. Closing the iterator cancels its active provider lifecycle and
            releases the shared quota reservation in that lifecycle's `finally`.
        """

        self._require_compatible_model_request(request)
        stream_adapter = resolve_endpoint_adapter(
            self.provider,
            "chat",
            endpoint_id=self.endpoint_id,
        )
        if "streaming" not in stream_adapter.implementation_capabilities:
            raise LLMUnsupportedFeatureError(
                self.provider,
                self.model,
                "streaming",
                f"endpoint adapter {stream_adapter.adapter_id!r} has no streaming implementation",
            )
        if request.response_format != "text":
            raise LLMUnsupportedFeatureError(
                self.provider,
                self.model,
                "streaming structured output",
                "generate_stream() currently accepts only text responses",
            )
        if request.tools:
            raise LLMUnsupportedFeatureError(
                self.provider,
                self.model,
                "streaming native Tools",
                "generate_stream() currently accepts no Tool catalog",
            )
        if request.prompt_cache is not None:
            raise LLMUnsupportedFeatureError(
                self.provider,
                self.model,
                "streaming prompt cache",
                "generate_stream() does not silently drop cache boundaries",
            )

        messages, _tool_request = prepare_model_request(request)
        queue: asyncio.Queue[ModelEvent | None] = asyncio.Queue(maxsize=64)
        failure: BaseException | None = None
        text_index = 0
        reasoning_index = 0
        usage_index = 0
        latest_usage_update: ModelUsage | None = None

        async def on_delta(delta: str) -> None:
            """Queue one typed assistant-text event.

            Intro:
                Converts a non-empty adapter callback into the next canonical
                text event without blocking provider parsing.

            Examples:
                Queue text:
                    ```python
                    await on_delta("Hello")
                    ```

                Preserve whitespace:
                    ```python
                    await on_delta(" ")
                    ```

            Args:
                delta: Exact adapter text delta.

            Returns:
                None: Completes after queuing the event.

            Notes:
                Empty provider frames carry no progress and are ignored.
            """

            nonlocal text_index
            if not delta:
                return
            await queue.put(ModelTextDelta(delta=delta, index=text_index))
            text_index += 1

        async def on_reasoning_delta(delta: str) -> None:
            """Queue one typed reasoning-summary event.

            Intro:
                Separates displayable reasoning text from assistant output while
                retaining its independent arrival index.

            Examples:
                Queue reasoning:
                    ```python
                    await on_reasoning_delta("Checking")
                    ```

                Preserve whitespace:
                    ```python
                    await on_reasoning_delta(" ")
                    ```

            Args:
                delta: Exact adapter reasoning-summary delta.

            Returns:
                None: Completes after queuing the event.

            Notes:
                Empty provider frames carry no progress and are ignored.
            """

            nonlocal reasoning_index
            if not delta:
                return
            await queue.put(ModelReasoningDelta(delta=delta, index=reasoning_index))
            reasoning_index += 1

        async def on_usage_update(raw_usage: dict[str, int]) -> None:
            """Queue one changed cumulative usage snapshot.

            Intro:
                Normalizes provider usage at the stream boundary and suppresses
                repeated cumulative receipts without affecting final accounting.

            Examples:
                Queue partial input usage:
                    ```python
                    await on_usage_update({"input_tokens": 3})
                    ```

                Queue a later complete snapshot:
                    ```python
                    await on_usage_update({"input_tokens": 3, "output_tokens": 2})
                    ```

            Args:
                raw_usage: Latest cumulative raw provider usage receipt.

            Returns:
                None: Completes after queuing a changed reported snapshot.

            Notes:
                Updates are informational. The terminal lifecycle performs the
                only quota reconciliation and metering operation.
            """

            nonlocal latest_usage_update, usage_index
            usage = ModelUsage.from_provider_usage(raw_usage)
            if usage.availability == "unavailable" or usage == latest_usage_update:
                return
            await queue.put(ModelUsageUpdate(usage=usage, index=usage_index))
            latest_usage_update = usage
            usage_index += 1

        async def run_stream() -> None:
            """Run one provider stream and publish its terminal outcome.

            Intro:
                Executes the shared lifecycle in one child task so async-iterator
                consumers receive deltas live and can cancel cleanly.

            Examples:
                Start the bridge task:
                    ```python
                    task = asyncio.create_task(run_stream())
                    ```

                Await direct completion:
                    ```python
                    await run_stream()
                    ```

            Args:
                None.

            Returns:
                None: Publishes completion, failure state, and a final sentinel.

            Notes:
                Exceptions are re-raised by the consuming iterator after queued
                deltas are delivered; private exception objects are never events.
            """

            nonlocal failure
            try:
                text, usage = await self._invoke_stream_runtime(
                    messages,
                    call_name=request.call_name,
                    reasoning_effort=request.generation.reasoning_effort,
                    thinking_budget=request.generation.reasoning_budget,
                    reasoning_summary=request.generation.reasoning_summary,
                    max_output_tokens=request.generation.max_output_tokens,
                    output_format="text",
                    on_delta=on_delta,
                    on_thinking_delta=on_reasoning_delta,
                    on_usage_update=on_usage_update,
                    **(
                        {"temperature": request.generation.temperature}
                        if request.generation.temperature is not None
                        else {}
                    ),
                )
                assistant_output = AssistantOutput(
                    output_id=assistant_output_identity(
                        provider=self.provider,
                        item_index=0,
                        content_index=0,
                        text=text,
                    ),
                    text=text,
                )
                response = ModelResponse(
                    items=(assistant_output,),
                    finish_reason="stop",
                    provider_metadata={
                        "provider": self.provider,
                        "model": self.model,
                        "endpoint_id": self.endpoint_id or "legacy_compat",
                        "effective_endpoint_id": stream_adapter.adapter_id,
                    },
                    usage=ModelUsage.from_provider_usage(usage),
                )
                await queue.put(ModelStreamCompleted(response=response))
            except BaseException as exc:
                failure = exc
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_stream())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
            if failure is not None:
                raise failure
        finally:
            if not task.done():
                task.cancel()
                while not queue.empty():
                    queue.get_nowait()
            with contextlib.suppress(asyncio.CancelledError):
                await task

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
        on_usage_update: UsageUpdateCallback | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        """Stream through the preserved public Chat compatibility boundary.

        Intro:
            Existing dictionary messages and callbacks retain their documented
            behavior while sharing the same private streaming lifecycle as typed
            canonical generation.

        Examples:
            Stream and collect text:
                ```python
                text, usage = await client.chat_stream(messages)
                ```

            Observe deltas:
                ```python
                text, usage = await client.chat_stream(messages, on_delta=on_delta)
                ```

        Args:
            messages: Provider-neutral legacy Chat message dictionaries.
            reasoning_effort: Optional reasoning-depth override.
            thinking_budget: Optional reasoning-token budget override.
            reasoning_summary: Optional reasoning-summary mode override.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text output mode.
            structured_output: Unsupported canonical schema request.
            json_schema: Deprecated direct schema argument.
            schema_name: Deprecated direct schema-name argument.
            strict_schema: Deprecated direct schema strictness argument.
            validate_json: Deprecated local JSON validation argument.
            fail_on_unsupported: Deprecated native-format failure argument.
            on_delta: Optional async assistant-text callback.
            on_thinking_delta: Optional async reasoning-summary callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded legacy streaming arguments.

        Returns:
            tuple[str, dict[str, int]]: Accumulated text and provider usage.

        Notes:
            The return type and callback timing remain compatible through the
            public `0.1.x` boundary. New code should consume `generate_stream()`.
        """

        return await self._invoke_stream_runtime(
            messages,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            reasoning_summary=reasoning_summary,
            max_output_tokens=max_output_tokens,
            output_format=output_format,
            structured_output=structured_output,
            json_schema=json_schema,
            schema_name=schema_name,
            strict_schema=strict_schema,
            validate_json=validate_json,
            fail_on_unsupported=fail_on_unsupported,
            on_delta=on_delta,
            on_thinking_delta=on_thinking_delta,
            on_usage_update=on_usage_update,
            **kw,
        )

    async def _invoke_stream_runtime(
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
        on_usage_update: UsageUpdateCallback | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        """Execute the shared text-streaming provider lifecycle.

        Intro:
            Applies common validation, estimation, quota reservation, retry,
            accounting, observation, and tracing around one pinned adapter call.

        Examples:
            Accumulate a stream:
                ```python
                text, usage = await client._invoke_stream_runtime(messages)
                ```

            Forward ordered deltas:
                ```python
                text, usage = await client._invoke_stream_runtime(
                    messages,
                    on_delta=on_delta,
                )
                ```

        Args:
            messages: Stable provider-projected conversation messages.
            reasoning_effort: Optional reasoning-depth override.
            thinking_budget: Optional reasoning-token budget override.
            reasoning_summary: Optional reasoning-summary mode override.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text output mode.
            structured_output: Unsupported canonical schema request.
            json_schema: Deprecated direct schema argument.
            schema_name: Deprecated direct schema-name argument.
            strict_schema: Deprecated direct schema strictness argument.
            validate_json: Deprecated local JSON validation argument.
            fail_on_unsupported: Deprecated native-format failure argument.
            on_delta: Optional async assistant-text callback.
            on_thinking_delta: Optional async reasoning-summary callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded adapter and observation options.

        Returns:
            tuple[str, dict[str, int]]: Accumulated text and provider usage.

        Notes:
            Every selected branch is a native streaming adapter. Endpoints without
            a streaming implementation fail before transport and never issue a
            non-streaming request as a fallback.
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
        stream_adapter = resolve_endpoint_adapter(
            self.provider,
            "chat",
            endpoint_id=self.endpoint_id,
        )
        if "streaming" not in stream_adapter.implementation_capabilities:
            raise LLMUnsupportedFeatureError(
                self.provider,
                self.model,
                "streaming",
                (f"endpoint adapter {stream_adapter.adapter_id!r} has no streaming implementation"),
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
        self._initialize_shared_quota_state()
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

        quota_reservation: _LLMQuotaReservation | None = None
        try:
            quota_reservation = self._preflight_llm_request(request_estimate)
            if self.provider == "openai" and self.endpoint_id != "openai_chat_completions":
                provider_result = await self._provider_retry.execute(
                    lambda: self._chat_openai_responses_stream(
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
                        on_usage_update=on_usage_update,
                        **kw,
                    ),
                    provider=self.provider,
                    model=model,
                    operation="chat_stream",
                    rate_limit_group=self.rate_limit_group,
                )
                text, usage = provider_result.value
            elif self.provider == "anthropic":
                provider_result = await self._provider_retry.execute(
                    lambda: self._chat_anthropic_messages_stream(
                        messages,
                        model=model,
                        thinking_budget=_thinking_budget,
                        max_output_tokens=max_output_tokens,
                        output_format=output_format,
                        json_schema=json_schema,
                        fail_on_unsupported=fail_on_unsupported,
                        on_delta=on_delta,
                        on_thinking_delta=on_thinking_delta,
                        on_usage_update=on_usage_update,
                        reasoning_effort=reasoning_effort,
                        **kw,
                    ),
                    provider=self.provider,
                    model=model,
                    operation="chat_stream",
                    rate_limit_group=self.rate_limit_group,
                )
                text, usage = provider_result.value
            elif stream_adapter.protocol_family == "chat.completions":
                if self.provider == "azure":
                    provider_result = await self._provider_retry.execute(
                        lambda: self._chat_azure_chat_completions_stream(
                            messages,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            max_output_tokens=max_output_tokens,
                            on_delta=on_delta,
                            on_usage_update=on_usage_update,
                            **kw,
                        ),
                        provider=self.provider,
                        model=model,
                        operation="chat_stream",
                        rate_limit_group=self.rate_limit_group,
                    )
                else:
                    provider_result = await self._provider_retry.execute(
                        lambda: self._chat_openai_like_chat_completions_stream(
                            messages,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            max_output_tokens=max_output_tokens,
                            on_delta=on_delta,
                            on_usage_update=on_usage_update,
                            **kw,
                        ),
                        provider=self.provider,
                        model=model,
                        operation="chat_stream",
                        rate_limit_group=self.rate_limit_group,
                    )
                text, usage = provider_result.value
            elif self.provider == "google" and stream_adapter.protocol_family == "generateContent":
                provider_result = await self._provider_retry.execute(
                    lambda: self._chat_gemini_generate_content_stream(
                        messages,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        reasoning_summary=_reasoning_summary,
                        thinking_mode=kw.get("thinking_mode"),
                        max_output_tokens=max_output_tokens,
                        on_delta=on_delta,
                        on_thinking_delta=on_thinking_delta,
                        on_usage_update=on_usage_update,
                        **{key: value for key, value in kw.items() if key != "thinking_mode"},
                    ),
                    provider=self.provider,
                    model=model,
                    operation="chat_stream",
                    rate_limit_group=self.rate_limit_group,
                )
                text, usage = provider_result.value
            else:
                raise LLMUnsupportedFeatureError(
                    provider=self.provider,
                    model=model,
                    feature="streaming",
                    detail=(
                        f"endpoint adapter {stream_adapter.adapter_id!r} "
                        "has no native streaming adapter"
                    ),
                )

            observation_record.attempts = provider_result.attempts

            latency_ms = int((time.perf_counter() - start) * 1000)
            observation_record.raw_text = text
            observation_record.usage = usage or {}
            observation_record.latency_ms = latency_ms

            normalized_usage = await self._account_llm_usage(
                model=model,
                usage=usage,
                latency_ms=latency_ms,
                reservation=quota_reservation,
            )
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
            if isinstance(exc, LLMProviderRequestError):
                observation_record.attempts = exc.attempts
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
            self._release_llm_quota_reservation(quota_reservation)
            if not getattr(span, "finished", True):
                with contextlib.suppress(Exception):
                    await span.fail(
                        RuntimeError("LLM stream interrupted before completion"),
                        metrics={"latency_ms": int((time.perf_counter() - start) * 1000)},
                    )

    async def _chat_dispatch(
        self,
        messages: list[dict[str, Any]],
        *,
        adapter_id: str,
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
        prompt_cache_stable_message_count: int | None = None,
        tool_request: ToolCallRequest | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Invoke one exact endpoint adapter without provider-name dispatch.

        Intro:
            Freezes lifecycle-prepared state into one adapter invocation and
            delegates exactly one physical attempt through the runtime registry.

        Examples:
            Invoke direct Chat:
                ```python
                result = await client._chat_dispatch(
                    messages,
                    adapter_id="openai_responses",
                    model="gpt-test",
                    reasoning_effort=None,
                    max_output_tokens=128,
                    output_format="text",
                    json_schema=None,
                    schema_name="Response",
                    strict_schema=True,
                    validate_json=True,
                    fail_on_unsupported=True,
                )
                ```

            Invoke native Tools:
                ```python
                result = await client._chat_dispatch(
                    messages,
                    adapter_id="anthropic_messages",
                    model="claude-test",
                    reasoning_effort=None,
                    max_output_tokens=128,
                    output_format="text",
                    json_schema=None,
                    schema_name="Response",
                    strict_schema=True,
                    validate_json=True,
                    fail_on_unsupported=True,
                    tool_request=tool_request,
                )
                ```

        Args:
            messages: Lifecycle-prepared stable conversation messages.
            adapter_id: Exact selected endpoint-adapter identity.
            model: Exact configured model or deployment identity.
            reasoning_effort: Optional normalized reasoning-depth override.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Prepared text, JSON, schema, or raw mode.
            json_schema: Optional prepared provider JSON schema.
            schema_name: Stable provider schema name.
            strict_schema: Whether native schema enforcement is strict.
            validate_json: Whether shared postprocessing validates JSON locally.
            fail_on_unsupported: Whether unsupported native fields must fail.
            structured_output_fields: Optional prepared native structured fields.
            prompt_cache_fields: Optional prepared native cache fields.
            prompt_cache_stable_message_count: Optional stable prefix length.
            tool_request: Optional canonical native Tool request.
            **kw: Additional bounded adapter-private options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Raw single-attempt adapter value and transport metadata.

        Notes:
            This method is the injectable physical-attempt seam used by transport
            tests. It contains no provider selection, retry, or fallback logic.
        """

        return await invoke_chat_adapter(
            self,
            adapter_id=adapter_id,
            invocation=ChatAdapterInvocation(
                messages=tuple(messages),
                model=model,
                reasoning_effort=reasoning_effort,
                max_output_tokens=max_output_tokens,
                output_format=output_format,
                json_schema=json_schema,
                schema_name=schema_name,
                strict_schema=strict_schema,
                validate_json=validate_json,
                fail_on_unsupported=fail_on_unsupported,
                structured_output_fields=structured_output_fields,
                prompt_cache_fields=prompt_cache_fields,
                prompt_cache_stable_message_count=prompt_cache_stable_message_count,
                tool_request=tool_request,
                options=kw,
            ),
        )

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
                    f"Model output was not valid JSON (line {exc.lineno}, column {exc.colno})."
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
            provider_result = await self._provider_retry.execute(
                lambda: self._image_dispatch(
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
                ),
                provider=self.provider,
                model=model,
                operation="image",
                rate_limit_group=self.rate_limit_group,
            )
            result = provider_result.value

            latency_ms = int((time.perf_counter() - start) * 1000)
            await self._account_llm_usage(
                model=model,
                usage=result.usage or {},
                latency_ms=latency_ms,
            )
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
    ) -> ProviderCallResult[ImageGenerationResult]:
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
    # Internals
    # ================================================================
    def _headers_openai_like(self):
        hdr = {"Content-Type": "application/json"}
        if self.provider in {"openai", "openrouter", "deepseek"} or (
            self.provider == "openai_compatible" and self.api_key
        ):
            hdr["Authorization"] = f"Bearer {self.api_key}"
        return hdr

    async def aclose(self) -> None:
        """Close active and safely retired provider HTTP clients.

        Intro:
            Connection hot reload preserves earlier transports for in-flight
            calls, so shutdown owns cleanup for the complete client lifetime.

        Examples:
            Close one client directly:
                ```python
                await client.aclose()
                ```

            Close every client through its service:
                ```python
                await service.aclose()
                ```

        Args:
            None.

        Returns:
            None: Closes every reachable HTTP transport.

        Notes:
            A transport bound to an already-closed event loop is logged and
            skipped so one retired connection cannot block remaining cleanup.
        """
        clients = [self._client, *self._retired_http_clients]
        self._retired_http_clients = []
        seen: set[int] = set()
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            try:
                await client.aclose()
            except RuntimeError as exc:
                self._logger.warning("llm_http_client_close_failed: %s", exc)

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
            metadata = checked_response_metadata(self.provider, self.model, "raw", r)
            return ProviderCallResult(r if return_response else r.json(), metadata)

        result = await self._provider_retry.execute(
            _call,
            provider=self.provider,
            model=self.model,
            operation="raw",
            rate_limit_group=self.rate_limit_group,
        )
        return result.value


# Convenience factory
def llm_from_env() -> GenericLLMClient:
    return GenericLLMClient()

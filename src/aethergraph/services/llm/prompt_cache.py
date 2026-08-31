"""Provider-neutral prompt-cache preparation and provider capability translation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal

from .catalog import resolve_model_catalog_capability_entry
from .profiles import PromptCachePolicy
from .registry import resolve_endpoint_adapter
from .tool_calling import (
    ToolCallRequest,
    tool_call_request_fingerprint,
    tool_call_surface_fingerprint,
)
from .types import LLMUnsupportedFeatureError, PromptCacheRequest

PromptCacheMode = Literal["disabled", "explicit", "implicit", "unavailable"]
_OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64
_PROMPT_CACHE_KEY_PREFIX = "agpc_"

_CACHE_SCOPE_KEYS = (
    "tenant_id",
    "project_id",
    "org_id",
    "user_id",
    "graph_id",
    "agent_id",
    "scope_key",
)


@dataclass(frozen=True)
class PreparedPromptCache:
    """Internal immutable result of prompt-cache capability preparation."""

    messages: tuple[dict[str, Any], ...]
    provider_request_fields: dict[str, Any]
    stable_message_count: int
    observation: dict[str, Any]


@dataclass(frozen=True)
class _PromptCacheCapability:
    """Provider cache-marker semantics resolved for one model family."""

    mode: PromptCacheMode
    capability_source: str
    max_total_boundaries: int | None = None
    max_new_writes_per_request: int | None = None


def prepare_prompt_cache(
    request: PromptCacheRequest,
    messages: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    scope_dimensions: dict[str, Any] | None = None,
    tool_request: ToolCallRequest | None = None,
    policy: PromptCachePolicy = "auto",
    endpoint_id: str | None = None,
) -> PreparedPromptCache:
    """Prepare one provider-neutral stable-prefix cache request.

    Intro:
        The function validates caller boundaries, derives an opaque provider key,
        and translates only capabilities declared for the selected provider,
        model, and endpoint. Caller-owned values remain unchanged.

    Examples:
        Prepare explicit OpenAI boundaries:
            ```python
            prepared = prepare_prompt_cache(
                PromptCacheRequest((0,), "agent.header.v1"),
                [{"role": "system", "content": "Stable"}],
                provider="openai",
                model="gpt-5.6",
            )
            assert prepared.observation["effective_mode"] == "explicit"
            ```

        Preserve an unknown provider request without cache fields:
            ```python
            prepared = prepare_prompt_cache(
                PromptCacheRequest((0,), "agent.header.v1"),
                [{"role": "system", "content": "Stable"}],
                provider="custom",
                model="custom-model",
            )
            assert prepared.provider_request_fields == {}
            ```

    Args:
        request: Validated provider-neutral prompt-cache request.
        messages: Final provider-neutral message list for the LLM call.
        provider: Configured provider identifier.
        model: Configured provider model identifier.
        scope_dimensions: Optional non-secret execution dimensions used only
            to isolate the opaque provider key.
        tool_request: Exact provider-visible native Tool contract, when used.
        policy: Profile-owned disabled, automatic, or required cache policy.
        endpoint_id: Optional endpoint adapter selected before request inspection.

    Returns:
        PreparedPromptCache: Detached translated messages, provider fields, and
        redacted observation metadata.

    Notes:
        Provider request rejection is handled by the ordinary call failure
        path. This preparation never creates a cache-disabled retry.
    """

    _validate_message_indexes(request.stable_message_indexes, messages)
    if policy not in {"disabled", "auto", "required"}:
        raise ValueError("prompt-cache policy must be disabled, auto, or required")
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    stable_message_count = request.stable_message_indexes[-1] + 1
    tool_contract_fingerprint = tool_call_request_fingerprint(tool_request)
    tool_surface_fingerprint = tool_call_surface_fingerprint(tool_request)
    if policy == "disabled":
        return PreparedPromptCache(
            messages=tuple(copy.deepcopy(messages)),
            provider_request_fields={},
            stable_message_count=stable_message_count,
            observation=_cache_observation(
                request=request,
                mode="disabled",
                capability_source="profile_policy_disabled",
                effective_boundary_count=0,
                key_fingerprint="",
                tool_request=tool_request,
                tool_contract_fingerprint=tool_contract_fingerprint,
                tool_surface_fingerprint=tool_surface_fingerprint,
            ),
        )
    capability = _resolve_capability(
        normalized_provider,
        normalized_model,
        endpoint_id=endpoint_id,
    )
    if policy == "required" and capability.mode == "unavailable":
        raise LLMUnsupportedFeatureError(
            normalized_provider,
            normalized_model,
            "prompt_cache",
            "profile policy requires a cataloged cache capability",
        )
    if capability.mode != "explicit":
        selected_indexes: tuple[int, ...] = ()
    elif capability.max_total_boundaries is None:
        selected_indexes = request.stable_message_indexes
    else:
        selected_indexes = _select_boundaries(
            request.stable_message_indexes,
            limit=capability.max_total_boundaries,
        )
    translated_messages = copy.deepcopy(messages)
    provider_fields: dict[str, Any] = {}
    implicit_latest_breakpoint = (
        normalized_provider == "openai"
        and capability.mode == "explicit"
        and endpoint_id in {None, "openai_responses"}
        and tool_request is not None
    )
    key = _derive_cache_key(
        provider=normalized_provider,
        model=normalized_model,
        prefix_family=request.prefix_family,
        scope_dimensions=scope_dimensions,
        tool_contract_fingerprint=tool_contract_fingerprint,
    )

    if normalized_provider == "openai" and capability.mode == "explicit":
        translated_messages = _mark_openai_boundaries(
            translated_messages,
            selected_indexes,
        )
        provider_fields = _openai_explicit_fields(key)
        if implicit_latest_breakpoint:
            # A Responses continuation grows at the latest Tool/message boundary.
            # Keep Engine-owned stable breakpoints, but retain OpenAI's implicit
            # latest-boundary write instead of restricting writes to explicit-only.
            provider_fields.pop("prompt_cache_options")
    elif normalized_provider == "anthropic" and capability.mode == "explicit":
        translated_messages = _mark_anthropic_boundaries(
            translated_messages,
            selected_indexes,
        )

    observation = _cache_observation(
        request=request,
        mode=capability.mode,
        capability_source=capability.capability_source,
        effective_boundary_count=(len(selected_indexes) if capability.mode == "explicit" else 0),
        key_fingerprint=hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
        tool_request=tool_request,
        tool_contract_fingerprint=tool_contract_fingerprint,
        tool_surface_fingerprint=tool_surface_fingerprint,
    )
    if implicit_latest_breakpoint:
        observation["implicit_latest_breakpoint"] = True
    if capability.max_new_writes_per_request is not None:
        observation["max_new_writes_per_request"] = capability.max_new_writes_per_request
    return PreparedPromptCache(
        messages=tuple(translated_messages),
        provider_request_fields=provider_fields,
        stable_message_count=stable_message_count,
        observation=observation,
    )


def _cache_observation(
    *,
    request: PromptCacheRequest,
    mode: PromptCacheMode,
    capability_source: str,
    effective_boundary_count: int,
    key_fingerprint: str,
    tool_request: ToolCallRequest | None,
    tool_contract_fingerprint: str,
    tool_surface_fingerprint: str,
) -> dict[str, Any]:
    return {
        "strategy": request.strategy,
        "requested_boundary_count": len(request.stable_message_indexes),
        "effective_boundary_count": effective_boundary_count,
        "effective_mode": mode,
        "capability_source": capability_source,
        "key_fingerprint": key_fingerprint,
        "tool_contract_fingerprint": tool_contract_fingerprint[:16],
        "tool_catalog_fingerprint": tool_contract_fingerprint[:16],
        "tool_surface_fingerprint": tool_surface_fingerprint[:16],
        "tool_discovery_mode": (
            tool_request.discovery.mode
            if tool_request is not None and tool_request.discovery is not None
            else ""
        ),
    }


def _validate_message_indexes(
    indexes: tuple[int, ...],
    messages: list[dict[str, Any]],
) -> None:
    if not messages:
        raise ValueError("prompt cache requires at least one message")
    last_index = len(messages) - 1
    if indexes[-1] > last_index:
        raise ValueError(
            "prompt cache message index "
            f"{indexes[-1]} is outside the message list ending at {last_index}"
        )


def _resolve_capability(
    provider: str,
    model: str,
    *,
    endpoint_id: str | None,
) -> _PromptCacheCapability:
    try:
        endpoint = resolve_endpoint_adapter(
            provider,
            "chat",
            endpoint_id=endpoint_id,
        )
        entry = resolve_model_catalog_capability_entry(
            provider,
            model,
            "chat",
            endpoint.adapter_id,
            capability="prompt_cache",
        )
    except (KeyError, ValueError):
        if endpoint_id is not None:
            raise
        entry = None
    if entry is not None and entry.prompt_cache is not None:
        facts = entry.prompt_cache
        return _PromptCacheCapability(
            mode=facts.mode,
            capability_source=facts.capability_source,
            max_total_boundaries=facts.max_total_boundaries,
            max_new_writes_per_request=facts.max_new_writes_per_request,
        )
    return _PromptCacheCapability(
        mode="unavailable",
        capability_source="provider_model_unavailable",
    )


def _select_boundaries(indexes: tuple[int, ...], *, limit: int) -> tuple[int, ...]:
    if limit <= 0:
        return ()
    if len(indexes) <= limit:
        return indexes
    selected = {indexes[0], indexes[-1]}
    for index in reversed(indexes[1:-1]):
        if len(selected) >= limit:
            break
        selected.add(index)
    return tuple(index for index in indexes if index in selected)


def _derive_cache_key(
    *,
    provider: str,
    model: str,
    prefix_family: str,
    scope_dimensions: dict[str, Any] | None,
    tool_contract_fingerprint: str,
) -> str:
    dimensions = dict(scope_dimensions or {})
    stable_scope = {
        key: str(dimensions[key]) for key in _CACHE_SCOPE_KEYS if dimensions.get(key) is not None
    }
    canonical = json.dumps(
        {
            "provider": provider,
            "model": model,
            "prefix_family": prefix_family,
            "scope": stable_scope,
            "tool_contract_fingerprint": tool_contract_fingerprint,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest_length = _OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH - len(_PROMPT_CACHE_KEY_PREFIX)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:digest_length]
    return _PROMPT_CACHE_KEY_PREFIX + digest


def _openai_explicit_fields(key: str) -> dict[str, Any]:
    if not key or len(key) > _OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        raise ValueError(
            "OpenAI prompt_cache_key must contain between 1 and "
            f"{_OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH} characters"
        )
    return {
        "prompt_cache_key": key,
        "prompt_cache_options": {"mode": "explicit"},
    }


def _mark_openai_boundaries(
    messages: list[dict[str, Any]],
    indexes: tuple[int, ...],
) -> list[dict[str, Any]]:
    for index in indexes:
        message = messages[index]
        role = str(message.get("role") or "user")
        content = message.get("content")
        marker = {"mode": "explicit"}
        if isinstance(content, str):
            message["content"] = [
                {
                    "type": ("output_text" if role == "assistant" else "input_text"),
                    "text": content,
                    "prompt_cache_breakpoint": marker,
                }
            ]
            continue
        if isinstance(content, list) and content:
            last = copy.deepcopy(content[-1])
            if not isinstance(last, dict):
                raise TypeError(
                    f"prompt cache boundary message {index} has a non-object final content block"
                )
            last["prompt_cache_breakpoint"] = marker
            message["content"] = [*copy.deepcopy(content[:-1]), last]
            continue
        raise ValueError(f"prompt cache boundary message {index} has no markable content")
    return messages


def _mark_anthropic_boundaries(
    messages: list[dict[str, Any]],
    indexes: tuple[int, ...],
) -> list[dict[str, Any]]:
    for index in indexes:
        messages[index]["cache_control"] = {"type": "ephemeral"}
    return messages


__all__ = [
    "PreparedPromptCache",
    "PromptCacheMode",
    "prepare_prompt_cache",
]

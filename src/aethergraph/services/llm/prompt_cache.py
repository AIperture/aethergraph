"""Provider-neutral prompt-cache preparation and provider capability translation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Literal

from .types import PromptCacheRequest

PromptCacheMode = Literal["explicit", "implicit", "unavailable"]

_OPENAI_EXPLICIT_BOUNDARY_LIMIT = 4
_ANTHROPIC_BOUNDARY_LIMIT = 4
_OPENAI_VERSION_PATTERN = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?")
_CACHE_SCOPE_KEYS = ("org_id", "app_id", "agent_id")


@dataclass(frozen=True)
class PreparedPromptCache:
    """Internal immutable result of prompt-cache capability preparation."""

    messages: tuple[dict[str, Any], ...]
    provider_request_fields: dict[str, Any]
    observation: dict[str, Any]


def prepare_prompt_cache(
    request: PromptCacheRequest,
    messages: list[dict[str, Any]],
    *,
    provider: str,
    model: str,
    scope_dimensions: dict[str, Any] | None = None,
) -> PreparedPromptCache:
    """
    Prepare one provider-neutral stable-prefix cache request.

    The function validates caller boundaries, derives an opaque provider key,
    and translates only capabilities declared for the selected provider and
    model. The caller-owned messages and request remain unchanged.

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

    Returns:
        PreparedPromptCache: Detached translated messages, provider fields, and
        redacted observation metadata.

    Notes:
        Provider request rejection is handled by the ordinary call failure
        path. This preparation never creates a cache-disabled retry.
    """

    _validate_message_indexes(request.stable_message_indexes, messages)
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip().lower()
    mode, limit, capability_source = _resolve_capability(
        normalized_provider,
        normalized_model,
    )
    selected_indexes = _select_boundaries(
        request.stable_message_indexes,
        limit=limit,
    )
    translated_messages = copy.deepcopy(messages)
    provider_fields: dict[str, Any] = {}
    key = _derive_cache_key(
        provider=normalized_provider,
        model=normalized_model,
        prefix_family=request.prefix_family,
        scope_dimensions=scope_dimensions,
    )

    if normalized_provider == "openai" and mode == "explicit":
        translated_messages = _mark_openai_boundaries(
            translated_messages,
            selected_indexes,
        )
        provider_fields = {
            "prompt_cache_key": key,
            "prompt_cache_options": {"mode": "explicit"},
        }
    elif normalized_provider == "anthropic" and mode == "explicit":
        translated_messages = _mark_anthropic_boundaries(
            translated_messages,
            selected_indexes,
        )

    observation = {
        "strategy": request.strategy,
        "requested_boundary_count": len(request.stable_message_indexes),
        "effective_boundary_count": len(selected_indexes) if mode == "explicit" else 0,
        "effective_mode": mode,
        "capability_source": capability_source,
        "key_fingerprint": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
    }
    return PreparedPromptCache(
        messages=tuple(translated_messages),
        provider_request_fields=provider_fields,
        observation=observation,
    )


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


def _resolve_capability(provider: str, model: str) -> tuple[PromptCacheMode, int, str]:
    if provider == "openai":
        match = _OPENAI_VERSION_PATTERN.match(model)
        if match is not None:
            major = int(match.group("major"))
            minor = int(match.group("minor") or 0)
            if major > 5 or (major == 5 and minor >= 6):
                return (
                    "explicit",
                    _OPENAI_EXPLICIT_BOUNDARY_LIMIT,
                    "openai_explicit_model_family",
                )
        return "implicit", 0, "openai_conservative_implicit"
    if provider == "anthropic" and model.startswith("claude-"):
        return "explicit", _ANTHROPIC_BOUNDARY_LIMIT, "anthropic_messages"
    if provider == "google" and model.startswith("gemini-"):
        return "implicit", 0, "gemini_generate_content"
    return "unavailable", 0, "provider_model_unavailable"


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
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "agpc_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _mark_openai_boundaries(
    messages: list[dict[str, Any]],
    indexes: tuple[int, ...],
) -> list[dict[str, Any]]:
    for index in indexes:
        message = messages[index]
        content = message.get("content")
        marker = {"mode": "explicit"}
        if isinstance(content, str):
            message["content"] = [
                {
                    "type": "input_text",
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

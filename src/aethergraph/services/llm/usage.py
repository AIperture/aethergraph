from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

UsageAvailability = Literal["complete", "partial", "unavailable"]


@dataclass(frozen=True)
class ModelUsage:
    """Represent normalized model usage without conflating missing data with zero.

    The value keeps canonical token components separate from the provider's raw
    receipt. Numeric fields are `None` when the provider did not report them.

    Examples:
        Preserve unavailable usage explicitly:
            ```python
            usage = ModelUsage.unavailable()
            assert usage.total_input_tokens is None
            ```

        Normalize a complete provider receipt:
            ```python
            usage = ModelUsage.from_provider_usage(
                {"prompt_tokens": 5, "completion_tokens": 1}
            )
            assert usage.availability == "complete"
            ```

    Args:
        availability: Whether the provider receipt is complete, partial, or
            unavailable.
        total_input_tokens: Total cache-inclusive input tokens when known.
        uncached_input_tokens: Input tokens that were not cache reads or writes.
        cache_read_tokens: Input tokens served from a provider cache.
        cache_write_tokens: Input tokens written into a provider cache.
        output_tokens: Generated output tokens when known.
        reasoning_tokens: Provider-reported reasoning tokens when known.
        provider_usage_raw: Detached provider usage receipt for diagnostics.

    Returns:
        ModelUsage: An immutable normalized usage value.

    Notes:
        Use `unavailable()` for responses that contain no provider usage receipt.
    """

    availability: UsageAvailability
    total_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    provider_usage_raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach one normalized usage value.

        Validation rejects negative counters and contradictory availability
        states before the value reaches quota, budget, or metering consumers.

        Examples:
            Validate a complete value through construction:
                ```python
                usage = ModelUsage(
                    availability="complete",
                    total_input_tokens=1,
                    output_tokens=1,
                )
                assert usage.output_tokens == 1
                ```

            Reject a negative counter:
                ```python
                try:
                    ModelUsage(availability="partial", output_tokens=-1)
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized usage value.

        Returns:
            None: The frozen value is normalized in place.

        Notes:
            None.
        """

        if self.availability not in {"complete", "partial", "unavailable"}:
            raise ValueError("usage availability must be complete, partial, or unavailable")
        numeric_fields = (
            "total_input_tokens",
            "uncached_input_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_tokens",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"usage {name} must be an integer or None")
            if value < 0:
                raise ValueError(f"usage {name} must be non-negative")
        if not isinstance(self.provider_usage_raw, dict):
            raise TypeError("provider_usage_raw must be an object")
        if self.availability == "unavailable" and any(
            getattr(self, name) is not None for name in numeric_fields
        ):
            raise ValueError("unavailable usage cannot contain normalized token counts")
        if self.availability == "complete" and (
            self.total_input_tokens is None or self.output_tokens is None
        ):
            raise ValueError("complete usage requires total input and output token counts")
        if self.availability == "partial" and not any(
            getattr(self, name) is not None for name in numeric_fields
        ):
            raise ValueError("partial usage requires at least one normalized token count")
        object.__setattr__(self, "provider_usage_raw", deepcopy(self.provider_usage_raw))

    @classmethod
    def unavailable(cls) -> ModelUsage:
        """Create an explicit unavailable-usage value.

        This constructor is used when a provider response, cancellation, or
        transport failure supplies no trustworthy usage receipt.

        Examples:
            Build unavailable usage:
                ```python
                usage = ModelUsage.unavailable()
                assert usage.availability == "unavailable"
                ```

            Distinguish unavailable usage from a zero-token receipt:
                ```python
                usage = ModelUsage.unavailable()
                assert usage.total_input_tokens is None
                ```

        Args:
            cls: The `ModelUsage` class.

        Returns:
            ModelUsage: A value with no normalized token counters.

        Notes:
            Unavailable usage is not billing truth and must not be coerced to zero.
        """

        return cls(availability="unavailable")

    @classmethod
    def from_provider_usage(
        cls,
        usage: Mapping[str, Any] | None,
        *,
        availability: UsageAvailability | None = None,
    ) -> ModelUsage:
        """Normalize one provider receipt into the typed usage contract.

        Availability is inferred from the token fields unless the adapter has
        stronger evidence and supplies an explicit state.

        Examples:
            Normalize a complete OpenAI-style receipt:
                ```python
                usage = ModelUsage.from_provider_usage(
                    {"prompt_tokens": 10, "completion_tokens": 2}
                )
                assert usage.total_input_tokens == 10
                ```

            Preserve a partial provider receipt:
                ```python
                usage = ModelUsage.from_provider_usage({"input_tokens": 4})
                assert usage.availability == "partial"
                ```

        Args:
            cls: The `ModelUsage` class.
            usage: Optional provider usage mapping.
            availability: Optional adapter-owned completeness classification.

        Returns:
            ModelUsage: Detached canonical counters and the raw provider receipt.

        Notes:
            An empty or absent receipt produces `unavailable` usage.
        """

        raw = dict(usage or {})
        if not raw:
            if availability not in {None, "unavailable"}:
                raise ValueError("non-unavailable usage requires a provider receipt")
            return cls.unavailable()

        normalized = normalize_llm_usage(raw)
        has_input = _has_any_key(
            raw,
            "prompt_tokens",
            "input_tokens",
            "cache_read_tokens",
            "cache_read_input_tokens",
            "cache_write_tokens",
            "cache_creation_input_tokens",
        )
        has_output = _has_any_key(raw, "completion_tokens", "output_tokens")
        inferred: UsageAvailability
        if has_input and has_output:
            inferred = "complete"
        elif has_input or has_output:
            inferred = "partial"
        else:
            inferred = "unavailable"
        effective = availability or inferred
        if effective == "unavailable":
            return cls(availability="unavailable", provider_usage_raw=raw)

        uncached = normalized["uncached_input_tokens"] if has_input else None
        cache_read = normalized["cache_read_tokens"] if has_input else None
        cache_write = normalized["cache_write_tokens"] if has_input else None
        total_input = (
            uncached + cache_read + cache_write
            if uncached is not None and cache_read is not None and cache_write is not None
            else None
        )
        return cls(
            availability=effective,
            total_input_tokens=total_input,
            uncached_input_tokens=uncached,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=normalized["output_tokens"] if has_output else None,
            reasoning_tokens=_reasoning_tokens(raw),
            provider_usage_raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize usage for versioned observations and compatibility codecs.

        The serialization preserves unavailable counters as `None` and detaches
        the provider receipt from the immutable source value.

        Examples:
            Serialize unavailable usage:
                ```python
                payload = ModelUsage.unavailable().to_dict()
                assert payload["availability"] == "unavailable"
                ```

            Serialize complete usage:
                ```python
                usage = ModelUsage.from_provider_usage(
                    {"prompt_tokens": 3, "completion_tokens": 1}
                )
                assert usage.to_dict()["total_input_tokens"] == 3
                ```

        Args:
            self: Normalized usage value.

        Returns:
            dict[str, Any]: Detached JSON-compatible usage data.

        Notes:
            Compatibility projections may retain legacy field names separately.
        """

        return {
            "availability": self.availability,
            "total_input_tokens": self.total_input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "provider_usage_raw": deepcopy(self.provider_usage_raw),
        }


def normalize_llm_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Return provider-neutral LLM usage fields without mutating provider usage."""

    raw = dict(usage or {})
    input_tokens = _first_int(raw, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(raw, "completion_tokens", "output_tokens")
    cache_read_tokens = _first_int(
        raw,
        "cache_read_tokens",
        "cache_read_input_tokens",
    )
    cache_write_tokens = _first_int(
        raw,
        "cache_write_tokens",
        "cache_creation_input_tokens",
    )

    for details_key in ("prompt_tokens_details", "input_tokens_details"):
        details = raw.get(details_key)
        if isinstance(details, dict):
            cache_read_tokens = max(
                cache_read_tokens,
                _int_or_zero(details.get("cached_tokens")),
            )
            cache_write_tokens = max(
                cache_write_tokens,
                _first_int(
                    details,
                    "cache_write_tokens",
                    "cache_creation_tokens",
                    "cache_creation_input_tokens",
                ),
            )

    if "uncached_input_tokens" in raw:
        uncached_input_tokens = _int_or_zero(raw.get("uncached_input_tokens"))
    elif "prompt_tokens" in raw or "input_tokens_details" in raw:
        uncached_input_tokens = max(
            0,
            input_tokens - cache_read_tokens - cache_write_tokens,
        )
    else:
        # Anthropic reports non-cache input separately from cache read/write.
        uncached_input_tokens = input_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "provider_usage_raw": deepcopy(raw),
    }


def normalized_usage_metrics(normalized_usage: dict[str, Any] | None) -> dict[str, int]:
    """Return numeric normalized usage fields suitable for trace metrics."""

    usage = dict(normalized_usage or {})
    return {
        "input_tokens": _int_or_zero(usage.get("input_tokens")),
        "output_tokens": _int_or_zero(usage.get("output_tokens")),
        "cache_read_tokens": _int_or_zero(usage.get("cache_read_tokens")),
        "cache_write_tokens": _int_or_zero(usage.get("cache_write_tokens")),
        "uncached_input_tokens": _int_or_zero(usage.get("uncached_input_tokens")),
    }


def _first_int(raw: dict[str, Any], *keys: str) -> int:
    for key in keys:
        if key in raw:
            return _int_or_zero(raw.get(key))
    return 0


def _has_any_key(raw: Mapping[str, Any], *keys: str) -> bool:
    return any(key in raw for key in keys)


def _reasoning_tokens(raw: Mapping[str, Any]) -> int | None:
    if "reasoning_tokens" in raw:
        return _int_or_zero(raw.get("reasoning_tokens"))
    for details_key in ("completion_tokens_details", "output_tokens_details"):
        details = raw.get(details_key)
        if isinstance(details, Mapping) and "reasoning_tokens" in details:
            return _int_or_zero(details.get("reasoning_tokens"))
    return None


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "ModelUsage",
    "UsageAvailability",
    "normalize_llm_usage",
    "normalized_usage_metrics",
]

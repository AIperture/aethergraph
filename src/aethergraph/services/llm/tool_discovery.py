"""Provider-neutral deferred Tool discovery transport contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
import re
from typing import Any, Literal, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
ToolExposure = Literal["immediate", "deferred"]
ToolDiscoveryMode = Literal["native_hosted", "native_client", "engine_projected"]
ToolDiscoverySource = Literal["engine", "provider_hosted", "provider_client"]
ToolDiscoveryStatus = Literal["completed", "failed"]
ToolReplayRequirement = Literal["none", "previous_response", "full_history"]
ToolResultLimitBehavior = Literal["request_bound", "provider_fixed"]

_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9_.:/-]{1,240}$")
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _bounded_text(value: str, *, field_name: str, maximum: int) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} must not exceed {maximum} characters")
    return normalized


def _reference(value: str, *, field_name: str) -> str:
    normalized = _bounded_text(value, field_name=field_name, maximum=240)
    if _REFERENCE_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} contains unsupported characters")
    return normalized


def _json_mapping(
    value: dict[str, Any],
    *,
    field_name: str,
    maximum_bytes: int,
    maximum_depth: int,
    maximum_items: int,
    maximum_string: int,
) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain JSON-compatible values") from exc
    if len(encoded) > maximum_bytes:
        raise ValueError(f"{field_name} must not exceed {maximum_bytes} UTF-8 bytes")

    item_count = 0

    def visit(item: JSONValue, *, depth: int) -> None:
        nonlocal item_count
        if depth > maximum_depth:
            raise ValueError(f"{field_name} must not exceed depth {maximum_depth}")
        if isinstance(item, str) and len(item) > maximum_string:
            raise ValueError(f"{field_name} strings must not exceed {maximum_string} characters")
        if isinstance(item, dict):
            item_count += len(item)
            for key, nested in item.items():
                if len(str(key)) > maximum_string:
                    raise ValueError(
                        f"{field_name} keys must not exceed {maximum_string} characters"
                    )
                visit(nested, depth=depth + 1)
        elif isinstance(item, list):
            item_count += len(item)
            for nested in item:
                visit(nested, depth=depth + 1)
        if item_count > maximum_items:
            raise ValueError(f"{field_name} must not exceed {maximum_items} items")

    visit(value, depth=1)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class ToolNamespace:
    """Describe one provider-neutral Tool namespace."""

    name: str
    description: str

    def __post_init__(self) -> None:
        """Validate and normalize one Tool namespace.

        The namespace remains independent from provider-specific grouping and
        reference syntax.

        Examples:
            Create a namespace:
                ```python
                namespace = ToolNamespace("change", "Project change Tools.")
                assert namespace.name == "change"
                ```

            Reject an empty description:
                ```python
                try:
                    ToolNamespace("change", "")
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized namespace.

        Returns:
            None: Normalizes the frozen namespace value.

        Notes:
            Provider adapters may project namespaces differently without
            changing this contract.
        """

        object.__setattr__(self, "name", _reference(self.name, field_name="namespace name"))
        object.__setattr__(
            self,
            "description",
            _bounded_text(
                self.description,
                field_name="namespace description",
                maximum=1_000,
            ),
        )


@dataclass(frozen=True)
class ToolDiscoveryRequest:
    """Select one exact deferred Tool discovery mode for a model request."""

    mode: ToolDiscoveryMode
    max_results: int = 5

    def __post_init__(self) -> None:
        """Validate one bounded discovery request.

        The request selects transport semantics; it does not authorize Tools
        or permit providers to choose a different mode.

        Examples:
            Select Engine-projected discovery:
                ```python
                request = ToolDiscoveryRequest(mode="engine_projected")
                assert request.max_results == 5
                ```

            Bound hosted results:
                ```python
                request = ToolDiscoveryRequest(
                    mode="native_hosted", max_results=10
                )
                assert request.mode == "native_hosted"
                ```

        Args:
            self: Newly initialized discovery request.

        Returns:
            None: Validates the selected mode and result bound.

        Notes:
            Provider-hosted services may impose a lower limit. Adapters must
            report that exact capability during binding.
        """

        if self.mode not in {"native_hosted", "native_client", "engine_projected"}:
            raise ValueError("Tool discovery mode is unsupported")
        if not 1 <= int(self.max_results) <= 50:
            raise ValueError("Tool discovery max_results must be between 1 and 50")
        object.__setattr__(self, "max_results", int(self.max_results))


@dataclass(frozen=True)
class ToolDiscoveryError:
    """Carry one bounded provider-neutral discovery failure."""

    code: str
    summary: str
    retryable: bool = False
    details: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach one discovery error.

        The value contains safe transport diagnostics only and excludes raw
        provider bodies, tracebacks, and credentials.

        Examples:
            Describe a stale reference:
                ```python
                error = ToolDiscoveryError(
                    code="tool_reference_stale",
                    summary="The Tool reference is no longer current.",
                )
                assert not error.retryable
                ```

            Include bounded safe details:
                ```python
                error = ToolDiscoveryError(
                    code="search_unavailable",
                    summary="Search is temporarily unavailable.",
                    retryable=True,
                    details={"provider_status": "unavailable"},
                )
                assert error.details["provider_status"] == "unavailable"
                ```

        Args:
            self: Newly initialized discovery error.

        Returns:
            None: Normalizes and detaches the error payload.

        Notes:
            Applications may apply stricter allowlists before presenting this
            value to a user.
        """

        object.__setattr__(self, "code", _reference(self.code, field_name="error code"))
        object.__setattr__(
            self,
            "summary",
            _bounded_text(self.summary, field_name="error summary", maximum=2_000),
        )
        object.__setattr__(self, "retryable", bool(self.retryable))
        object.__setattr__(
            self,
            "details",
            _json_mapping(
                self.details,
                field_name="error details",
                maximum_bytes=16 * 1024,
                maximum_depth=8,
                maximum_items=128,
                maximum_string=4_000,
            ),
        )


@dataclass(frozen=True)
class ToolDiscoveryEvent:
    """Normalize one ordered Tool discovery result or failure."""

    event_id: str
    mode: ToolDiscoveryMode
    source: ToolDiscoverySource
    arguments: dict[str, JSONValue]
    tool_refs: tuple[str, ...] = ()
    query: str | None = None
    status: ToolDiscoveryStatus = "completed"
    error: ToolDiscoveryError | None = None
    provider_reference_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate one normalized discovery event.

        The event preserves provider arguments without pretending every search
        mode has a natural-language query, score, or Engine ranking reason.

        Examples:
            Record Engine search references:
                ```python
                event = ToolDiscoveryEvent(
                    event_id="search_1",
                    mode="engine_projected",
                    source="engine",
                    arguments={"query": "read a document"},
                    query="read a document",
                    tool_refs=("docs.read",),
                )
                assert event.tool_refs == ("docs.read",)
                ```

            Record a hosted search failure:
                ```python
                event = ToolDiscoveryEvent(
                    event_id="search_2",
                    mode="native_hosted",
                    source="provider_hosted",
                    arguments={"paths": ["docs"]},
                    status="failed",
                    error=ToolDiscoveryError(
                        code="search_failed", summary="Search failed."
                    ),
                )
                assert event.error is not None
                ```

        Args:
            self: Newly initialized discovery event.

        Returns:
            None: Validates ordering data and detaches mutable payloads.

        Notes:
            Engine ranking evidence belongs to Engine search result data, not
            this cross-provider transport envelope.
        """

        if self.mode not in {"native_hosted", "native_client", "engine_projected"}:
            raise ValueError("Tool discovery event mode is unsupported")
        if self.source not in {"engine", "provider_hosted", "provider_client"}:
            raise ValueError("Tool discovery event source is unsupported")
        if self.status not in {"completed", "failed"}:
            raise ValueError("Tool discovery event status is unsupported")
        if self.status == "failed" and self.error is None:
            raise ValueError("Failed Tool discovery events require an error")
        if self.status == "completed" and self.error is not None:
            raise ValueError("Completed Tool discovery events cannot carry an error")
        event_id = _reference(self.event_id, field_name="discovery event id")
        tool_refs = tuple(_reference(item, field_name="tool reference") for item in self.tool_refs)
        if len(tool_refs) > 50:
            raise ValueError("Tool discovery event must not exceed 50 Tool references")
        if len(tool_refs) != len(set(tool_refs)):
            raise ValueError("Tool discovery event references must be unique")
        provider_refs = tuple(
            _reference(item, field_name="provider reference id")
            for item in self.provider_reference_ids
        )
        if len(provider_refs) > 50:
            raise ValueError("Tool discovery event must not exceed 50 provider references")
        if len(provider_refs) != len(set(provider_refs)):
            raise ValueError("Provider reference ids must be unique")
        query = None if self.query is None else str(self.query).strip()
        if query is not None and len(query) > 2_000:
            raise ValueError("Tool discovery query must not exceed 2000 characters")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "arguments",
            _json_mapping(
                self.arguments,
                field_name="discovery arguments",
                maximum_bytes=32 * 1024,
                maximum_depth=8,
                maximum_items=256,
                maximum_string=4_000,
            ),
        )
        object.__setattr__(self, "tool_refs", tool_refs)
        object.__setattr__(self, "query", query or None)
        object.__setattr__(self, "provider_reference_ids", provider_refs)


@dataclass(frozen=True)
class ToolTransportCheckpoint:
    """Preserve opaque provider replay state for one semantic turn."""

    checkpoint_id: str
    revision: int
    provider: str
    model: str
    contract_version: str
    turn_id: str
    integrity_digest: str
    expires_at: Literal["end_of_turn"] = "end_of_turn"
    opaque_payload: dict[str, JSONValue] | None = None
    durable_ref: str | None = None

    def __post_init__(self) -> None:
        """Validate and detach one opaque replay checkpoint.

        A checkpoint carries either an in-memory JSON payload, a durable
        storage reference, or both. Consumers transport it without interpreting
        provider-native item structure.

        Examples:
            Preserve an inline provider item:
                ```python
                checkpoint = ToolTransportCheckpoint(
                    checkpoint_id="checkpoint_1",
                    revision=1,
                    provider="openai",
                    model="example-model",
                    contract_version="responses/v1",
                    turn_id="turn_1",
                    integrity_digest="0" * 64,
                    opaque_payload={"output": []},
                )
                assert checkpoint.opaque_payload == {"output": []}
                ```

            Preserve a durable reference:
                ```python
                checkpoint = ToolTransportCheckpoint(
                    checkpoint_id="checkpoint_2",
                    revision=2,
                    provider="anthropic",
                    model="example-model",
                    contract_version="messages/v1",
                    turn_id="turn_1",
                    integrity_digest="f" * 64,
                    durable_ref="artifact:checkpoint_2",
                )
                assert checkpoint.durable_ref is not None
                ```

        Args:
            self: Newly initialized transport checkpoint.

        Returns:
            None: Validates identity, integrity, and storage representation.

        Notes:
            Raw checkpoint data is private transport state and must not be
            copied into prompts, user-visible events, or exception messages.
        """

        for attribute, label in (
            ("checkpoint_id", "checkpoint id"),
            ("provider", "checkpoint provider"),
            ("model", "checkpoint model"),
            ("contract_version", "checkpoint contract version"),
            ("turn_id", "checkpoint turn id"),
        ):
            object.__setattr__(
                self,
                attribute,
                _reference(getattr(self, attribute), field_name=label),
            )
        revision = int(self.revision)
        if revision < 1:
            raise ValueError("Checkpoint revision must be at least 1")
        if _DIGEST_PATTERN.fullmatch(str(self.integrity_digest or "")) is None:
            raise ValueError("Checkpoint integrity_digest must be lowercase SHA-256")
        if self.expires_at != "end_of_turn":
            raise ValueError("Checkpoint expiration must be end_of_turn")
        durable_ref = None
        if self.durable_ref is not None:
            durable_ref = _reference(self.durable_ref, field_name="checkpoint durable ref")
        payload = None
        if self.opaque_payload is not None:
            payload = _json_mapping(
                self.opaque_payload,
                field_name="checkpoint opaque payload",
                maximum_bytes=256 * 1024,
                maximum_depth=16,
                maximum_items=4_096,
                maximum_string=32_000,
            )
        if payload is None and durable_ref is None:
            raise ValueError("Checkpoint requires opaque_payload or durable_ref")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "integrity_digest", str(self.integrity_digest))
        object.__setattr__(self, "opaque_payload", payload)
        object.__setattr__(self, "durable_ref", durable_ref)


@dataclass(frozen=True)
class ToolDiscoveryModeCapability:
    """Declare discovery behavior for one exact mode on a model binding."""

    mode: ToolDiscoveryMode
    replay_requirement: ToolReplayRequirement = "none"
    result_limit_behavior: ToolResultLimitBehavior = "request_bound"
    max_results: int = 5
    protocol_version: str = ""

    def __post_init__(self) -> None:
        """
        Validate one mode-specific provider capability.

        Validation freezes replay semantics, result-limit behavior, and the
        exact protocol version on the mode that owns them.

        Examples:
            Validate a request-bound mode:
                ```python
                capability = ToolDiscoveryModeCapability(
                    mode="engine_projected",
                    max_results=8,
                )
                assert capability.max_results == 8
                ```

            Validate a provider-fixed mode:
                ```python
                capability = ToolDiscoveryModeCapability(
                    mode="native_hosted",
                    result_limit_behavior="provider_fixed",
                    max_results=5,
                )
                assert capability.result_limit_behavior == "provider_fixed"
                ```

        Args:
            self: Newly initialized mode capability.

        Returns:
            None: Normalizes and validates the frozen capability.

        Notes:
            Provider-fixed limits describe provider behavior; they do not
            claim the request can enforce a smaller result count.
        """

        if self.mode not in {"native_hosted", "native_client", "engine_projected"}:
            raise ValueError("Capability mode is unsupported")
        if self.replay_requirement not in {"none", "previous_response", "full_history"}:
            raise ValueError("Capability replay_requirement is unsupported")
        if self.result_limit_behavior not in {"request_bound", "provider_fixed"}:
            raise ValueError("Capability result_limit_behavior is unsupported")
        max_results = int(self.max_results)
        if not 1 <= max_results <= 50:
            raise ValueError("Capability max_results must be between 1 and 50")
        object.__setattr__(self, "max_results", max_results)
        object.__setattr__(
            self,
            "protocol_version",
            str(self.protocol_version or "").strip(),
        )

    def supports(self, request: ToolDiscoveryRequest) -> bool:
        """
        Return whether this mode can honor the requested result bound.

        Request-bound modes accept requests at or below their maximum.
        Provider-fixed modes accept only bounds that are no smaller than the
        provider's fixed maximum output.

        Examples:
            Check a request-bound mode:
                ```python
                capability = ToolDiscoveryModeCapability(
                    mode="engine_projected",
                    max_results=8,
                )
                assert capability.supports(
                    ToolDiscoveryRequest("engine_projected", 5)
                )
                ```

            Reject an unenforceable provider-fixed bound:
                ```python
                capability = ToolDiscoveryModeCapability(
                    mode="native_hosted",
                    result_limit_behavior="provider_fixed",
                    max_results=5,
                )
                assert not capability.supports(
                    ToolDiscoveryRequest("native_hosted", 4)
                )
                ```

        Args:
            request: Exact discovery mode and requested result bound.

        Returns:
            bool: True when this mode can satisfy the requested semantics.

        Notes:
            The method never substitutes another mode or relaxes a bound.
        """

        if not isinstance(request, ToolDiscoveryRequest):
            raise TypeError("Discovery capability checks require ToolDiscoveryRequest")
        if request.mode != self.mode:
            return False
        if self.result_limit_behavior == "request_bound":
            return request.max_results <= self.max_results
        return request.max_results >= self.max_results


@dataclass(frozen=True)
class ToolDiscoveryCapabilities:
    """Declare discovery support for one exact provider/model binding."""

    provider: str
    model: str
    endpoint_family: str
    supported_modes: tuple[ToolDiscoveryModeCapability, ...]

    def __post_init__(self) -> None:
        """Validate one model-specific discovery capability record.

        Capability is attached to an exact binding and never inferred only
        from a provider name.

        Examples:
            Declare projected-only support:
                ```python
                capabilities = ToolDiscoveryCapabilities(
                    provider="google",
                    model="example-model",
                    endpoint_family="generateContent",
                    supported_modes=(
                        ToolDiscoveryModeCapability(mode="engine_projected"),
                    ),
                )
                assert capabilities.supported_modes[0].mode == "engine_projected"
                ```

            Declare replay requirements:
                ```python
                capabilities = ToolDiscoveryCapabilities(
                    provider="openai",
                    model="example-model",
                    endpoint_family="responses",
                    supported_modes=(
                        ToolDiscoveryModeCapability(
                            mode="native_client",
                            replay_requirement="previous_response",
                        ),
                    ),
                )
                assert capabilities.supported_modes[0].replay_requirement == (
                    "previous_response"
                )
                ```

        Args:
            self: Newly initialized capability record.

        Returns:
            None: Validates and normalizes the binding record.

        Notes:
            Provider adapters own the authoritative table of these records.
        """

        provider = _reference(self.provider, field_name="capability provider")
        model = _reference(self.model, field_name="capability model")
        endpoint = _reference(
            self.endpoint_family,
            field_name="capability endpoint family",
        )
        modes = tuple(self.supported_modes)
        if not modes or not all(isinstance(mode, ToolDiscoveryModeCapability) for mode in modes):
            raise TypeError(
                "Capability supported_modes must contain ToolDiscoveryModeCapability values"
            )
        mode_names = tuple(mode.mode for mode in modes)
        if len(mode_names) != len(set(mode_names)):
            raise ValueError("Capability supported_modes must be unique")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "endpoint_family", endpoint)
        object.__setattr__(self, "supported_modes", modes)

    def supports(self, request: ToolDiscoveryRequest) -> bool:
        """Return whether this exact binding supports one discovery request.

        The check includes the selected mode and the binding's exact result
        limit without attempting a provider call.

        Examples:
            Accept a supported request:
                ```python
                capabilities = ToolDiscoveryCapabilities(
                    provider="google",
                    model="example-model",
                    endpoint_family="generateContent",
                    supported_modes=(
                        ToolDiscoveryModeCapability(mode="engine_projected"),
                    ),
                )
                assert capabilities.supports(
                    ToolDiscoveryRequest("engine_projected")
                )
                ```

            Reject an unsupported mode:
                ```python
                capabilities = ToolDiscoveryCapabilities(
                    provider="google",
                    model="example-model",
                    endpoint_family="generateContent",
                    supported_modes=(
                        ToolDiscoveryModeCapability(mode="engine_projected"),
                    ),
                )
                assert not capabilities.supports(
                    ToolDiscoveryRequest("native_hosted")
                )
                ```

        Args:
            request: Requested discovery mode and result bound.

        Returns:
            bool: True only when both mode and result count are supported.

        Notes:
            This method performs no fallback or mode substitution.
        """

        if not isinstance(request, ToolDiscoveryRequest):
            raise TypeError("Discovery capability checks require ToolDiscoveryRequest")
        return any(mode.supports(request) for mode in self.supported_modes)


def resolve_tool_discovery_capabilities(
    provider: str,
    model: str,
    endpoint_family: str,
) -> ToolDiscoveryCapabilities | None:
    """Resolve one exact built-in provider discovery capability record.

    The registry contains only implemented and evidence-frozen bindings; it
    never expands support from a provider name or model family prefix.

    Examples:
        Resolve the implemented OpenAI client mode:
            ```python
            record = resolve_tool_discovery_capabilities(
                "openai", "gpt-5.6", "responses"
            )
            assert record is not None
            ```

        Reject an unregistered model:
            ```python
            record = resolve_tool_discovery_capabilities(
                "openai", "gpt-5.5", "responses"
            )
            assert record is None
            ```

    Args:
        provider: Exact normalized provider identifier.
        model: Exact model or deployment identifier.
        endpoint_family: Exact active transport endpoint family.

    Returns:
        ToolDiscoveryCapabilities | None: Exact implemented record, if present.

    Notes:
        Absent modes fail closed and are never substituted by an adapter.
    """

    binding = (
        str(provider or "").strip().lower(),
        str(model or "").strip(),
        str(endpoint_family or "").strip(),
    )
    if binding == ("openai", "gpt-5.6", "responses"):
        return ToolDiscoveryCapabilities(
            provider="openai",
            model="gpt-5.6",
            endpoint_family="responses",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="native_client",
                    replay_requirement="previous_response",
                    result_limit_behavior="request_bound",
                    max_results=50,
                    protocol_version="responses.tool_search",
                ),
            ),
        )
    if binding == ("google", "gemini-2.5-pro", "generateContent"):
        return ToolDiscoveryCapabilities(
            provider="google",
            model="gemini-2.5-pro",
            endpoint_family="generateContent",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="engine_projected",
                    replay_requirement="full_history",
                    result_limit_behavior="request_bound",
                    max_results=50,
                    protocol_version="generateContent.v1",
                ),
            ),
        )
    if binding == (
        "anthropic",
        "claude-sonnet-4-5-20250929",
        "messages",
    ):
        return ToolDiscoveryCapabilities(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            endpoint_family="messages",
            supported_modes=(
                ToolDiscoveryModeCapability(
                    mode="native_hosted",
                    replay_requirement="full_history",
                    result_limit_behavior="provider_fixed",
                    max_results=5,
                    protocol_version="tool_search_tool_bm25_20251119",
                ),
                ToolDiscoveryModeCapability(
                    mode="native_client",
                    replay_requirement="full_history",
                    result_limit_behavior="request_bound",
                    max_results=50,
                    protocol_version="messages.tool_reference",
                ),
            ),
        )
    return None


__all__ = [
    "JSONScalar",
    "JSONValue",
    "ToolDiscoveryCapabilities",
    "ToolDiscoveryError",
    "ToolDiscoveryEvent",
    "ToolDiscoveryMode",
    "ToolDiscoveryModeCapability",
    "ToolDiscoveryRequest",
    "ToolDiscoverySource",
    "ToolDiscoveryStatus",
    "ToolExposure",
    "ToolNamespace",
    "ToolReplayRequirement",
    "ToolResultLimitBehavior",
    "ToolTransportCheckpoint",
    "resolve_tool_discovery_capabilities",
]

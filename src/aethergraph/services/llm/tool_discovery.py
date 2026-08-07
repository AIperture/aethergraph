"""Provider-neutral deferred Tool discovery transport contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import re
from typing import Any, Literal, TypeAlias

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
ToolExposure = Literal["immediate", "deferred"]
ToolDiscoveryMode = Literal["native_hosted", "native_client", "engine_projected"]
ToolDiscoverySource = Literal["engine", "provider_hosted", "provider_client"]
ToolDiscoveryStatus = Literal["completed", "failed"]
ToolReplayRequirement = Literal["none", "previous_response", "full_history"]

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


def _json_mapping(value: dict[str, Any], *, field_name: str) -> dict[str, JSONValue]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    try:
        import json

        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain JSON-compatible values") from exc
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
            _json_mapping(self.details, field_name="error details"),
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
        if len(tool_refs) != len(set(tool_refs)):
            raise ValueError("Tool discovery event references must be unique")
        provider_refs = tuple(
            _reference(item, field_name="provider reference id")
            for item in self.provider_reference_ids
        )
        if len(provider_refs) != len(set(provider_refs)):
            raise ValueError("Provider reference ids must be unique")
        query = None if self.query is None else str(self.query).strip()
        if query is not None and len(query) > 2_000:
            raise ValueError("Tool discovery query must not exceed 2000 characters")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(
            self,
            "arguments",
            _json_mapping(self.arguments, field_name="discovery arguments"),
        )
        object.__setattr__(self, "tool_refs", tool_refs)
        object.__setattr__(self, "query", query or None)
        object.__setattr__(self, "provider_reference_ids", provider_refs)


@dataclass(frozen=True)
class ToolTransportCheckpoint:
    """Preserve opaque provider replay state for one semantic turn."""

    checkpoint_id: str
    provider: str
    model: str
    contract_version: str
    turn_id: str
    discovery_event_id: str
    integrity_digest: str
    expires_at: str
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
                    provider="openai",
                    model="example-model",
                    contract_version="responses/v1",
                    turn_id="turn_1",
                    discovery_event_id="search_1",
                    integrity_digest="0" * 64,
                    expires_at="end_of_turn",
                    opaque_payload={"output": []},
                )
                assert checkpoint.opaque_payload == {"output": []}
                ```

            Preserve a durable reference:
                ```python
                checkpoint = ToolTransportCheckpoint(
                    checkpoint_id="checkpoint_2",
                    provider="anthropic",
                    model="example-model",
                    contract_version="messages/v1",
                    turn_id="turn_1",
                    discovery_event_id="search_2",
                    integrity_digest="f" * 64,
                    expires_at="end_of_turn",
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
            ("discovery_event_id", "checkpoint discovery event id"),
        ):
            object.__setattr__(
                self,
                attribute,
                _reference(getattr(self, attribute), field_name=label),
            )
        if _DIGEST_PATTERN.fullmatch(str(self.integrity_digest or "")) is None:
            raise ValueError("Checkpoint integrity_digest must be lowercase SHA-256")
        expires_at = _bounded_text(
            self.expires_at,
            field_name="checkpoint expiration",
            maximum=100,
        )
        durable_ref = None
        if self.durable_ref is not None:
            durable_ref = _reference(self.durable_ref, field_name="checkpoint durable ref")
        payload = None
        if self.opaque_payload is not None:
            payload = _json_mapping(
                self.opaque_payload,
                field_name="checkpoint opaque payload",
            )
        if payload is None and durable_ref is None:
            raise ValueError("Checkpoint requires opaque_payload or durable_ref")
        object.__setattr__(self, "integrity_digest", str(self.integrity_digest))
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "opaque_payload", payload)
        object.__setattr__(self, "durable_ref", durable_ref)


@dataclass(frozen=True)
class ToolDiscoveryCapabilities:
    """Declare discovery support for one exact provider/model binding."""

    provider: str
    model: str
    endpoint_family: str
    supported_modes: tuple[ToolDiscoveryMode, ...]
    replay_requirement: ToolReplayRequirement = "none"
    max_results: int = 5
    protocol_version: str = ""

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
                    supported_modes=("engine_projected",),
                )
                assert capabilities.supported_modes == ("engine_projected",)
                ```

            Declare replay requirements:
                ```python
                capabilities = ToolDiscoveryCapabilities(
                    provider="openai",
                    model="example-model",
                    endpoint_family="responses",
                    supported_modes=("native_client",),
                    replay_requirement="previous_response",
                )
                assert capabilities.replay_requirement == "previous_response"
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
        allowed_modes = {"native_hosted", "native_client", "engine_projected"}
        if not modes or any(mode not in allowed_modes for mode in modes):
            raise ValueError("Capability supported_modes must contain known modes")
        if len(modes) != len(set(modes)):
            raise ValueError("Capability supported_modes must be unique")
        if self.replay_requirement not in {"none", "previous_response", "full_history"}:
            raise ValueError("Capability replay_requirement is unsupported")
        if not 1 <= int(self.max_results) <= 50:
            raise ValueError("Capability max_results must be between 1 and 50")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "endpoint_family", endpoint)
        object.__setattr__(self, "supported_modes", modes)
        object.__setattr__(self, "max_results", int(self.max_results))
        object.__setattr__(self, "protocol_version", str(self.protocol_version or ""))

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
                    supported_modes=("engine_projected",),
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
                    supported_modes=("engine_projected",),
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
        return request.mode in self.supported_modes and request.max_results <= self.max_results


__all__ = [
    "JSONScalar",
    "JSONValue",
    "ToolDiscoveryCapabilities",
    "ToolDiscoveryError",
    "ToolDiscoveryEvent",
    "ToolDiscoveryMode",
    "ToolDiscoveryRequest",
    "ToolDiscoverySource",
    "ToolDiscoveryStatus",
    "ToolExposure",
    "ToolNamespace",
    "ToolReplayRequirement",
    "ToolTransportCheckpoint",
]

"""Provider-neutral native Tool-calling contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Literal

from .types import LLMError

ToolChoice = Literal["auto", "required", "none"]


class LLMToolCallError(LLMError):
    """Base class for native Tool-call transport failures."""


class LLMToolCallCapabilityError(LLMToolCallError):
    """Report that the selected provider cannot satisfy native Tool calling."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        feature: str = "native_tool_calling",
    ) -> None:
        """
        Initialize an unsupported native Tool-call capability failure.

        The error identifies the selected provider and model without switching
        providers or falling back to assistant-authored JSON.

        Examples:
            Report an unsupported local provider:
                ```python
                error = LLMToolCallCapabilityError(
                    provider="ollama",
                    model="local-model",
                )
                assert error.code == "tool_calling_unsupported"
                ```

            Preserve an unspecified model:
                ```python
                error = LLMToolCallCapabilityError(
                    provider="custom",
                    model=None,
                )
                assert error.model is None
                ```

        Args:
            provider: Configured provider name.
            model: Configured model or deployment identifier.
            feature: Exact native Tool capability the request requires.

        Returns:
            None: Initializes the exception.

        Notes:
            Capability failures are terminal for the selected configuration.
        """

        normalized_provider = str(provider or "").strip()
        if not normalized_provider:
            raise ValueError("Tool-call provider must not be empty")
        normalized_feature = str(feature or "native_tool_calling").strip()
        super().__init__(
            f"Provider '{normalized_provider}' / model '{model or '?'}' does not "
            f"support required capability '{normalized_feature}'."
        )
        self.code = "tool_calling_unsupported"
        self.provider = normalized_provider
        self.model = model
        self.feature = normalized_feature


class LLMToolCallProviderRequestError(LLMToolCallError):
    """Report provider rejection of one prepared native Tool-call request."""

    def __init__(
        self,
        *,
        provider: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        """
        Initialize a provider Tool-request rejection.

        The exception separates non-repairable request rejection from malformed
        model output so the Engine does not ask the model to repair a request it
        never received successfully.

        Examples:
            Preserve an HTTP status:
                ```python
                error = LLMToolCallProviderRequestError(
                    provider="openai",
                    message="Invalid request.",
                    status_code=400,
                )
                assert error.status_code == 400
                ```

            Preserve a provider without an HTTP status:
                ```python
                error = LLMToolCallProviderRequestError(
                    provider="anthropic",
                    message="Request rejected.",
                )
                assert error.code == "tool_request_rejected"
                ```

        Args:
            provider: Stable configured provider name.
            message: Bounded provider rejection diagnostic.
            status_code: HTTP response status when available.

        Returns:
            None: Initializes the exception.

        Notes:
            Provider request errors are terminal for the current Engine run and
            are never classified as model-output repair candidates.
        """

        normalized_provider = str(provider or "").strip()
        normalized_message = str(message or "").strip()
        if not normalized_provider:
            raise ValueError("Tool-call provider must not be empty")
        if not normalized_message:
            raise ValueError("Tool-call provider error message must not be empty")
        super().__init__(normalized_message)
        self.code = "tool_request_rejected"
        self.provider = normalized_provider
        self.status_code = status_code


class LLMToolCallResponseError(LLMToolCallError):
    """Describe one malformed or incomplete native Tool-call response."""

    def __init__(self, *, code: str, message: str) -> None:
        """
        Initialize a typed native Tool-call response failure.

        The exception preserves a stable machine-readable code while retaining
        the provider-neutral diagnostic intended for Engine and Studio surfaces.

        Examples:
            Report invalid arguments:
                ```python
                error = LLMToolCallResponseError(
                    code="invalid_arguments",
                    message="Arguments must decode to an object.",
                )
                assert error.code == "invalid_arguments"
                ```

            Report a truncated response:
                ```python
                error = LLMToolCallResponseError(
                    code="truncated",
                    message="Provider stopped before completing Tool calls.",
                )
                assert str(error).startswith("Provider stopped")
                ```

        Args:
            code: Stable provider-neutral failure code.
            message: Human-readable failure diagnostic.

        Returns:
            None: Initializes the exception.

        Notes:
            Provider-specific payloads remain in observability records and are
            not embedded in the public exception.
        """

        normalized_code = str(code or "").strip()
        normalized_message = str(message or "").strip()
        if not normalized_code:
            raise ValueError("Tool-call response error code must not be empty")
        if not normalized_message:
            raise ValueError("Tool-call response error message must not be empty")
        super().__init__(normalized_message)
        self.code = normalized_code


@dataclass(frozen=True)
class ToolDefinition:
    """Declare one provider-neutral Tool available to an LLM."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def __post_init__(self) -> None:
        """
        Validate and detach one Tool definition.

        The definition remains provider-neutral and owns only the Tool name,
        description, and JSON Schema for its arguments.

        Examples:
            Define a Tool:
                ```python
                tool = ToolDefinition(
                    name="lookup",
                    description="Look up one record.",
                    input_schema={"type": "object", "properties": {}},
                )
                assert tool.name == "lookup"
                ```

            Reject an empty name:
                ```python
                try:
                    ToolDefinition("", "Missing name.", {"type": "object"})
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized Tool definition.

        Returns:
            None: Validates and normalizes the frozen value.

        Notes:
            Result schemas are intentionally excluded because the provider
            selects calls; the Engine owns Tool execution and results.
        """

        name = str(self.name or "").strip()
        description = str(self.description or "").strip()
        if not name:
            raise ValueError("Tool definition name must not be empty")
        if not isinstance(self.input_schema, dict):
            raise TypeError("Tool definition input_schema must be an object")
        schema = copy.deepcopy(self.input_schema)
        if schema.get("type") not in {None, "object"}:
            raise ValueError("Tool definition input_schema must describe an object")
        schema.setdefault("type", "object")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "input_schema", schema)


@dataclass(frozen=True)
class ToolCallRequest:
    """Request native provider Tool selection for one model decision."""

    tools: tuple[ToolDefinition, ...]
    choice: ToolChoice = "required"
    max_calls: int = 1

    def __post_init__(self) -> None:
        """
        Validate one native Tool-selection request.

        The request bounds the number of proposed calls without deciding how
        the Engine later schedules accepted calls.

        Examples:
            Require exactly one available Tool selection:
                ```python
                request = ToolCallRequest(
                    tools=(ToolDefinition("finish", "Finish.", {"type": "object"}),),
                    max_calls=1,
                )
                assert request.choice == "required"
                ```

            Permit an ordered multi-call proposal:
                ```python
                request = ToolCallRequest(
                    tools=(ToolDefinition("read", "Read.", {"type": "object"}),),
                    max_calls=4,
                )
                assert request.max_calls == 4
                ```

        Args:
            self: Newly initialized Tool-call request.

        Returns:
            None: Validates and normalizes the frozen value.

        Notes:
            `max_calls` bounds model output cardinality. It does not represent
            Engine concurrency or `max_in_flight`.
        """

        tools = tuple(self.tools)
        if not tools:
            raise ValueError("Tool-call request must contain at least one Tool")
        if not all(isinstance(tool, ToolDefinition) for tool in tools):
            raise TypeError("Tool-call request tools must be ToolDefinition values")
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names):
            raise ValueError("Tool-call request Tool names must be unique")
        if self.choice not in {"auto", "required", "none"}:
            raise ValueError("Tool-call request choice must be auto, required, or none")
        if not 1 <= int(self.max_calls) <= 4:
            raise ValueError("Tool-call request max_calls must be between 1 and 4")
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "max_calls", int(self.max_calls))


def tool_call_request_fingerprint(request: ToolCallRequest | None) -> str:
    """Return a deterministic fingerprint of the provider-visible Tool contract."""

    if request is None:
        return ""
    payload = {
        "choice": request.choice,
        "max_calls": request.max_calls,
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in request.tools
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolCall:
    """Represent one provider-framed native Tool call."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and detach one native Tool call.

        Provider adapters construct this value only after arguments have been
        decoded into one JSON object.

        Examples:
            Preserve a provider call identifier:
                ```python
                call = ToolCall("call_1", "lookup", {"key": "A"})
                assert call.call_id == "call_1"
                ```

            Reject non-object arguments:
                ```python
                try:
                    ToolCall("call_1", "lookup", [])
                except TypeError:
                    pass
                ```

        Args:
            self: Newly initialized Tool call.

        Returns:
            None: Validates and normalizes the frozen value.

        Notes:
            Provider metadata may retain opaque continuation fields but must not
            contain executable behavior.
        """

        call_id = str(self.call_id or "").strip()
        name = str(self.name or "").strip()
        if not call_id:
            raise ValueError("Tool call call_id must not be empty")
        if not name:
            raise ValueError("Tool call name must not be empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("Tool call arguments must be an object")
        if not isinstance(self.provider_metadata, dict):
            raise TypeError("Tool call provider_metadata must be an object")
        object.__setattr__(self, "call_id", call_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arguments", copy.deepcopy(self.arguments))
        object.__setattr__(
            self,
            "provider_metadata",
            copy.deepcopy(self.provider_metadata),
        )


@dataclass(frozen=True)
class ToolCallResponse:
    """Return provider-framed Tool calls without flattening them into text."""

    calls: tuple[ToolCall, ...]
    text: str = ""
    finish_reason: str = ""
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate and detach one native Tool-call response.

        The response preserves ordered call items and optional assistant text
        while remaining independent from Engine scheduling.

        Examples:
            Return one Tool call:
                ```python
                response = ToolCallResponse(
                    calls=(ToolCall("call_1", "finish", {}),),
                    finish_reason="tool_calls",
                )
                assert response.calls[0].name == "finish"
                ```

            Represent a provider response with no call:
                ```python
                response = ToolCallResponse(calls=(), text="No Tool selected.")
                assert not response.calls
                ```

        Args:
            self: Newly initialized Tool-call response.

        Returns:
            None: Validates and normalizes the frozen value.

        Notes:
            Call-count policy remains Engine-owned; a provider may return more
            calls than the request allowed so the Engine can reject them.
        """

        calls = tuple(self.calls)
        if not all(isinstance(call, ToolCall) for call in calls):
            raise TypeError("Tool-call response calls must be ToolCall values")
        if not isinstance(self.provider_metadata, dict):
            raise TypeError("Tool-call response provider_metadata must be an object")
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "finish_reason", str(self.finish_reason or ""))
        object.__setattr__(
            self,
            "provider_metadata",
            copy.deepcopy(self.provider_metadata),
        )

    def observation_text(self) -> str:
        """
        Serialize the normalized response for observability capture.

        The representation is canonical diagnostic JSON and is never reparsed
        as the model's Tool-call transport.

        Examples:
            Serialize one call:
                ```python
                response = ToolCallResponse(
                    calls=(ToolCall("call_1", "finish", {}),)
                )
                assert '"finish"' in response.observation_text()
                ```

            Serialize an empty response:
                ```python
                response = ToolCallResponse(calls=(), text="No call")
                assert '"calls":[]' in response.observation_text()
                ```

        Args:
            self: Normalized Tool-call response.

        Returns:
            str: Canonical JSON used only for trace and log persistence.

        Notes:
            Provider item boundaries have already been preserved in `calls`.
        """

        return json.dumps(
            {
                "calls": [
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "provider_metadata": call.provider_metadata,
                    }
                    for call in self.calls
                ],
                "text": self.text,
                "finish_reason": self.finish_reason,
                "provider_metadata": self.provider_metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


__all__ = [
    "LLMToolCallCapabilityError",
    "LLMToolCallError",
    "LLMToolCallProviderRequestError",
    "LLMToolCallResponseError",
    "ToolCall",
    "ToolCallRequest",
    "ToolCallResponse",
    "ToolChoice",
    "ToolDefinition",
    "tool_call_request_fingerprint",
]

"""Canonical provider-neutral model generation contracts."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import json
from typing import Any, Literal, TypeAlias

from .tool_calling import ModelToolSpec, ToolCallOutput, ToolChoice
from .tool_discovery import (
    ModelContinuation,
    ToolDiscoveryRequest,
    ToolDiscoveryResult,
)
from .types import ImageInput, PromptCacheRequest, StructuredOutputRequest

MessageRole = Literal["system", "developer", "user", "assistant", "tool"]
ModelResponseFormat: TypeAlias = Literal["text", "json_object", "raw"] | StructuredOutputRequest
MODEL_REQUEST_CONTRACT_VERSION = "model_request/v1"


@dataclass(frozen=True)
class TextPart:
    """Represent one canonical text content part.

    Text remains detached from provider-specific input/output block shapes and
    can be projected by every chat endpoint adapter.

    Examples:
        Build a user text part:
            ```python
            part = TextPart("Hello")
            assert part.text == "Hello"
            ```

        Preserve an intentional empty part:
            ```python
            part = TextPart("")
            assert part.text == ""
            ```

    Args:
        text: Exact caller-authored text.

    Returns:
        TextPart: An immutable canonical content part.

    Notes:
        Provider-specific content block identifiers belong to response items,
        not request text parts.
    """

    text: str

    def __post_init__(self) -> None:
        """Normalize one text value.

        Dynamic callers may supply string-like values; the contract stores the
        resulting text without trimming semantic whitespace.

        Examples:
            Normalize a numeric value:
                ```python
                assert TextPart(3).text == "3"
                ```

            Preserve surrounding whitespace:
                ```python
                assert TextPart(" x ").text == " x "
                ```

        Args:
            self: Newly initialized text part.

        Returns:
            None: The frozen value is normalized in place.

        Notes:
            None.
        """

        object.__setattr__(self, "text", str(self.text))


ImagePart = ImageInput
ContentPart: TypeAlias = TextPart | ImagePart


@dataclass(frozen=True)
class ChatMessage:
    """Represent one canonical model conversation message.

    A message owns ordered content parts and optional Tool-result correlation
    without retaining a provider dictionary or wire block.

    Examples:
        Build a user message:
            ```python
            message = ChatMessage("user", (TextPart("Hello"),))
            assert message.role == "user"
            ```

        Build a correlated Tool-result message:
            ```python
            message = ChatMessage(
                "tool",
                (TextPart("result"),),
                tool_call_id="call_1",
            )
            assert message.tool_call_id == "call_1"
            ```

    Args:
        role: Provider-neutral conversation role.
        content: Ordered canonical content parts.
        name: Optional bounded participant or Tool name.
        tool_call_id: Optional provider Tool-call identity for a result message.

    Returns:
        ChatMessage: An immutable canonical message.

    Notes:
        Tool-result continuation may instead use `ModelRequest.tool_outputs`
        when the selected provider requires an opaque continuation protocol.
    """

    role: MessageRole
    content: tuple[ContentPart, ...]
    name: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize one canonical message.

        Validation preserves content order and rejects provider-specific values
        before request preparation begins.

        Examples:
            Normalize list content to a tuple:
                ```python
                message = ChatMessage("user", [TextPart("Hello")])
                assert isinstance(message.content, tuple)
                ```

            Reject an unsupported role:
                ```python
                try:
                    ChatMessage("provider", (TextPart("Hello"),))
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized message.

        Returns:
            None: The frozen value is validated and normalized in place.

        Notes:
            Empty content is allowed for provider-neutral assistant placeholders.
        """

        if self.role not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError("model message role is unsupported")
        content = tuple(self.content)
        if not all(isinstance(part, (TextPart, ImageInput)) for part in content):
            raise TypeError("model message content must contain TextPart or ImagePart values")
        name = None if self.name is None else str(self.name).strip()
        tool_call_id = None if self.tool_call_id is None else str(self.tool_call_id).strip()
        if self.name is not None and not name:
            raise ValueError("model message name must not be empty")
        if self.tool_call_id is not None and not tool_call_id:
            raise ValueError("model message tool_call_id must not be empty")
        if self.role == "tool" and not tool_call_id:
            raise ValueError("Tool-result messages require tool_call_id")
        if self.role != "tool" and tool_call_id:
            raise ValueError("tool_call_id is valid only for Tool-result messages")
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "tool_call_id", tool_call_id)


@dataclass(frozen=True)
class GenerationOptions:
    """Represent shared provider-neutral generation controls.

    The options contain only controls with canonical cross-provider meaning;
    provider-private fields stay adjacent to the selected endpoint adapter.

    Examples:
        Bound model output:
            ```python
            options = GenerationOptions(max_output_tokens=256)
            assert options.max_output_tokens == 256
            ```

        Request bounded reasoning:
            ```python
            options = GenerationOptions(reasoning_effort="high", reasoning_budget=1024)
            assert options.reasoning_budget == 1024
            ```

    Args:
        temperature: Optional provider-neutral sampling temperature.
        max_output_tokens: Optional output-token ceiling.
        reasoning_effort: Optional normalized reasoning effort label.
        reasoning_budget: Optional provider-neutral reasoning-token budget.
        reasoning_summary: Optional normalized summary mode.

    Returns:
        GenerationOptions: Immutable shared generation controls.

    Notes:
        Adapter-specific option types may be added beside an adapter and are not
        accepted through this shared contract.
    """

    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_budget: int | None = None
    reasoning_summary: str | None = None

    def __post_init__(self) -> None:
        """Validate shared generation controls.

        Numeric limits are normalized before estimation, quota reservation, or
        provider projection.

        Examples:
            Accept a zero temperature:
                ```python
                assert GenerationOptions(temperature=0).temperature == 0.0
                ```

            Reject an invalid output ceiling:
                ```python
                try:
                    GenerationOptions(max_output_tokens=0)
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized generation controls.

        Returns:
            None: The frozen value is validated and normalized in place.

        Notes:
            Capability support is resolved separately for the selected binding.
        """

        if self.temperature is not None:
            if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
                raise TypeError("generation temperature must be numeric or None")
            if not 0 <= float(self.temperature) <= 2:
                raise ValueError("generation temperature must be between 0 and 2")
            object.__setattr__(self, "temperature", float(self.temperature))
        for name in ("max_output_tokens", "reasoning_budget"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"generation {name} must be an integer or None")
            if value <= 0:
                raise ValueError(f"generation {name} must be positive")
        for name in ("reasoning_effort", "reasoning_summary"):
            value = getattr(self, name)
            if value is not None:
                normalized = str(value).strip()
                if not normalized:
                    raise ValueError(f"generation {name} must not be empty")
                object.__setattr__(self, name, normalized)


@dataclass(frozen=True)
class ModelRequest:
    """Represent one immutable provider-neutral model generation request.

    The request carries the complete logical decision state required for direct
    completion, native Tool calls, and provider-native discovery continuation.

    Examples:
        Build a direct-completion request:
            ```python
            request = ModelRequest(
                messages=(ChatMessage("user", (TextPart("Hello"),)),)
            )
            assert request.tool_choice == "none"
            ```

        Build a native Tool request:
            ```python
            request = ModelRequest(
                messages=(ChatMessage("user", (TextPart("Find A"),)),),
                tools=(tool,),
                tool_choice="auto",
                max_tool_calls=2,
            )
            assert request.max_tool_calls == 2
            ```

    Args:
        messages: Ordered canonical conversation messages.
        tools: Model-visible Tool specifications.
        tool_choice: Provider-neutral Tool-selection policy.
        max_tool_calls: Maximum ordered Tool calls in one response.
        native_tool_search: Optional provider-native hosted/client search request.
        active_tool_names: Exact currently active Tool names for native discovery.
        turn_id: Optional semantic turn identity used to bind continuation state.
        tool_outputs: Completed Tool outputs submitted through a continuation.
        response_format: Text, JSON object, raw, or canonical structured schema.
        generation: Shared provider-neutral generation controls.
        prompt_cache: Optional stable-prefix cache request.
        continuation: Optional opaque provider replay state.
        call_name: Optional stable logical invocation identity for observations.

    Returns:
        ModelRequest: Immutable canonical request state.

    Notes:
        `engine_projected` discovery is intentionally invalid here. Callers
        express projected search/load controls as ordinary `tools`.
    """

    messages: tuple[ChatMessage, ...]
    tools: tuple[ModelToolSpec, ...] = ()
    tool_choice: ToolChoice = "none"
    max_tool_calls: int = 1
    native_tool_search: ToolDiscoveryRequest | None = None
    active_tool_names: tuple[str, ...] = ()
    turn_id: str | None = None
    tool_outputs: tuple[ToolCallOutput, ...] = ()
    discovery_result: ToolDiscoveryResult | None = None
    response_format: ModelResponseFormat = "text"
    generation: GenerationOptions = field(default_factory=GenerationOptions)
    prompt_cache: PromptCacheRequest | None = None
    continuation: ModelContinuation | None = None
    call_name: str | None = None
    trace_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach one canonical generation request.

        Validation protects ordering, continuation identity, and Tool-surface
        invariants before capability resolution or any physical provider attempt.

        Examples:
            Normalize sequence inputs:
                ```python
                request = ModelRequest(
                    messages=[ChatMessage("user", [TextPart("Hello")])]
                )
                assert isinstance(request.messages, tuple)
                ```

            Reject Engine-projected discovery as a native request:
                ```python
                try:
                    ModelRequest(
                        messages=(),
                        native_tool_search=ToolDiscoveryRequest("engine_projected"),
                    )
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized generation request.

        Returns:
            None: The frozen request is validated and detached in place.

        Notes:
            Whole-request model/adapter compatibility is validated after binding
            resolution; this value performs provider-independent validation only.
        """

        messages = tuple(self.messages)
        tools = tuple(self.tools)
        active_tool_names = tuple(str(name or "").strip() for name in self.active_tool_names)
        tool_outputs = tuple(self.tool_outputs)
        if not all(isinstance(message, ChatMessage) for message in messages):
            raise TypeError("model request messages must be ChatMessage values")
        if not all(isinstance(tool, ModelToolSpec) for tool in tools):
            raise TypeError("model request tools must be ModelToolSpec values")
        tool_names = tuple(tool.name for tool in tools)
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("model request Tool names must be unique")
        if self.tool_choice not in {"none", "auto", "required"}:
            raise ValueError("model request tool_choice must be none, auto, or required")
        if not tools and self.tool_choice != "none":
            raise ValueError("model request without Tools requires tool_choice='none'")
        if isinstance(self.max_tool_calls, bool) or not isinstance(self.max_tool_calls, int):
            raise TypeError("model request max_tool_calls must be an integer")
        if not 1 <= self.max_tool_calls <= 4:
            raise ValueError("model request max_tool_calls must be between 1 and 4")
        if self.native_tool_search is not None:
            if not isinstance(self.native_tool_search, ToolDiscoveryRequest):
                raise TypeError("native_tool_search must be ToolDiscoveryRequest or None")
            if self.native_tool_search.mode not in {"native_hosted", "native_client"}:
                raise ValueError("native_tool_search supports only native_hosted or native_client")
            if not tools:
                raise ValueError("native Tool search requires a deferred Tool catalog")
        if any(not name for name in active_tool_names):
            raise ValueError("active Tool names must not be empty")
        if len(active_tool_names) != len(set(active_tool_names)):
            raise ValueError("active Tool names must be unique")
        if set(active_tool_names) - set(tool_names):
            raise ValueError("active Tool names must exist in the request Tool catalog")
        turn_id = None if self.turn_id is None else str(self.turn_id).strip()
        if self.turn_id is not None and not turn_id:
            raise ValueError("model request turn_id must not be empty")
        if self.native_tool_search is not None and not turn_id:
            raise ValueError("native Tool search requires turn_id")
        call_name = None if self.call_name is None else str(self.call_name).strip()
        if self.call_name is not None and not call_name:
            raise ValueError("model request call_name must not be empty")
        if not isinstance(self.trace_context, dict):
            raise TypeError("model request trace_context must be an object")
        try:
            encoded_trace_context = json.dumps(
                self.trace_context,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "model request trace_context must contain JSON-compatible values"
            ) from exc
        if len(encoded_trace_context) > 64 * 1024:
            raise ValueError("model request trace_context must not exceed 65536 bytes")
        if not all(isinstance(output, ToolCallOutput) for output in tool_outputs):
            raise TypeError("model request tool_outputs must be ToolCallOutput values")
        call_ids = tuple(output.call_id for output in tool_outputs)
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("model request Tool outputs must have unique call ids")
        if self.continuation is not None:
            if not isinstance(self.continuation, ModelContinuation):
                raise TypeError("model request continuation must be ModelContinuation or None")
            if not turn_id:
                raise ValueError("model continuation requires turn_id")
            if self.continuation.turn_id != turn_id:
                raise ValueError("model continuation turn_id must match the request")
        if tool_outputs and self.continuation is None:
            raise ValueError("model Tool outputs require a continuation")
        if self.discovery_result is not None:
            if not isinstance(self.discovery_result, ToolDiscoveryResult):
                raise TypeError(
                    "model request discovery_result must be ToolDiscoveryResult or None"
                )
            if self.continuation is None:
                raise ValueError("model discovery result requires a continuation")
            if tool_outputs:
                raise ValueError(
                    "model discovery result cannot accompany ordinary Tool outputs"
                )
            if (
                self.discovery_result.status == "completed"
                and not set(self.discovery_result.tool_names).issubset(
                    set(active_tool_names)
                )
            ):
                raise ValueError(
                    "completed model discovery result Tools must be active"
                )
        if not isinstance(self.generation, GenerationOptions):
            raise TypeError("model request generation must be GenerationOptions")
        if self.prompt_cache is not None and not isinstance(self.prompt_cache, PromptCacheRequest):
            raise TypeError("model request prompt_cache must be PromptCacheRequest or None")
        if not (
            isinstance(self.response_format, StructuredOutputRequest)
            or self.response_format in ("text", "json_object", "raw")
        ):
            raise TypeError("model request response_format is unsupported")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "active_tool_names", active_tool_names)
        object.__setattr__(self, "turn_id", turn_id)
        object.__setattr__(self, "tool_outputs", tool_outputs)
        object.__setattr__(self, "call_name", call_name)
        object.__setattr__(self, "trace_context", copy.deepcopy(self.trace_context))


def message_from_text(role: MessageRole, text: str) -> ChatMessage:
    """Create one canonical single-text message.

    This convenience constructor avoids provider dictionaries in new callers
    while preserving exact text and the requested canonical role.

    Examples:
        Create a user message:
            ```python
            message = message_from_text("user", "Hello")
            assert message.content[0].text == "Hello"
            ```

        Create a system message:
            ```python
            message = message_from_text("system", "Be concise")
            assert message.role == "system"
            ```

    Args:
        role: Canonical conversation role.
        text: Exact message text.

    Returns:
        ChatMessage: A message containing one `TextPart`.

    Notes:
        Tool-result messages require correlation and must use `ChatMessage`
        directly.
    """

    return ChatMessage(role=role, content=(TextPart(text),))


__all__ = [
    "ChatMessage",
    "ContentPart",
    "GenerationOptions",
    "ImagePart",
    "MessageRole",
    "MODEL_REQUEST_CONTRACT_VERSION",
    "ModelRequest",
    "ModelResponseFormat",
    "TextPart",
    "message_from_text",
]

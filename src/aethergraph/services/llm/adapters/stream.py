"""Exact streaming Chat endpoint-adapter dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import copy
from dataclasses import dataclass, field
from typing import Any

from aethergraph.services.llm.adapters.anthropic import AnthropicMessagesAdapter
from aethergraph.services.llm.adapters.azure import AzureChatAdapter
from aethergraph.services.llm.adapters.gemini import GeminiGenerateContentAdapter
from aethergraph.services.llm.adapters.openai_compatible import OpenAICompatibleChatAdapter
from aethergraph.services.llm.adapters.openai_responses import OpenAIResponsesAdapter
from aethergraph.services.llm.provider_transport import ProviderCallResult
from aethergraph.services.llm.types import ChatOutputFormat, LLMUnsupportedFeatureError

TextCallback = Callable[[str], Awaitable[None]]
UsageCallback = Callable[[dict[str, int]], Awaitable[None]]
StreamResult = ProviderCallResult[tuple[str, dict[str, int]]]
StreamHandler = Callable[[Any, "ChatStreamInvocation"], Awaitable[StreamResult]]


@dataclass(frozen=True)
class ChatStreamInvocation:
    """Carry one prepared single-attempt streaming Chat invocation.

    Intro:
        Freezes shared lifecycle state before exact endpoint dispatch and keeps
        adapter-private options detached from caller-owned dictionaries.

    Examples:
        Build a basic invocation:
            ```python
            invocation = ChatStreamInvocation(
                messages=({"role": "user", "content": "Hello"},),
                model="model-a",
                reasoning_effort=None,
                reasoning_summary=None,
                thinking_budget=None,
                thinking_mode=None,
                max_output_tokens=128,
                output_format="text",
                json_schema=None,
                schema_name="Response",
                strict_schema=True,
                fail_on_unsupported=True,
            )
            ```

        Attach live callbacks:
            ```python
            invocation = ChatStreamInvocation(
                messages=tuple(messages),
                model="model-a",
                reasoning_effort="medium",
                reasoning_summary="auto",
                thinking_budget=None,
                thinking_mode="on",
                max_output_tokens=256,
                output_format="text",
                json_schema=None,
                schema_name="Response",
                strict_schema=True,
                fail_on_unsupported=True,
                on_delta=on_delta,
                on_usage_update=on_usage_update,
            )
            ```

    Args:
        messages: Detached provider-projected stable conversation messages.
        model: Exact configured model or deployment identity.
        reasoning_effort: Optional normalized reasoning-depth override.
        reasoning_summary: Optional displayable reasoning-summary mode.
        thinking_budget: Optional reasoning-token budget.
        thinking_mode: Optional provider thinking on/off mode.
        max_output_tokens: Optional maximum generated tokens.
        output_format: Prepared text output mode.
        json_schema: Optional prepared JSON schema.
        schema_name: Stable structured-output schema name.
        strict_schema: Whether native schema enforcement is strict.
        fail_on_unsupported: Whether unsupported native fields must fail.
        on_delta: Optional async assistant-text callback.
        on_thinking_delta: Optional async reasoning-summary callback.
        on_usage_update: Optional async cumulative usage callback.
        options: Detached bounded adapter-private options.

    Returns:
        ChatStreamInvocation: Immutable prepared adapter-call state.

    Notes:
        Retry, rate gating, accounting, metering, and observations remain outside
        this value and execute once around adapter dispatch.
    """

    messages: tuple[dict[str, Any], ...]
    model: str
    reasoning_effort: str | None
    reasoning_summary: str | None
    thinking_budget: int | None
    thinking_mode: str | None
    max_output_tokens: int | None
    output_format: ChatOutputFormat
    json_schema: dict[str, Any] | None
    schema_name: str
    strict_schema: bool
    fail_on_unsupported: bool
    on_delta: TextCallback | None = None
    on_thinking_delta: TextCallback | None = None
    on_usage_update: UsageCallback | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach streaming invocation state.

        Intro:
            Rejects missing model identities and deep-copies mutable request data
            before the value reaches a physical adapter.

        Examples:
            Validate normal construction:
                ```python
                invocation = ChatStreamInvocation(...)
                assert invocation.model
                ```

            Reject an empty model:
                ```python
                ChatStreamInvocation(..., model="")
                ```

        Args:
            self: Newly initialized stream invocation.

        Returns:
            None: Completes after normalized values are stored.

        Notes:
            Callback identities are preserved; only mutable payload data is copied.
        """

        model = str(self.model or "").strip()
        if not model:
            raise ValueError("stream adapter invocation requires a model")
        object.__setattr__(self, "model", model)
        object.__setattr__(
            self,
            "messages",
            tuple(copy.deepcopy(message) for message in self.messages),
        )
        object.__setattr__(self, "json_schema", copy.deepcopy(self.json_schema))
        object.__setattr__(self, "options", copy.deepcopy(self.options))

    def message_list(self) -> list[dict[str, Any]]:
        """Return detached mutable messages for one adapter attempt.

        Intro:
            Gives each physical attempt an isolated request list while retaining
            immutable lifecycle preparation.

        Examples:
            Read messages:
                ```python
                messages = invocation.message_list()
                ```

            Mutate without changing the invocation:
                ```python
                invocation.message_list().append({"role": "user", "content": "x"})
                ```

        Args:
            self: Prepared stream invocation.

        Returns:
            list[dict[str, Any]]: Deep-copied provider messages.

        Notes:
            Retry attempts cannot share adapter mutations through this projection.
        """

        return [copy.deepcopy(message) for message in self.messages]

    def option_dict(self) -> dict[str, Any]:
        """Return detached adapter-private options.

        Intro:
            Supplies one mutable option mapping per physical attempt.

        Examples:
            Read options:
                ```python
                options = invocation.option_dict()
                ```

            Consume an option locally:
                ```python
                mode = invocation.option_dict().pop("thinking_mode", None)
                ```

        Args:
            self: Prepared stream invocation.

        Returns:
            dict[str, Any]: Deep-copied adapter-private options.

        Notes:
            Handler option consumption never mutates lifecycle observation state.
        """

        return copy.deepcopy(self.options)


async def _stream_openai_responses(host: Any, call: ChatStreamInvocation) -> StreamResult:
    options = call.option_dict()
    return await OpenAIResponsesAdapter.stream(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        reasoning_summary=call.reasoning_summary,
        max_output_tokens=call.max_output_tokens,
        output_format=call.output_format,
        json_schema=call.json_schema,
        schema_name=call.schema_name,
        strict_schema=call.strict_schema,
        fail_on_unsupported=call.fail_on_unsupported,
        on_delta=call.on_delta,
        on_thinking_delta=call.on_thinking_delta,
        on_usage_update=call.on_usage_update,
        **options,
    )


async def _stream_openai_compatible(host: Any, call: ChatStreamInvocation) -> StreamResult:
    return await OpenAICompatibleChatAdapter.stream(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        max_output_tokens=call.max_output_tokens,
        on_delta=call.on_delta,
        on_usage_update=call.on_usage_update,
        **call.option_dict(),
    )


async def _stream_azure_chat_completions(
    host: Any,
    call: ChatStreamInvocation,
) -> StreamResult:
    return await AzureChatAdapter.stream_chat_completions(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        max_output_tokens=call.max_output_tokens,
        on_delta=call.on_delta,
        on_usage_update=call.on_usage_update,
        **call.option_dict(),
    )


async def _stream_anthropic_messages(host: Any, call: ChatStreamInvocation) -> StreamResult:
    options = call.option_dict()
    options["reasoning_effort"] = call.reasoning_effort
    return await AnthropicMessagesAdapter.stream(
        host,
        call.message_list(),
        model=call.model,
        thinking_budget=call.thinking_budget,
        max_output_tokens=call.max_output_tokens,
        output_format=call.output_format,
        json_schema=call.json_schema,
        fail_on_unsupported=call.fail_on_unsupported,
        on_delta=call.on_delta,
        on_thinking_delta=call.on_thinking_delta,
        on_usage_update=call.on_usage_update,
        **options,
    )


async def _stream_gemini_generate_content(
    host: Any,
    call: ChatStreamInvocation,
) -> StreamResult:
    options = call.option_dict()
    options.pop("thinking_mode", None)
    return await GeminiGenerateContentAdapter.stream(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        reasoning_summary=call.reasoning_summary,
        thinking_mode=call.thinking_mode,
        max_output_tokens=call.max_output_tokens,
        on_delta=call.on_delta,
        on_thinking_delta=call.on_thinking_delta,
        on_usage_update=call.on_usage_update,
        **options,
    )


_STREAM_HANDLERS: dict[str, StreamHandler] = {
    "openai_responses": _stream_openai_responses,
    "openai_chat_completions": _stream_openai_compatible,
    "azure_chat_completions": _stream_azure_chat_completions,
    "anthropic_messages": _stream_anthropic_messages,
    "gemini_generate_content": _stream_gemini_generate_content,
}


def registered_chat_stream_adapter_ids() -> frozenset[str]:
    """Return exact endpoint IDs with physical streaming handlers.

    Intro:
        Exposes immutable runtime coverage for registry-conformance validation
        without exposing mutable handler state.

    Examples:
        Check OpenAI Responses coverage:
            ```python
            assert "openai_responses" in registered_chat_stream_adapter_ids()
            ```

        Check unsupported Azure Responses coverage:
            ```python
            assert "azure_responses" not in registered_chat_stream_adapter_ids()
            ```

    Args:
        This function accepts no arguments.

    Returns:
        frozenset[str]: Exact registered physical streaming adapter identities.

    Notes:
        Capability descriptors remain owned by the canonical registry; this
        projection reports executable runtime truth for conformance tests.
    """

    return frozenset(_STREAM_HANDLERS)


async def invoke_chat_stream_adapter(
    host: Any,
    *,
    adapter_id: str,
    invocation: ChatStreamInvocation,
) -> StreamResult:
    """Invoke one exact registered streaming Chat adapter.

    Intro:
        Resolves physical streaming behavior solely by endpoint adapter identity
        and fails closed when no implementation is registered.

    Examples:
        Invoke OpenAI Responses streaming:
            ```python
            result = await invoke_chat_stream_adapter(
                client,
                adapter_id="openai_responses",
                invocation=invocation,
            )
            ```

        Invoke Gemini streaming:
            ```python
            result = await invoke_chat_stream_adapter(
                client,
                adapter_id="gemini_generate_content",
                invocation=invocation,
            )
            ```

    Args:
        host: Bound generic client owning shared transport primitives.
        adapter_id: Exact selected endpoint-adapter identity.
        invocation: Frozen prepared streaming invocation.

    Returns:
        StreamResult: Accumulated text, usage, and transport metadata.

    Notes:
        The registry contains no provider-name selection, retry, accounting, or
        fallback behavior. Missing handlers fail before provider transport.
    """

    handler = _STREAM_HANDLERS.get(str(adapter_id or "").strip())
    if handler is None:
        raise LLMUnsupportedFeatureError(
            host.provider,
            invocation.model,
            "streaming",
            f"endpoint adapter {adapter_id!r} has no native streaming adapter",
        )
    return await handler(host, invocation)

"""Exact non-streaming Chat endpoint-adapter dispatch."""

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
from aethergraph.services.llm.tool_calling import ToolCallRequest, ToolCallResponse
from aethergraph.services.llm.types import ChatOutputFormat, LLMUnsupportedFeatureError

AdapterResult = ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]
AdapterHandler = Callable[[Any, "ChatAdapterInvocation"], Awaitable[AdapterResult]]


@dataclass(frozen=True)
class ChatAdapterInvocation:
    """Carry one prepared single-attempt Chat adapter invocation.

    Intro:
        The value freezes shared lifecycle preparation before exact endpoint
        dispatch and keeps provider-private options in one detached mapping.

    Examples:
        Build a direct text invocation:
            ```python
            invocation = ChatAdapterInvocation(
                messages=({"role": "user", "content": "Hello"},),
                model="model-a",
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

        Attach native Tool state:
            ```python
            invocation = ChatAdapterInvocation(
                messages=tuple(messages),
                model="model-a",
                reasoning_effort="medium",
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
        messages: Prepared stable conversation messages.
        model: Exact configured model or deployment identity.
        reasoning_effort: Optional normalized reasoning-depth override.
        max_output_tokens: Optional maximum generated tokens.
        output_format: Prepared text, JSON, schema, or raw output mode.
        json_schema: Optional prepared provider JSON schema.
        schema_name: Stable provider schema name.
        strict_schema: Whether native schema enforcement is strict.
        validate_json: Whether shared postprocessing validates JSON locally.
        fail_on_unsupported: Whether unsupported native fields must fail.
        structured_output_fields: Optional prepared native structured fields.
        prompt_cache_fields: Optional prepared native cache fields.
        prompt_cache_stable_message_count: Optional stable prefix length.
        tool_request: Optional canonical native Tool request.
        options: Additional bounded adapter-private options.

    Returns:
        ChatAdapterInvocation: Immutable prepared adapter call state.

    Notes:
        Retry, rate gating, quota, metering, and observations remain outside this
        value in the shared invocation lifecycle.
    """

    messages: tuple[dict[str, Any], ...]
    model: str
    reasoning_effort: str | None
    max_output_tokens: int | None
    output_format: ChatOutputFormat
    json_schema: dict[str, Any] | None
    schema_name: str
    strict_schema: bool
    validate_json: bool
    fail_on_unsupported: bool
    structured_output_fields: dict[str, Any] | None = None
    prompt_cache_fields: dict[str, Any] | None = None
    prompt_cache_stable_message_count: int | None = None
    tool_request: ToolCallRequest | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach prepared adapter state.

        Intro:
            Prevents an adapter from mutating lifecycle-owned messages, schemas,
            structured fields, cache fields, or option mappings.

        Examples:
            Observe detached options:
                ```python
                options = {"temperature": 0.2}
                invocation = make_invocation(options=options)
                options["temperature"] = 1.0
                assert invocation.options["temperature"] == 0.2
                ```

            Reject a blank model:
                ```python
                try:
                    make_invocation(model="")
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized adapter invocation.

        Returns:
            None: Validates and detaches the frozen value in place.

        Notes:
            Tool continuation is already immutable and is retained by identity.
        """

        model = str(self.model or "").strip()
        if not model:
            raise ValueError("Chat adapter invocation model must not be empty")
        messages = tuple(copy.deepcopy(message) for message in self.messages)
        if not all(isinstance(message, dict) for message in messages):
            raise TypeError("Chat adapter invocation messages must be objects")
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "json_schema", copy.deepcopy(self.json_schema))
        object.__setattr__(
            self,
            "structured_output_fields",
            copy.deepcopy(self.structured_output_fields),
        )
        object.__setattr__(self, "prompt_cache_fields", copy.deepcopy(self.prompt_cache_fields))
        object.__setattr__(self, "options", copy.deepcopy(self.options))

    def message_list(self) -> list[dict[str, Any]]:
        """Return detached messages for one physical adapter attempt.

        Intro:
            Each retry attempt receives a fresh message collection so an adapter
            cannot modify the frozen logical invocation.

        Examples:
            Read messages:
                ```python
                messages = invocation.message_list()
                ```

            Modify a detached copy:
                ```python
                messages = invocation.message_list()
                messages[0]["content"] = "changed"
                assert invocation.messages[0]["content"] != "changed"
                ```

        Args:
            self: Prepared adapter invocation.

        Returns:
            list[dict[str, Any]]: Fresh detached stable messages.

        Notes:
            Provider wire projection remains owned by the exact adapter method.
        """

        return [copy.deepcopy(message) for message in self.messages]

    def option_dict(self) -> dict[str, Any]:
        """Return detached provider-private options for one attempt.

        Intro:
            Adapter handlers may consume keys without mutating lifecycle-owned
            request or observation state.

        Examples:
            Read sampling options:
                ```python
                options = invocation.option_dict()
                ```

            Consume an adapter key:
                ```python
                thinking_mode = invocation.option_dict().pop("thinking_mode", None)
                ```

        Args:
            self: Prepared adapter invocation.

        Returns:
            dict[str, Any]: Fresh detached adapter-private options.

        Notes:
            Unsupported-field behavior remains explicit in each handler.
        """

        return copy.deepcopy(self.options)


async def _invoke_openai_responses(host: Any, call: ChatAdapterInvocation) -> AdapterResult:
    """Invoke the exact OpenAI Responses adapter.

    Intro:
        Projects prepared structured output, cache state, legacy Tool fields, and
        canonical native Tool state into one single-attempt provider method.

    Examples:
        Invoke direct text:
            ```python
            result = await _invoke_openai_responses(client, invocation)
            ```

        Invoke native Tools:
            ```python
            result = await _invoke_openai_responses(client, tool_invocation)
            ```

    Args:
        host: Bound generic client owning the OpenAI single-attempt method.
        call: Frozen prepared adapter invocation.

    Returns:
        AdapterResult: Raw single-attempt OpenAI result and transport metadata.

    Notes:
        Shared lifecycle behavior is intentionally absent from this handler.
    """

    options = call.option_dict()
    tools = options.pop("tools", None)
    tool_choice = options.pop("tool_choice", None)
    return await OpenAIResponsesAdapter.invoke(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        max_output_tokens=call.max_output_tokens,
        output_format=call.output_format,
        json_schema=call.json_schema,
        schema_name=call.schema_name,
        strict_schema=call.strict_schema,
        tools=tools,
        tool_choice=tool_choice,
        tool_request=call.tool_request,
        structured_output_fields=call.structured_output_fields,
        prompt_cache_fields=call.prompt_cache_fields,
        prompt_cache_stable_message_count=call.prompt_cache_stable_message_count,
        **options,
    )


async def _invoke_chat_completions(host: Any, call: ChatAdapterInvocation) -> AdapterResult:
    """Invoke the shared OpenAI-compatible Chat Completions adapter.

    Intro:
        Serves OpenAI and compatible provider bindings through one exact protocol
        implementation while the selected model capability remains external.

    Examples:
        Invoke OpenAI Chat Completions:
            ```python
            result = await _invoke_chat_completions(client, invocation)
            ```

        Invoke a compatible local endpoint:
            ```python
            result = await _invoke_chat_completions(local_client, invocation)
            ```

    Args:
        host: Bound generic client owning compatible single-attempt methods.
        call: Frozen prepared adapter invocation.

    Returns:
        AdapterResult: Raw compatible result and transport metadata.

    Notes:
        Provider identity is used only inside the adjacent compatible method for
        documented wire options such as DeepSeek reasoning controls.
    """

    options = call.option_dict()
    tools = options.pop("tools", None)
    tool_choice = options.pop("tool_choice", None)
    return await OpenAICompatibleChatAdapter.invoke(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        max_output_tokens=call.max_output_tokens,
        output_format=call.output_format,
        json_schema=call.json_schema,
        fail_on_unsupported=call.fail_on_unsupported,
        tools=tools,
        tool_choice=tool_choice,
        tool_request=call.tool_request,
        schema_name=call.schema_name,
        strict_schema=call.strict_schema,
        structured_output_fields=call.structured_output_fields,
        **options,
    )


async def _invoke_azure_responses(host: Any, call: ChatAdapterInvocation) -> AdapterResult:
    """Invoke the pinned Azure Responses native Tool adapter.

    Intro:
        Enforces the adapter's current Tool-only implementation before issuing one
        Azure Responses request.

    Examples:
        Invoke native Tool search:
            ```python
            result = await _invoke_azure_responses(client, tool_invocation)
            ```

        Reject direct Chat:
            ```python
            await _invoke_azure_responses(client, direct_invocation)
            ```

    Args:
        host: Bound generic client owning the Azure Responses method.
        call: Frozen prepared adapter invocation.

    Returns:
        AdapterResult: Raw Azure Tool response and transport metadata.

    Notes:
        Direct Azure Responses Chat remains unimplemented and fails without
        switching to Chat Completions.
    """

    if call.tool_request is None:
        raise LLMUnsupportedFeatureError(
            host.provider,
            call.model,
            "direct_chat",
            "the pinned Azure Responses adapter currently requires a Tool request",
        )
    return await AzureChatAdapter.invoke_responses(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        max_output_tokens=call.max_output_tokens,
        tool_request=call.tool_request,
        prompt_cache_fields=call.prompt_cache_fields,
        **call.option_dict(),
    )


async def _invoke_azure_chat_completions(
    host: Any,
    call: ChatAdapterInvocation,
) -> AdapterResult:
    """Invoke the pinned Azure Chat Completions adapter.

    Intro:
        Projects prepared direct, structured, and native Tool requests through
        the deployment-scoped Azure Chat Completions implementation.

    Examples:
        Invoke direct Chat:
            ```python
            result = await _invoke_azure_chat_completions(client, invocation)
            ```

        Invoke native Tools:
            ```python
            result = await _invoke_azure_chat_completions(client, tool_invocation)
            ```

    Args:
        host: Bound generic client owning the Azure Chat method.
        call: Frozen prepared adapter invocation.

    Returns:
        AdapterResult: Raw Azure Chat result and transport metadata.

    Notes:
        This handler never switches to Azure Responses.
    """

    options = call.option_dict()
    tools = options.pop("tools", None)
    tool_choice = options.pop("tool_choice", None)
    return await AzureChatAdapter.invoke_chat_completions(
        host,
        call.message_list(),
        model=call.model,
        max_output_tokens=call.max_output_tokens,
        output_format=call.output_format,
        json_schema=call.json_schema,
        fail_on_unsupported=call.fail_on_unsupported,
        tools=tools,
        tool_choice=tool_choice,
        tool_request=call.tool_request,
        structured_output_fields=call.structured_output_fields,
        **options,
    )


async def _invoke_anthropic_messages(host: Any, call: ChatAdapterInvocation) -> AdapterResult:
    """Invoke the exact Anthropic Messages adapter.

    Intro:
        Projects prepared system, thinking, structured-output, cache, and native
        Tool state into one Messages request.

    Examples:
        Invoke direct Messages generation:
            ```python
            result = await _invoke_anthropic_messages(client, invocation)
            ```

        Invoke native Tools:
            ```python
            result = await _invoke_anthropic_messages(client, tool_invocation)
            ```

    Args:
        host: Bound generic client owning the Anthropic single-attempt method.
        call: Frozen prepared adapter invocation.

    Returns:
        AdapterResult: Raw Anthropic result and transport metadata.

    Notes:
        Thinking controls are consumed once to prevent duplicate keyword paths.
    """

    options = call.option_dict()
    tools = options.pop("tools", None)
    options.pop("tool_choice", None)
    thinking_budget = options.pop("thinking_budget", None)
    thinking_mode = options.pop("thinking_mode", None)
    return await AnthropicMessagesAdapter.invoke(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        max_output_tokens=call.max_output_tokens,
        thinking_budget=thinking_budget,
        thinking_mode=thinking_mode,
        output_format=call.output_format,
        json_schema=call.json_schema,
        fail_on_unsupported=call.fail_on_unsupported,
        tools=tools,
        tool_request=call.tool_request,
        schema_name=call.schema_name,
        structured_output_fields=call.structured_output_fields,
        **options,
    )


async def _invoke_gemini_generate_content(
    host: Any,
    call: ChatAdapterInvocation,
) -> AdapterResult:
    """Invoke the exact Gemini GenerateContent adapter.

    Intro:
        Projects prepared thinking, structured-output, and native Tool state into
        one GenerateContent request without duplicate option keywords.

    Examples:
        Invoke direct generation:
            ```python
            result = await _invoke_gemini_generate_content(client, invocation)
            ```

        Invoke native Tools:
            ```python
            result = await _invoke_gemini_generate_content(client, tool_invocation)
            ```

    Args:
        host: Bound generic client owning the Gemini single-attempt method.
        call: Frozen prepared adapter invocation.

    Returns:
        AdapterResult: Raw Gemini result and transport metadata.

    Notes:
        `thinking_mode` is removed from the option mapping before the explicit
        argument is passed, fixing the prior duplicate-keyword risk.
    """

    options = call.option_dict()
    tools = options.pop("tools", None)
    options.pop("tool_choice", None)
    thinking_mode = options.pop("thinking_mode", None)
    return await GeminiGenerateContentAdapter.invoke(
        host,
        call.message_list(),
        model=call.model,
        reasoning_effort=call.reasoning_effort,
        thinking_mode=thinking_mode,
        max_output_tokens=call.max_output_tokens,
        output_format=call.output_format,
        json_schema=call.json_schema,
        fail_on_unsupported=call.fail_on_unsupported,
        tools=tools,
        tool_request=call.tool_request,
        structured_output_fields=call.structured_output_fields,
        **options,
    )


_CHAT_ADAPTER_RUNTIMES: dict[str, AdapterHandler] = {
    "openai_responses": _invoke_openai_responses,
    "openai_chat_completions": _invoke_chat_completions,
    "azure_responses": _invoke_azure_responses,
    "azure_chat_completions": _invoke_azure_chat_completions,
    "anthropic_messages": _invoke_anthropic_messages,
    "gemini_generate_content": _invoke_gemini_generate_content,
}


async def invoke_chat_adapter(
    host: Any,
    *,
    adapter_id: str,
    invocation: ChatAdapterInvocation,
) -> AdapterResult:
    """Invoke one exact registered non-streaming Chat adapter.

    Intro:
        Resolves only by pinned endpoint identity and delegates one physical
        attempt without provider fallback or shared lifecycle behavior.

    Examples:
        Invoke OpenAI Responses:
            ```python
            result = await invoke_chat_adapter(
                client,
                adapter_id="openai_responses",
                invocation=invocation,
            )
            ```

        Invoke Gemini GenerateContent:
            ```python
            result = await invoke_chat_adapter(
                client,
                adapter_id="gemini_generate_content",
                invocation=invocation,
            )
            ```

    Args:
        host: Bound generic client owning current single-attempt implementations.
        adapter_id: Exact selected endpoint-adapter identity.
        invocation: Frozen prepared single-attempt invocation.

    Returns:
        AdapterResult: Raw adapter value plus sanitized transport metadata.

    Notes:
        Missing runtime implementations fail closed. Descriptor registration alone
        never manufactures an executable adapter.
    """

    handler = _CHAT_ADAPTER_RUNTIMES.get(adapter_id)
    if handler is None:
        raise LLMUnsupportedFeatureError(
            host.provider,
            invocation.model,
            "chat",
            f"endpoint adapter {adapter_id!r} has no non-streaming implementation",
        )
    return await handler(host, invocation)


__all__ = ["ChatAdapterInvocation", "invoke_chat_adapter"]

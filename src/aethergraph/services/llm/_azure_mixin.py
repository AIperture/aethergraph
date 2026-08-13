"""Azure OpenAI methods (chat completions + image generation)."""

from __future__ import annotations

import json
from typing import Any

from aethergraph.services.llm._openai_mixin import (
    _openai_checkpoint_payload,
    _openai_function_tool,
    _openai_request_tools,
    _openai_tool_call_response,
)
from aethergraph.services.llm.adapters.openai_compatible import (
    _openai_like_continuation_messages,
    _openai_like_tool_call_response,
    _openai_like_tool_definitions,
    _stream_openai_like_chat_completions,
)
from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.tool_calling import (
    LLMToolCallResponseError,
    ToolCallRequest,
    ToolCallResponse,
)
from aethergraph.services.llm.types import (
    ChatOutputFormat,
    GeneratedImage,
    ImageGenerationResult,
)
from aethergraph.services.llm.utils import (
    _azure_images_generations_url,
    _ensure_system_json_directive,
    _guess_mime_from_format,
    _normalize_openai_responses_input,
)


def _first_text(choices):
    """Extract text and usage from OpenAI-style choices list."""
    if not choices:
        return "", {}
    c = choices[0]
    text = (c.get("message", {}) or {}).get("content") or c.get("text") or ""
    usage = {}
    return text, usage


class _AzureMixin:
    """Provider methods for Azure OpenAI."""

    async def _chat_azure_chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_output_tokens: int | None = None,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tool_request: ToolCallRequest | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
        """Invoke one Azure Chat Completions request.

        Intro:
            The adapter uses the explicitly pinned Azure Chat Completions route
            for direct, structured, Tool-call, and Tool-result continuation calls.

        Examples:
            Send a direct request:
                ```python
                result = await client._chat_azure_chat_completions(
                    messages,
                    model="deployment-a",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=False,
                )
                ```

            Continue a Tool request:
                ```python
                result = await client._chat_azure_chat_completions(
                    messages,
                    model="deployment-a",
                    output_format="text",
                    json_schema=None,
                    fail_on_unsupported=False,
                    tool_request=continued_request,
                )
                ```

        Args:
            messages: Provider-projected stable conversation messages.
            model: Exact Azure deployment identity.
            max_output_tokens: Optional maximum generated tokens.
            output_format: Requested text, JSON, or raw response mode.
            json_schema: Optional canonical JSON schema.
            fail_on_unsupported: Whether unsupported native formatting fails.
            tools: Optional legacy provider Tool declarations.
            tool_choice: Optional legacy provider Tool-selection policy.
            tool_request: Optional canonical native Tool request and continuation.
            **kw: Additional bounded Azure request options.

        Returns:
            ProviderCallResult[tuple[str | ToolCallResponse, dict[str, int]]]:
                Normalized response and raw Azure usage with transport metadata.

        Notes:
            The selected endpoint never switches to Azure Responses according to
            Tool presence. Shared retry and accounting remain caller-owned.
        """

        await self._ensure_client()
        assert self._client is not None

        if not (self.base_url and self.azure_deployment):
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
            )

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        replay_messages: tuple[dict[str, Any], ...] = ()
        if tool_request is not None:
            msg_for_provider, replay_messages = _openai_like_continuation_messages(
                messages,
                tool_request=tool_request,
                provider="azure",
                model=model,
            )
        payload: dict[str, Any] = {
            "messages": msg_for_provider,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        structured_output_fields = kw.pop("structured_output_fields", None)

        if structured_output_fields:
            payload.update(structured_output_fields)
        elif output_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            if fail_on_unsupported:
                raise RuntimeError(
                    "Azure native json_schema not guaranteed; set fail_on_unsupported=False for best-effort"
                )
            payload["messages"] = _ensure_system_json_directive(messages, schema=json_schema)

        if tool_request is not None:
            payload["tools"] = _openai_like_tool_definitions(tool_request)
            payload["tool_choice"] = tool_request.choice
            payload["parallel_tool_calls"] = tool_request.max_calls > 1
        elif tools is not None:
            payload["tools"] = tools
        if tool_request is None and tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async def _call():
            r = await self._client.post(
                f"{self.base_url}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2024-08-01-preview",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            metadata = checked_response_metadata("azure", model, "chat", r)

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return ProviderCallResult((txt, usage), metadata)

            if tool_request is not None:
                response = _openai_like_tool_call_response(
                    data,
                    tool_request=tool_request,
                    provider="azure",
                    model=model,
                    stable_messages=messages,
                    replay_messages=replay_messages,
                )
                return ProviderCallResult((response, usage), metadata)

            txt, _ = _first_text(data.get("choices", []))
            return ProviderCallResult((txt, usage), metadata)

        return await _call()

    async def _chat_azure_chat_completions_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        on_delta: Any = None,
        on_usage_update: Any = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[str, dict[str, int]]]:
        """Stream one pinned Azure Chat Completions request.

        Intro:
            Uses Azure's deployment-scoped Chat Completions SSE route and requests
            the terminal usage chunk without switching to Azure Responses.

        Examples:
            Stream default text:
                ```python
                result = await client._chat_azure_chat_completions_stream(
                    messages,
                    model="deployment-a",
                )
                ```

            Forward text deltas:
                ```python
                result = await client._chat_azure_chat_completions_stream(
                    messages,
                    model="deployment-a",
                    max_output_tokens=256,
                    on_delta=on_delta,
                    on_usage_update=on_usage_update,
                )
                ```

        Args:
            messages: Provider-projected stable conversation messages.
            model: Exact Azure deployment identity.
            reasoning_effort: Optional reasoning-depth override retained for the
                shared signature; the pinned legacy Azure route does not project it.
            max_output_tokens: Optional maximum generated tokens.
            on_delta: Optional async assistant-text callback.
            on_usage_update: Optional async cumulative usage callback.
            **kw: Additional bounded Azure sampling options.

        Returns:
            ProviderCallResult[tuple[str, dict[str, int]]]: Accumulated text,
                terminal provider usage when received, and transport metadata.

        Notes:
            Azure may omit the final usage chunk when a stream is interrupted.
            Canonical terminal usage then remains explicitly unavailable.
        """

        await self._ensure_client()
        assert self._client is not None
        if not (self.base_url and self.azure_deployment):
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
            )
        if str(model or "").strip() != str(self.azure_deployment).strip():
            raise ValueError("Azure Chat Completions model must match the configured deployment")

        body: dict[str, Any] = {
            "messages": messages,
            "temperature": kw.get("temperature", 0.5),
            "top_p": kw.get("top_p", 1.0),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        url = (
            f"{self.base_url}/openai/deployments/{self.azure_deployment}"
            "/chat/completions?api-version=2024-08-01-preview"
        )
        return await _stream_openai_like_chat_completions(
            http_client=self._client,
            provider="azure",
            model=model,
            url=url,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            body=body,
            on_delta=on_delta,
            on_usage_update=on_usage_update,
        )

    async def _chat_azure_responses(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
        tool_request: ToolCallRequest,
        prompt_cache_fields: dict[str, Any] | None = None,
        **kw: Any,
    ) -> ProviderCallResult[tuple[ToolCallResponse, dict[str, int]]]:
        """Run one Azure OpenAI Responses native Tool decision.

        The transport uses Azure's `/openai/v1/responses` route and API-key
        authentication while sharing the Responses Tool-search item protocol.

        Examples:
            Start client Tool search:
                ```python
                result = await client._chat_azure_responses(
                    messages,
                    model="gpt-5.5",
                    reasoning_effort=None,
                    max_output_tokens=512,
                    tool_request=request,
                )
                ```

            Continue a pending search:
                ```python
                result = await client._chat_azure_responses(
                    messages,
                    model="gpt-5.5",
                    reasoning_effort="medium",
                    max_output_tokens=512,
                    tool_request=continued_request,
                )
                ```

        Args:
            messages: Provider-neutral conversation messages.
            model: Exact Azure deployment binding.
            reasoning_effort: Optional Responses reasoning effort.
            max_output_tokens: Optional maximum response tokens.
            tool_request: Exact native Tool and discovery request.
            prompt_cache_fields: Optional prepared Responses cache fields.
            **kw: Additional bounded transport options.

        Returns:
            ProviderCallResult[tuple[ToolCallResponse, dict[str, int]]]: Normalized
                ordered response and raw provider usage.

        Notes:
            Chat Completions remains the non-Tool Azure path and is never treated
            as discovery-compatible.
        """

        await self._ensure_client()
        assert self._client is not None
        if not self.base_url:
            raise RuntimeError("Azure OpenAI Responses requires AZURE_OPENAI_ENDPOINT")
        deployment = str(self.azure_deployment or model or "").strip()
        if not deployment or deployment != str(model or "").strip():
            raise ValueError(
                "Azure discovery requires the exact model and deployment binding to match"
            )
        body: dict[str, Any] = {
            "model": deployment,
            "input": _normalize_openai_responses_input(messages),
        }
        if prompt_cache_fields:
            body.update(prompt_cache_fields)
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens

        checkpoint_payload: dict[str, Any] | None = None
        if tool_request.transport_checkpoint is not None:
            checkpoint_payload = _openai_checkpoint_payload(
                tool_request.transport_checkpoint,
                provider="azure",
            )
            if checkpoint_payload["state"] == "pending_search":
                prior_active_names = {
                    str(name) for name in list(checkpoint_payload.get("active_tool_names") or [])
                }
                newly_active_names = set(tool_request.active_tool_names) - prior_active_names
                loaded_tools = [
                    tool
                    for tool in tool_request.tools
                    if tool.exposure == "deferred" and tool.name in newly_active_names
                ]
                if not loaded_tools:
                    raise LLMToolCallResponseError(
                        code="discovery_result_missing",
                        message="Azure client Tool search has no newly activated result.",
                    )
                assert tool_request.discovery is not None
                if len(loaded_tools) > tool_request.discovery.max_results:
                    raise LLMToolCallResponseError(
                        code="discovery_result_limit_exceeded",
                        message="Azure client Tool-search results exceed the request bound.",
                    )
                previous_output = checkpoint_payload.get("response_output")
                if not isinstance(previous_output, list) or not previous_output:
                    raise ValueError("Azure Tool checkpoint output history is invalid")
                body["input"] = [
                    *[dict(item) for item in previous_output if isinstance(item, dict)],
                    {
                        "type": "tool_search_output",
                        "execution": "client",
                        "call_id": str(checkpoint_payload.get("call_id") or ""),
                        "status": "completed",
                        "tools": [
                            _openai_function_tool(tool, defer_loading=True) for tool in loaded_tools
                        ],
                    },
                ]
        if checkpoint_payload is None or checkpoint_payload["state"] != "pending_search":
            body["tools"] = _openai_request_tools(tool_request)
            body["tool_choice"] = tool_request.choice
            body["parallel_tool_calls"] = tool_request.max_calls > 1

        request_timeout = kw.get("request_timeout_s") or kw.get("timeout")
        normalized_base = str(self.base_url).rstrip("/")
        url = (
            f"{normalized_base}/responses"
            if normalized_base.endswith("/openai/v1")
            else f"{normalized_base}/openai/v1/responses"
        )
        r = await self._client.post(
            url,
            headers={"api-key": self.api_key, "Content-Type": "application/json"},
            json=body,
            timeout=request_timeout,
        )
        metadata = checked_response_metadata("azure", model, "chat", r)
        data = r.json()
        if data.get("status") == "incomplete":
            raise LLMToolCallResponseError(
                code="truncated",
                message="Azure Tool-call response was incomplete.",
            )
        return ProviderCallResult(
            (
                _openai_tool_call_response(
                    data,
                    tool_request=tool_request,
                    model=model,
                    provider="azure",
                ),
                data.get("usage", {}) or {},
            ),
            metadata,
        )

    async def _image_azure_generate(
        self,
        prompt: str,
        *,
        model: str,
        n: int,
        size: str | None,
        quality: str | None,
        style: str | None,
        output_format: Any | None,
        response_format: Any | None,
        background: str | None,
        azure_api_version: str | None,
        **kw: Any,
    ) -> ProviderCallResult[ImageGenerationResult]:
        assert self._client is not None

        if not self.base_url or not self.azure_deployment:
            raise RuntimeError(
                "Azure generate_image requires base_url=<resource endpoint> and azure_deployment=<deployment name>"
            )

        api_version = azure_api_version or "2025-04-01-preview"
        url = _azure_images_generations_url(self.base_url, self.azure_deployment, api_version)

        headers = {"api-key": self.api_key, "Content-Type": "application/json"}

        body: dict[str, Any] = {"prompt": prompt, "n": n}

        if model:
            body["model"] = model
        if size is not None:
            body["size"] = size
        if quality is not None:
            body["quality"] = quality
        if style is not None:
            body["style"] = style
        if response_format is not None:
            body["response_format"] = response_format
        if output_format is not None:
            body["output_format"] = output_format.upper()
        if background is not None:
            body["background"] = background

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            metadata = checked_response_metadata("azure", model, "image", r)

            data = r.json()
            imgs: list[GeneratedImage] = []
            for item in data.get("data", []) or []:
                imgs.append(
                    GeneratedImage(
                        b64=item.get("b64_json"),
                        url=item.get("url"),
                        mime_type=_guess_mime_from_format((output_format or "png").lower())
                        if item.get("b64_json")
                        else None,
                        revised_prompt=item.get("revised_prompt"),
                    )
                )

            return ProviderCallResult(
                ImageGenerationResult(images=imgs, usage=data.get("usage", {}) or {}, raw=data),
                metadata,
            )

        return await _call()

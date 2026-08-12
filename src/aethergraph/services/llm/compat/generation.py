"""Temporary canonical-request projection into the existing Chat boundary."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

from ..contracts import ChatMessage, ModelRequest, TextPart
from ..tool_calling import ToolCallRequest
from ..types import ImageInput, StructuredOutputRequest


@dataclass(frozen=True)
class LegacyChatProjection:
    """Carry one explicit canonical-request projection into `chat()`.

    The value exists only at the named compatibility boundary while the current
    provider implementation is moved behind `generate()`.

    Examples:
        Inspect projected messages:
            ```python
            projection = project_model_request_to_chat(request)
            assert projection.messages[0]["role"] == "user"
            ```

        Inspect projected keyword arguments:
            ```python
            projection = project_model_request_to_chat(request)
            assert "max_output_tokens" in projection.kwargs
            ```

    Args:
        messages: Detached legacy message dictionaries.
        kwargs: Explicit legacy `chat()` keyword arguments.

    Returns:
        LegacyChatProjection: Immutable boundary projection.

    Notes:
        Canonical runtime code must not consume this value after the Chat
        implementation is inverted behind `generate()`.
    """

    messages: tuple[dict[str, Any], ...]
    kwargs: dict[str, Any]


def project_model_request_to_chat(request: ModelRequest) -> LegacyChatProjection:
    """Project one canonical model request into the current Chat facade.

    The projection is deterministic and rejects no capability combination;
    canonical and adapter preflight remain responsible for those decisions.

    Examples:
        Project a direct request:
            ```python
            projection = project_model_request_to_chat(request)
            assert projection.kwargs["output_format"] == "text"
            ```

        Project a Tool request:
            ```python
            projection = project_model_request_to_chat(tool_request)
            assert projection.kwargs["tool_request"].choice == "auto"
            ```

    Args:
        request: Canonical provider-neutral generation request.

    Returns:
        LegacyChatProjection: Detached messages and explicit facade arguments.

    Notes:
        This is a one-way cutover adapter, not a retry or provider fallback.
    """

    if not isinstance(request, ModelRequest):
        raise TypeError("request must be a ModelRequest")
    kwargs: dict[str, Any] = {
        "reasoning_effort": request.generation.reasoning_effort,
        "max_output_tokens": request.generation.max_output_tokens,
        "prompt_cache": request.prompt_cache,
    }
    if request.generation.temperature is not None:
        kwargs["temperature"] = request.generation.temperature
    if request.generation.reasoning_budget is not None:
        kwargs["thinking_budget"] = request.generation.reasoning_budget
    if request.generation.reasoning_summary is not None:
        kwargs["reasoning_summary"] = request.generation.reasoning_summary
    if isinstance(request.response_format, StructuredOutputRequest):
        kwargs["structured_output"] = request.response_format
    else:
        kwargs["output_format"] = request.response_format
    if request.tools:
        kwargs["tool_request"] = ToolCallRequest(
            tools=request.tools,
            choice=request.tool_choice,
            max_calls=request.max_tool_calls,
            discovery=request.native_tool_search,
            turn_id=request.turn_id,
            active_tool_names=request.active_tool_names,
            transport_checkpoint=request.continuation,
            tool_outputs=request.tool_outputs,
        )
    return LegacyChatProjection(
        messages=tuple(_project_message(message) for message in request.messages),
        kwargs=kwargs,
    )


def _project_message(message: ChatMessage) -> dict[str, Any]:
    content: str | list[dict[str, Any]]
    if len(message.content) == 1 and isinstance(message.content[0], TextPart):
        content = message.content[0].text
    else:
        content = [_project_content_part(part) for part in message.content]
    projected: dict[str, Any] = {"role": message.role, "content": content}
    if message.name is not None:
        projected["name"] = message.name
    if message.tool_call_id is not None:
        projected["tool_call_id"] = message.tool_call_id
    return projected


def _project_content_part(part: TextPart | ImageInput) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if part.url:
        return {"type": "image_url", "image_url": {"url": part.url}}
    if part.b64 is not None and part.mime_type:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime_type,
                "data": part.b64,
            },
        }
    if part.data is not None and part.mime_type:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime_type,
                "data": base64.b64encode(part.data).decode("ascii"),
            },
        }
    raise ValueError("ImagePart requires a URL or bytes/base64 with MIME type")


__all__ = ["LegacyChatProjection", "project_model_request_to_chat"]

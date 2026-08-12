"""Canonical request preparation shared by estimation and generation."""

from __future__ import annotations

import base64
from typing import Any

from .contracts import MODEL_REQUEST_CONTRACT_VERSION, ModelRequest, TextPart
from .tool_calling import ToolCallRequest
from .types import ImageInput


def prepare_model_request(
    request: ModelRequest,
) -> tuple[list[dict[str, Any]], ToolCallRequest | None]:
    """Prepare canonical messages and Tools for the shared adapter seam.

    Intro:
        Converts only provider-neutral content parts and Tool contracts into the
        stable dictionary inputs consumed by the adjacent endpoint adapters.

    Examples:
        Prepare a direct request:
            ```python
            messages, tool_request = prepare_model_request(request)
            assert tool_request is None
            ```

        Prepare a native Tool request:
            ```python
            messages, tool_request = prepare_model_request(request)
            assert tool_request.fingerprint_version == "model_request/v1"
            ```

    Args:
        request: Complete immutable canonical generation request.

    Returns:
        tuple[list[dict[str, Any]], ToolCallRequest | None]: Detached stable
            messages and the optional canonical Tool transport request.

    Notes:
        This is provider preparation inside the generation runtime, not a
        projection into the public `chat()` facade or an alternate request type.
    """

    messages: list[dict[str, Any]] = []
    for message in request.messages:
        content: str | list[dict[str, Any]]
        if len(message.content) == 1 and isinstance(message.content[0], TextPart):
            content = message.content[0].text
        else:
            content = [_prepare_content_part(part) for part in message.content]
        prepared: dict[str, Any] = {"role": message.role, "content": content}
        if message.name is not None:
            prepared["name"] = message.name
        if message.tool_call_id is not None:
            prepared["tool_call_id"] = message.tool_call_id
        messages.append(prepared)

    tool_request = None
    if request.tools:
        tool_request = ToolCallRequest(
            tools=request.tools,
            choice=request.tool_choice,
            max_calls=request.max_tool_calls,
            discovery=request.native_tool_search,
            turn_id=request.turn_id,
            active_tool_names=request.active_tool_names,
            transport_checkpoint=request.continuation,
            tool_outputs=request.tool_outputs,
            fingerprint_version=MODEL_REQUEST_CONTRACT_VERSION,
        )
    return messages, tool_request


def _prepare_content_part(part: TextPart | ImageInput) -> dict[str, Any]:
    """Prepare one canonical content part for adjacent adapters.

    Intro:
        Text and image values are converted into the stable internal block shapes
        already normalized by every pinned provider adapter.

    Examples:
        Prepare text:
            ```python
            assert _prepare_content_part(TextPart("Hi"))["type"] == "text"
            ```

        Prepare a URL image:
            ```python
            block = _prepare_content_part(ImageInput(url="https://example/image.png"))
            assert block["type"] == "image_url"
            ```

    Args:
        part: One provider-neutral text or image content part.

    Returns:
        dict[str, Any]: Detached stable internal content block.

    Notes:
        Byte images are base64 encoded once. Provider wire conversion remains
        owned by the selected adapter's existing message normalizer.
    """

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


__all__ = ["prepare_model_request"]

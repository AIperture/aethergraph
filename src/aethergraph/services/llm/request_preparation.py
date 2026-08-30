"""Canonical request preparation shared by estimation and generation."""

from __future__ import annotations

import base64
import copy
from typing import Any

from .contracts import MODEL_REQUEST_CONTRACT_VERSION, ModelRequest, TextPart
from .media import ImagePreparationPolicy, prepare_image_inputs
from .tool_calling import ToolCallRequest
from .types import ImageInput


def prepare_model_request(
    request: ModelRequest,
    *,
    image_policy: ImagePreparationPolicy | None = None,
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
            messages, tool_request = prepare_model_request(
                request,
                image_policy=client.image_preparation_policy,
            )
            assert tool_request.fingerprint_version == "model_request/v1"
            ```

    Args:
        request: Complete immutable canonical generation request.
        image_policy: Optional managed profile policy for whole-request image
            admission and normalization.

    Returns:
        tuple[list[dict[str, Any]], ToolCallRequest | None]: Detached stable
            messages and the optional canonical Tool transport request.

    Notes:
        This is provider preparation inside the generation runtime, not a
        projection into the public `chat()` facade or an alternate request type.
    """

    image_parts = tuple(
        part
        for message in request.messages
        for part in message.content
        if isinstance(part, ImageInput)
    )
    prepared_images = iter(
        prepare_image_inputs(image_parts, policy=image_policy)
        if image_policy is not None
        else image_parts
    )
    messages: list[dict[str, Any]] = []
    for message in request.messages:
        content: str | list[dict[str, Any]]
        if len(message.content) == 1 and isinstance(message.content[0], TextPart):
            content = message.content[0].text
        else:
            content = [
                _prepare_content_part(
                    next(prepared_images) if isinstance(part, ImageInput) else part
                )
                for part in message.content
            ]
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
            discovery_result=request.discovery_result,
            fingerprint_version=MODEL_REQUEST_CONTRACT_VERSION,
        )
    return messages, tool_request


def prepare_chat_messages(
    messages: list[dict[str, Any]],
    *,
    image_policy: ImagePreparationPolicy | None,
) -> list[dict[str, Any]]:
    """Apply managed media policy to legacy dictionary Chat messages.

    Intro:
        Finds established OpenAI, Responses, and Anthropic-compatible image
        blocks, runs the same whole-request admission as canonical generation,
        and replaces only their payload fields in a detached message copy.

    Examples:
        Preserve text-only message identity:
            ```python
            messages = [{"role": "user", "content": "Hello"}]
            assert prepare_chat_messages(messages, image_policy=None) is messages
            ```

        Admit a remote image under an explicit policy:
            ```python
            messages = [{"role": "user", "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.test/image.png"},
            }]}]
            prepared = prepare_chat_messages(
                messages,
                image_policy=ImagePreparationPolicy(allow_remote_urls=True),
            )
            ```

    Args:
        messages: Legacy dictionary messages accepted by `chat()` and
            `chat_stream()`.
        image_policy: Optional managed image policy; `None` preserves the
            standalone compatibility behavior exactly.

    Returns:
        list[dict[str, Any]]: Original messages when unmanaged or image-free,
            otherwise a detached list with normalized image payloads.

    Notes:
        Unknown or malformed content block shapes remain untouched for the
        pinned provider adapter to diagnose as before.
    """

    if image_policy is None:
        return messages
    occurrences: list[tuple[int, int, str]] = []
    images: list[ImageInput] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            parsed = _image_input_from_block(part)
            if parsed is None:
                continue
            image, shape = parsed
            occurrences.append((message_index, part_index, shape))
            images.append(image)
    if not images:
        return messages

    prepared_images = prepare_image_inputs(tuple(images), policy=image_policy)
    prepared_messages = copy.deepcopy(messages)
    for (message_index, part_index, shape), image in zip(
        occurrences,
        prepared_images,
        strict=True,
    ):
        block = prepared_messages[message_index]["content"][part_index]
        _replace_image_block(block, shape=shape, image=image)
    return prepared_messages


def _image_input_from_block(part: Any) -> tuple[ImageInput, str] | None:
    if not isinstance(part, dict):
        return None
    block_type = part.get("type")
    if block_type == "image_url":
        image_url = part.get("image_url")
        url = image_url.get("url") if isinstance(image_url, dict) else part.get("url")
        return (ImageInput(url=url), "image_url") if isinstance(url, str) else None
    if block_type == "input_image":
        url = part.get("image_url")
        return (ImageInput(url=url), "input_image") if isinstance(url, str) else None
    if block_type != "image" or not isinstance(part.get("source"), dict):
        return None
    source = part["source"]
    if source.get("type") == "base64":
        data = source.get("data")
        mime_type = source.get("media_type")
        if isinstance(data, str) and isinstance(mime_type, str):
            return ImageInput(b64=data, mime_type=mime_type), "image_base64"
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return ImageInput(url=source["url"]), "image_source_url"
    return None


def _replace_image_block(
    block: dict[str, Any],
    *,
    shape: str,
    image: ImageInput,
) -> None:
    if image.url is not None:
        projected_url = image.url
    elif image.data is not None and image.mime_type:
        encoded = base64.b64encode(image.data).decode("ascii")
        projected_url = f"data:{image.mime_type};base64,{encoded}"
    else:
        raise ValueError("prepared image requires a URL or bytes with MIME type")

    if shape == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            image_url["url"] = projected_url
        else:
            block["url"] = projected_url
        return
    if shape == "input_image":
        block["image_url"] = projected_url
        return
    if shape == "image_source_url" and image.url is not None:
        block["source"]["url"] = image.url
        return
    if image.data is None or not image.mime_type:
        raise ValueError("base64 image block requires prepared bytes")
    block["source"] = {
        **block["source"],
        "type": "base64",
        "media_type": image.mime_type,
        "data": base64.b64encode(image.data).decode("ascii"),
    }


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


__all__ = ["prepare_chat_messages", "prepare_model_request"]

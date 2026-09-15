"""Canonical pixels must survive real preparation into each adapter's HTTP body."""

import asyncio
import base64
import io
import json

import httpx
import pytest
from PIL import Image

from aethergraph.services.llm.contracts import ChatMessage, ModelRequest, TextPart
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.media import ImagePreparationPolicy, MediaPreparationError
from aethergraph.services.llm.request_preparation import prepare_model_request
from aethergraph.services.llm.types import ImageInput, LLMUnsupportedFeatureError
from aethergraph.services.llm.tool_calling import ModelToolSpec
from aethergraph.services.llm.utils import (
    _normalize_openai_responses_input,
    _to_anthropic_blocks,
    _to_gemini_parts,
)


def raster(format_name="PNG", color="red"):
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format=format_name)
    return output.getvalue()


@pytest.mark.parametrize("format_name,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
@pytest.mark.parametrize("representation", ["data", "b64", "url"])
def test_prepared_images_have_one_shared_adapter_representation(format_name, mime, representation):
    raw = raster(format_name)
    value = {
        "data": raw,
        "b64": base64.b64encode(raw).decode(),
        "url": f"data:{mime};base64,{base64.b64encode(raw).decode()}",
    }[representation]
    request = ModelRequest(
        messages=(
            ChatMessage(
                "user",
                (
                    TextPart("first"),
                    ImageInput(**{representation: value}, mime_type=mime),
                    TextPart("second"),
                    ImageInput(data=raster(format_name, "blue"), mime_type=mime),
                ),
            ),
        )
    )
    messages, _ = prepare_model_request(
        request, image_policy=ImagePreparationPolicy(resize_enabled=False)
    )
    parts = messages[0]["content"]
    assert [part["type"] for part in parts] == ["text", "image_url", "text", "image_url"]
    assert parts[0]["text"] == "first" and parts[2]["text"] == "second"
    assert base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]) == raw
    assert parts[1] != parts[3]
    assert _normalize_openai_responses_input(messages)[0]["content"][1]["type"] == "input_image"
    assert _to_gemini_parts(parts)[1]["inline_data"]["mime_type"] == mime
    assert _to_anthropic_blocks(parts)[1]["source"]["media_type"] == mime


class TransportReached(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint",
    ["openai_responses", "openai_chat_completions", "azure_responses", "azure_chat_completions"],
)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("native_tools", [False, True])
async def test_image_projection_reaches_real_transport(endpoint, streaming, native_tools):
    captured = []

    def receive(request):
        captured.append(json.loads(request.content))
        raise TransportReached()

    client = GenericLLMClient(
        provider="azure" if endpoint.startswith("azure") else "openai",
        endpoint_id=endpoint,
        model="gpt-4o",
        azure_deployment="gpt-4o",
        base_url="https://example.test",
        api_key="test",
        image_preparation_policy=ImagePreparationPolicy(resize_enabled=False),
    )
    request = ModelRequest(
        messages=(
            ChatMessage(
                "user", (TextPart("Describe"), ImageInput(data=raster(), mime_type="image/png"))
            ),
        ),
        tools=(
            ModelToolSpec(
                name="inspect",
                description="Inspect",
                input_schema={"type": "object", "properties": {}},
            ),
        )
        if native_tools
        else (),
        tool_choice="auto" if native_tools else "none",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as transport:
        client._client = transport
        client._bound_loop = asyncio.get_running_loop()
        unsupported_stream = (streaming and (endpoint == "azure_responses" or native_tools)) or (
            endpoint == "azure_responses" and not native_tools
        )
        with pytest.raises(LLMUnsupportedFeatureError if unsupported_stream else TransportReached):
            if streaming:
                async for _ in client.generate_stream(request):
                    pass
            else:
                await client.generate(request)
    if unsupported_stream:
        assert captured == []
        return
    assert len(captured) == 1
    key = "input" if "responses" in endpoint else "messages"
    expected_type = "input_image" if key == "input" else "image_url"
    assert captured[0][key][0]["content"][1]["type"] == expected_type


@pytest.mark.parametrize("resize", [False, True])
def test_mime_mismatch_fails_in_preparation(resize):
    request = ModelRequest(
        messages=(ChatMessage("user", (ImageInput(data=raster(), mime_type="image/jpeg"),)),)
    )
    with pytest.raises(MediaPreparationError, match="MIME.*match"):
        prepare_model_request(request, image_policy=ImagePreparationPolicy(resize_enabled=resize))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "invalid_base64",
        "invalid_bytes",
        "mime_mismatch",
        "missing_source",
        "count",
        "byte_limit",
        "assistant_role",
    ],
)
async def test_invalid_canonical_image_never_reaches_transport(case):
    policy = ImagePreparationPolicy(resize_enabled=False)
    image = ImageInput(data=raster(), mime_type="image/png")
    role = "user"
    if case == "disabled":
        policy = ImagePreparationPolicy(image_input_enabled=False)
    elif case == "invalid_base64":
        image = ImageInput(b64="invalid!", mime_type="image/png")
    elif case == "invalid_bytes":
        image = ImageInput(data=b"not a raster", mime_type="image/png")
    elif case == "mime_mismatch":
        image = ImageInput(data=raster(), mime_type="image/jpeg")
    elif case == "missing_source":
        image = ImageInput(mime_type="image/png")
    elif case == "count":
        policy = ImagePreparationPolicy(max_images=1)
    elif case == "byte_limit":
        policy = ImagePreparationPolicy(max_image_bytes=1, resize_enabled=False)
    elif case == "assistant_role":
        role = "assistant"
    images = (image, image) if case == "count" else (image,)
    client = GenericLLMClient(
        provider="openai",
        endpoint_id="openai_responses",
        model="gpt-4o",
        api_key="test",
        image_preparation_policy=policy,
    )

    def forbidden(request):
        pytest.fail("invalid image reached HTTP transport")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as transport:
        client._client = transport
        client._bound_loop = asyncio.get_running_loop()
        with pytest.raises(ValueError):
            await client.generate(ModelRequest(messages=(ChatMessage(role, images),)))


def test_responses_projection_preserves_explicit_detail():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,eA==", "detail": "high"},
                }
            ],
        }
    ]
    assert _normalize_openai_responses_input(messages)[0]["content"][0]["detail"] == "high"

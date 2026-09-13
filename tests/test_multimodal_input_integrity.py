"""No admitted facade may silently turn multimodal input into text-only work."""

import pytest

from aethergraph.services.llm.generic_image_client import GenericImageGenerationClient
from aethergraph.services.llm.profiles import ImageGenerationCapabilityOverrides
from aethergraph.services.llm.request_preparation import prepare_chat_messages
from aethergraph.services.llm.types import LLMUnsupportedFeatureError
from aethergraph.services.llm.utils import (
    _normalize_openai_responses_input,
    _to_anthropic_blocks,
    _to_gemini_parts,
)


@pytest.mark.parametrize(
    "convert,block",
    [
        (
            _to_anthropic_blocks,
            {"type": "document", "source": {"type": "base64", "data": "x"}},
        ),
        (_to_gemini_parts, {"file_data": {"file_uri": "gs://example/file"}}),
        (
            lambda parts: _normalize_openai_responses_input(
                [{"role": "user", "content": parts}]
            ),
            {"type": "input_file", "file_id": "file-1"},
        ),
    ],
)
def test_unsupported_parts_are_rejected_not_emptied(convert, block):
    with pytest.raises(ValueError, match=r"content\[0\]"):
        convert([block])


@pytest.mark.parametrize("policy", [None])
@pytest.mark.parametrize(
    "block",
    [
        {"type": "audio"},
        {"type": "video"},
        {"type": "image_url", "image_url": {}},
        {"type": "future_part"},
        42,
    ],
)
def test_legacy_preparation_rejects_unknown_shapes_without_provider_io(policy, block):
    with pytest.raises(ValueError, match=r"messages\[0\].content\[0\]"):
        prepare_chat_messages(
            [{"role": "user", "content": [block]}], image_policy=policy
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,endpoint", [("openai", "openai_images"), ("azure", "azure_images")]
)
async def test_malformed_edits_cannot_call_transport_even_with_profile_override(
    monkeypatch, provider, endpoint
):
    async def forbidden(*args, **kwargs):
        pytest.fail("unsupported editing reached transport")

    monkeypatch.setattr(
        "aethergraph.services.llm.generic_image_client._execute_image_generation",
        forbidden,
    )
    client = GenericImageGenerationClient(
        provider=provider,
        endpoint_id=endpoint,
        model="gpt-image-2",
        api_key="test",
        capability_overrides=ImageGenerationCapabilityOverrides(
            image_editing="supported"
        ),
    )
    from aethergraph.services.llm.media import MediaPreparationError
    with pytest.raises(MediaPreparationError):
        await client.generate_image(
            "Edit", input_images=["data:image/png;base64,aW1hZ2U="]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("uncataloged-test-model", "gpt-image-2.5-unknown", "gpt-image-2.5-flare-2099-01-01"))
async def test_unknown_per_call_model_does_not_inherit_catalog_support(monkeypatch, model):
    async def forbidden(*args, **kwargs):
        pytest.fail("unknown model reached transport")

    monkeypatch.setattr(
        "aethergraph.services.llm.generic_image_client._execute_image_generation",
        forbidden,
    )
    client = GenericImageGenerationClient(
        provider="openai",
        endpoint_id="openai_images",
        model="gpt-image-2",
        api_key="test",
    )
    with pytest.raises(LLMUnsupportedFeatureError, match="unknown"):
        await client.generate_image("Generate", model=model)

@pytest.mark.asyncio
@pytest.mark.parametrize("provider,endpoint,model", [
    ("azure", "azure_images", "gpt-image-2"),
    *(("openai", "openai_images", model) for model in (
        "gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare",
        "gpt-image-2.5-sunburst-2026-09-08", "gpt-image-2.5-flare-2026-09-08",
    )),
])
async def test_edit_reference_bytes_reach_selected_multipart_endpoint(provider, endpoint, model):
    import asyncio
    import io

    import httpx
    from PIL import Image

    from aethergraph.services.llm.types import ImageInput

    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    payload = output.getvalue()
    requests = []

    def receive(request):
        requests.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": "eA=="}]})

    client = GenericImageGenerationClient(
        provider=provider, endpoint_id=endpoint, model=model, api_key="test",
        base_url="https://example.test", azure_deployment="images",
        # Azure deployment facts are explicit; OpenAI must resolve from the catalog alone.
        capability_overrides=(
            ImageGenerationCapabilityOverrides(image_editing="supported")
            if provider == "azure" else ImageGenerationCapabilityOverrides()
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(receive)) as transport:
        client._client = transport
        client._bound_loop = asyncio.get_running_loop()
        result = await client.generate_image("Edit", input_images=[ImageInput(data=payload, mime_type="image/png")])
    assert len(result.images) == 1
    assert len(requests) == 1
    request = requests[0]
    assert "/images/edits" in str(request.url)
    assert "generations" not in str(request.url)
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert payload in request.content
    assert b'name="image[]"' in request.content
    assert b"Edit" in request.content
    if provider == "openai":
        assert model.encode() in request.content
    if provider == "azure":
        assert request.url.params["api-version"] == "2025-04-01-preview"

@pytest.mark.asyncio
async def test_image_edit_profile_and_unprojected_options_fail_before_http(monkeypatch):
    from aethergraph.services.llm.media import MediaPreparationError
    from aethergraph.services.llm.profiles import MultimodalInputPolicy
    client = GenericImageGenerationClient(
        provider="openai", endpoint_id="openai_images", model="gpt-image-2", api_key="test",
        input_policy=MultimodalInputPolicy(image_input_enabled=False),
    )
    with pytest.raises(MediaPreparationError, match="disabled"):
        await client.generate_image("Edit", input_images=["data:image/png;base64,eA=="])

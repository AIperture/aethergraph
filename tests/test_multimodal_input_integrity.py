"""No admitted facade may silently turn multimodal input into text-only work."""

import pytest
from aethergraph.services.llm.generic_image_client import GenericImageGenerationClient
from aethergraph.services.llm.profiles import ImageGenerationCapabilityOverrides
from aethergraph.services.llm.request_preparation import prepare_chat_messages
from aethergraph.services.llm.types import LLMUnsupportedFeatureError
from aethergraph.services.llm.utils import (
    _to_anthropic_blocks,
    _to_gemini_parts,
    _normalize_openai_responses_input,
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
async def test_unimplemented_edits_cannot_call_transport_even_with_profile_override(
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
    with pytest.raises(LLMUnsupportedFeatureError, match="image_editing"):
        await client.generate_image(
            "Edit", input_images=["data:image/png;base64,aW1hZ2U="]
        )


@pytest.mark.asyncio
async def test_unknown_per_call_model_does_not_inherit_catalog_support(monkeypatch):
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
        await client.generate_image("Generate", model="uncataloged-test-model")

"""Image transport bytes must not inflate the temporary text-only estimate."""
import copy
import io

from PIL import Image

from aethergraph.services.llm.contracts import ChatMessage, ModelRequest, TextPart
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.media import ImagePreparationPolicy
from aethergraph.services.llm.request_preparation import prepare_model_request
from aethergraph.services.llm.types import ImageInput


def test_estimate_ignores_image_payload_and_preserves_request():
    client = GenericLLMClient(provider="openai", model="gpt-5.6-luna", api_key="test", endpoint_id="openai_responses", image_preparation_policy=ImagePreparationPolicy(resize_enabled=False))
    raster = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(raster, format="PNG")
    request = ModelRequest(messages=(ChatMessage("user", (TextPart("Describe the image"), ImageInput(data=raster.getvalue(), mime_type="image/png"))),))
    before, _ = prepare_model_request(request, image_policy=client.image_preparation_policy)
    estimate = client.estimate(request)
    after, _ = prepare_model_request(request, image_policy=client.image_preparation_policy)
    assert before == after
    assert after[0]["content"][1]["type"] == "image_url"
    assert estimate.source == "approximate_chars_div_4_images_excluded"
    assert estimate.estimated_input_tokens < 100


def test_transport_size_and_url_do_not_change_estimate_but_text_does():
    client = object.__new__(GenericLLMClient)
    messages = [{"role": "user", "content": [{"type": "text", "text": "Describe it"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 400000}}]}]
    saved = copy.deepcopy(messages)
    small = copy.deepcopy(messages)
    small[0]["content"][1]["image_url"]["url"] = "https://example.test/image.png"
    assert client._estimate_messages_tokens(messages) == client._estimate_messages_tokens(small)
    assert messages == saved
    larger_text = copy.deepcopy(small)
    larger_text[0]["content"][0]["text"] += "word " * 200
    assert client._estimate_messages_tokens(larger_text) > client._estimate_messages_tokens(small) + 200
    only_image = [{"role": "user", "content": [saved[0]["content"][1]]}]
    assert client._estimate_messages_tokens(only_image) == client._estimate_messages_tokens([{"role": "user", "content": None}])


def test_non_image_structured_content_is_still_counted():
    client = object.__new__(GenericLLMClient)
    short = [{"role": "user", "content": [{"type": "custom", "value": "x"}]}]
    long = [{"role": "user", "content": [{"type": "custom", "value": "x" * 1000}]}]
    assert client._estimate_messages_tokens(long) > client._estimate_messages_tokens(short) + 200

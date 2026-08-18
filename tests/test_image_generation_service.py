from __future__ import annotations

import asyncio

import httpx
import pytest

from aethergraph.config.llm import ImageGenerationProfileSettings, ImageGenerationSettings
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.llm.generic_image_client import GenericImageGenerationClient
from aethergraph.services.llm.image_factory import build_image_generation_clients
from aethergraph.services.llm.image_service import ImageGenerationService
from aethergraph.services.llm.service import LLMService
from aethergraph.services.llm.types import ImageGenerationResult, LLMUnsupportedFeatureError


class _Secrets:
    def get(self, name: str) -> str | None:
        return None


class _FakeHttp:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_url: str | None = None
        self.last_json: dict | None = None

    async def post(self, url: str, *, headers: dict, json: dict) -> httpx.Response:
        self.last_url = url
        self.last_json = json
        return httpx.Response(
            200,
            json=self.payload,
            request=httpx.Request("POST", url, headers=headers),
        )


@pytest.mark.asyncio
async def test_independent_image_client_applies_profile_defaults_and_exact_endpoint() -> None:
    class _Metering:
        def __init__(self) -> None:
            self.image_records: list[dict] = []
            self.llm_records: list[dict] = []

        async def record_image_generation(self, **record) -> None:
            self.image_records.append(record)

        async def record_llm(self, **record) -> None:
            self.llm_records.append(record)

    metering = _Metering()
    client = GenericImageGenerationClient(
        provider="openai",
        model="gpt-image-test",
        endpoint_id="openai_images",
        api_key="test",
        default_count=2,
        default_size="1024x1024",
        default_output_format="png",
        default_response_format="b64_json",
        metering=metering,  # type: ignore[arg-type]
    )
    fake_http = _FakeHttp(
        {
            "data": [{"b64_json": "aW1hZ2U="}, {"b64_json": "aW1hZ2Uy"}],
            "usage": {"input_tokens": 4, "output_tokens": 6},
        }
    )
    client._client = fake_http  # type: ignore[assignment]
    client._bound_loop = asyncio.get_running_loop()

    result = await client.generate_image("A glass compass")

    assert len(result.images) == 2
    assert fake_http.last_url == "https://api.openai.com/v1/images/generations"
    assert fake_http.last_json is not None
    assert fake_http.last_json["n"] == 2
    assert fake_http.last_json["size"] == "1024x1024"
    assert fake_http.last_json["output_format"] == "png"
    assert fake_http.last_json["response_format"] == "b64_json"
    assert result.usage == {"input_tokens": 4, "output_tokens": 6}
    assert result.usage_receipt is not None
    assert result.usage_receipt.availability == "complete"
    assert result.usage_receipt.total_tokens == 10
    assert metering.llm_records == []
    assert metering.image_records == [
        {
            "user_id": None,
            "org_id": None,
            "run_id": None,
            "graph_id": None,
            "provider": "openai",
            "model": "gpt-image-test",
            "image_count": 2,
            "size": "1024x1024",
            "quality": None,
            "input_tokens": 4,
            "output_tokens": 6,
            "total_tokens": 10,
            "usage_availability": "complete",
            "latency_ms": metering.image_records[0]["latency_ms"],
        }
    ]


def test_image_factory_builds_named_profiles_without_chat_configuration() -> None:
    clients = build_image_generation_clients(
        ImageGenerationSettings(
            default=ImageGenerationProfileSettings(model="gpt-image-default"),
            profiles={
                "design": ImageGenerationProfileSettings(
                    provider="google",
                    model="gemini-image-test",
                    count=2,
                )
            },
        ),
        _Secrets(),
    )

    assert clients["default"].endpoint_id == "openai_images"
    assert clients["design"].endpoint_id == "gemini_image_generation"
    assert clients["design"].default_count == 2


def test_image_factory_respects_disabled_operation() -> None:
    clients = build_image_generation_clients(
        ImageGenerationSettings(enabled=False),
        _Secrets(),
    )

    assert clients == {}


@pytest.mark.asyncio
async def test_image_service_selects_exact_profile_and_closes_alias_once() -> None:
    class _Client:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.close_count = 0

        async def generate_image(self, prompt: str, **kwargs):
            self.prompts.append(prompt)
            return kwargs

        async def aclose(self) -> None:
            self.close_count += 1

    client = _Client()
    service = ImageGenerationService(  # type: ignore[arg-type]
        {"default": client, "design": client}
    )

    result = await service.generate_image("A compass", profile="design", n=2)
    await service.aclose()

    assert result["n"] == 2
    assert client.prompts == ["A compass"]
    assert client.close_count == 1


@pytest.mark.asyncio
async def test_chat_facade_uses_explicit_image_assignment_without_provider_inference() -> None:
    class _ImageClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []
            self.close_count = 0

        async def generate_image(self, prompt: str, **kwargs) -> ImageGenerationResult:
            self.calls.append((prompt, kwargs))
            return ImageGenerationResult(images=[], usage={})

        async def aclose(self) -> None:
            self.close_count += 1

    chat_client = GenericLLMClient(provider="anthropic", model="claude-test", api_key="test")
    llm_service = LLMService({"default": chat_client})
    image_client = _ImageClient()

    with pytest.raises(LLMUnsupportedFeatureError, match="disabled or no default"):
        await chat_client.generate_image("Before assignment")

    llm_service.bind_image_service(
        ImageGenerationService({"default": image_client})  # type: ignore[dict-item]
    )
    result = await chat_client.generate_image(
        "After assignment",
        model="image-model",
        n=2,
    )
    await llm_service.aclose()

    assert result.images == []
    assert image_client.calls == [
        (
            "After assignment",
            {
                "model": "image-model",
                "n": 2,
                "size": None,
                "quality": None,
                "style": None,
                "output_format": None,
                "response_format": None,
                "background": None,
                "input_images": None,
                "azure_api_version": None,
            },
        )
    ]
    assert image_client.close_count == 1

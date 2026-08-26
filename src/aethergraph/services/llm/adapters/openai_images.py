"""Physical OpenAI Images adapter."""

from __future__ import annotations

from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.types import GeneratedImage, ImageGenerationResult
from aethergraph.services.llm.utils import (
    _guess_mime_from_format,
    _normalize_base_url_no_trailing_slash,
)


class OpenAIImagesAdapter:
    """Physical adapter for the OpenAI Images endpoint."""

    @staticmethod
    async def invoke(
        host: Any,
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
        **kw: Any,
    ) -> ProviderCallResult[ImageGenerationResult]:
        """Generate images through the OpenAI Images endpoint.

        Intro:
            Projects the existing image-generation facade arguments into one
            physical OpenAI request and normalizes returned image references.

        Examples:
            Generate one default image:
                ```python
                result = await OpenAIImagesAdapter.invoke(
                    client,
                    "A quiet observatory",
                    model="gpt-image-2",
                    n=1,
                    size=None,
                    quality=None,
                    style=None,
                    output_format=None,
                    response_format=None,
                    background=None,
                )
                ```

            Generate a transparent PNG:
                ```python
                result = await OpenAIImagesAdapter.invoke(
                    client,
                    "A glass compass",
                    model="gpt-image-2",
                    n=1,
                    size="1024x1024",
                    quality="high",
                    style=None,
                    output_format="png",
                    response_format="b64_json",
                    background="transparent",
                )
                ```

        Args:
            host: Bound generic client owning the OpenAI transport.
            prompt: Text description of the requested image.
            model: Exact configured image model identity.
            n: Number of images requested.
            size: Optional provider image dimensions.
            quality: Optional provider quality mode.
            style: Optional provider style mode.
            output_format: Optional encoded image format.
            response_format: Optional response transport format.
            background: Optional background mode.
            **kw: Additional compatibility arguments reserved by the facade.

        Returns:
            ProviderCallResult[ImageGenerationResult]: Normalized images, usage,
                raw provider data, and sanitized transport metadata.

        Notes:
            Retry, metering, and client lifecycle remain owned by the shared
            image-generation facade. The adapter performs one physical attempt.
        """

        assert host._client is not None

        url = f"{_normalize_base_url_no_trailing_slash(host.base_url)}/images/generations"
        headers = {"Authorization": f"Bearer {host.api_key}", "Content-Type": "application/json"}
        body: dict[str, Any] = {"model": model, "prompt": prompt, "n": n}
        if size is not None:
            body["size"] = size
        if quality is not None:
            body["quality"] = quality
        if style is not None:
            body["style"] = style
        if output_format is not None:
            body["output_format"] = output_format
        if background is not None:
            body["background"] = background
        if response_format is not None:
            body["response_format"] = response_format

        response = await host._client.post(url, headers=headers, json=body)
        metadata = checked_response_metadata("openai", model, "image", response)
        data = response.json()
        images = [
            GeneratedImage(
                b64=item.get("b64_json"),
                url=item.get("url"),
                mime_type=(
                    _guess_mime_from_format(output_format or "png")
                    if item.get("b64_json")
                    else None
                ),
                revised_prompt=item.get("revised_prompt"),
            )
            for item in data.get("data", []) or []
        ]
        return ProviderCallResult(
            ImageGenerationResult(images=images, usage=data.get("usage", {}) or {}, raw=data),
            metadata,
        )

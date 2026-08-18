"""Physical Azure OpenAI Images adapter."""

from __future__ import annotations

from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.types import GeneratedImage, ImageGenerationResult
from aethergraph.services.llm.utils import (
    _azure_images_generations_url,
    _guess_mime_from_format,
)


class AzureImagesAdapter:
    """Physical adapter for the Azure OpenAI Images endpoint."""

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
        azure_api_version: str | None,
        **kw: Any,
    ) -> ProviderCallResult[ImageGenerationResult]:
        """Generate images through the Azure OpenAI Images endpoint.

        Intro:
            Projects the existing image facade into Azure's deployment-scoped
            Images request and normalizes returned image references.

        Examples:
            Generate one image:
                ```python
                result = await AzureImagesAdapter.invoke(
                    client,
                    "A quiet observatory",
                    model="image-deployment",
                    n=1,
                    size=None,
                    quality=None,
                    style=None,
                    output_format=None,
                    response_format=None,
                    background=None,
                    azure_api_version=None,
                )
                ```

            Generate a transparent PNG:
                ```python
                result = await AzureImagesAdapter.invoke(
                    client,
                    "A glass compass",
                    model="image-deployment",
                    n=1,
                    size="1024x1024",
                    quality="high",
                    style=None,
                    output_format="png",
                    response_format="b64_json",
                    background="transparent",
                    azure_api_version="2025-04-01-preview",
                )
                ```

        Args:
            host: Bound generic client owning the Azure transport.
            prompt: Text description of the requested image.
            model: Exact configured image deployment identity.
            n: Number of images requested.
            size: Optional provider image dimensions.
            quality: Optional provider quality mode.
            style: Optional provider style mode.
            output_format: Optional encoded image format.
            response_format: Optional response transport format.
            background: Optional background mode.
            azure_api_version: Optional Azure Images API version.
            **kw: Additional compatibility arguments reserved by the facade.

        Returns:
            ProviderCallResult[ImageGenerationResult]: Normalized images, usage,
                raw provider data, and sanitized transport metadata.

        Notes:
            Retry, metering, and client lifecycle remain owned by the shared
            image-generation facade. The adapter performs one physical attempt.
        """

        assert host._client is not None
        if not host.base_url or not host.azure_deployment:
            raise RuntimeError(
                "Azure generate_image requires base_url=<resource endpoint> and "
                "azure_deployment=<deployment name>"
            )

        api_version = azure_api_version or "2025-04-01-preview"
        url = _azure_images_generations_url(host.base_url, host.azure_deployment, api_version)
        headers = {"api-key": host.api_key, "Content-Type": "application/json"}
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

        response = await host._client.post(url, headers=headers, json=body)
        metadata = checked_response_metadata("azure", model, "image", response)
        data = response.json()
        images = [
            GeneratedImage(
                b64=item.get("b64_json"),
                url=item.get("url"),
                mime_type=(
                    _guess_mime_from_format((output_format or "png").lower())
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

"""Physical Gemini image-generation adapter."""

from __future__ import annotations

from typing import Any

from aethergraph.services.llm.provider_transport import (
    ProviderCallResult,
    checked_response_metadata,
)
from aethergraph.services.llm.types import GeneratedImage, ImageGenerationResult
from aethergraph.services.llm.utils import (
    _data_url_to_b64_and_mime,
    _is_data_url,
    _normalize_base_url_no_trailing_slash,
)


class GeminiImagesAdapter:
    """Physical adapter for Gemini image generation."""

    @staticmethod
    async def invoke(
        host: Any,
        prompt: str,
        *,
        model: str,
        input_images: list[str] | None,
        **kw: Any,
    ) -> ProviderCallResult[ImageGenerationResult]:
        """Generate images through Gemini GenerateContent.

        Intro:
            Projects the existing image facade into Gemini text and inline-data
            parts and normalizes generated inline images.

        Examples:
            Generate an image from text:
                ```python
                result = await GeminiImagesAdapter.invoke(
                    client,
                    "A quiet observatory",
                    model="gemini-image-test",
                    input_images=None,
                )
                ```

            Edit an inline source image:
                ```python
                result = await GeminiImagesAdapter.invoke(
                    client,
                    "Make the sky violet",
                    model="gemini-image-test",
                    input_images=["data:image/png;base64,aW1hZ2U="],
                )
                ```

        Args:
            host: Bound generic client owning the Gemini transport.
            prompt: Text description of the requested image.
            model: Exact configured Gemini image model identity.
            input_images: Optional base64 data URLs for image-conditioned generation.
            **kw: Additional compatibility arguments reserved by the facade.

        Returns:
            ProviderCallResult[ImageGenerationResult]: Normalized images, usage,
                raw provider data, and sanitized transport metadata.

        Notes:
            Retry, metering, and client lifecycle remain owned by the shared
            image-generation facade. The adapter performs one physical attempt.
        """

        assert host._client is not None
        base = (
            _normalize_base_url_no_trailing_slash(host.base_url)
            or "https://generativelanguage.googleapis.com"
        )
        url = f"{base}/v1beta/models/{model}:generateContent"

        parts: list[dict[str, Any]] = []
        if input_images:
            for image in input_images:
                if not _is_data_url(image):
                    raise ValueError("Gemini input_images must be data: URLs (base64) for now.")
                b64, mime = _data_url_to_b64_and_mime(image)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
        parts.append({"text": prompt})

        response = await host._client.post(
            url,
            headers={"x-goog-api-key": host.api_key, "Content-Type": "application/json"},
            json={"contents": [{"parts": parts}]},
        )
        metadata = checked_response_metadata("google", model, "image", response)
        data = response.json()
        candidate = (data.get("candidates") or [{}])[0]
        output_parts = (candidate.get("content") or {}).get("parts") or []
        images: list[GeneratedImage] = []
        for part in output_parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                mime = inline.get("mimeType") or inline.get("mime_type")
                images.append(GeneratedImage(b64=inline["data"], mime_type=mime))

        usage_metadata = data.get("usageMetadata") or {}
        usage = {
            "input_tokens": int(usage_metadata.get("promptTokenCount", 0) or 0),
            "output_tokens": int(usage_metadata.get("candidatesTokenCount", 0) or 0),
        }
        return ProviderCallResult(
            ImageGenerationResult(images=images, usage=usage, raw=data),
            metadata,
        )

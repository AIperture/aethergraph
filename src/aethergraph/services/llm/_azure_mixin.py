"""Azure OpenAI methods (chat completions + image generation)."""

from __future__ import annotations

import json
from typing import Any

import httpx

from aethergraph.services.llm.types import (
    ChatOutputFormat,
    GeneratedImage,
    ImageGenerationResult,
)
from aethergraph.services.llm.utils import (
    _azure_images_generations_url,
    _ensure_system_json_directive,
    _guess_mime_from_format,
)


def _first_text(choices):
    """Extract text and usage from OpenAI-style choices list."""
    if not choices:
        return "", {}
    c = choices[0]
    text = (c.get("message", {}) or {}).get("content") or c.get("text") or ""
    usage = {}
    return text, usage


class _AzureMixin:
    """Provider methods for Azure OpenAI."""

    async def _chat_azure_chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        if not (self.base_url and self.azure_deployment):
            raise RuntimeError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT"
            )

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        msg_for_provider = messages
        payload: dict[str, Any] = {
            "messages": msg_for_provider,
            "temperature": temperature,
            "top_p": top_p,
        }

        if output_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
            payload["messages"] = _ensure_system_json_directive(messages, schema=None)
        elif output_format == "json_schema":
            if fail_on_unsupported:
                raise RuntimeError(
                    "Azure native json_schema not guaranteed; set fail_on_unsupported=False for best-effort"
                )
            payload["messages"] = _ensure_system_json_directive(messages, schema=json_schema)

        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        async def _call():
            r = await self._client.post(
                f"{self.base_url}/openai/deployments/{self.azure_deployment}/chat/completions?api-version=2024-08-01-preview",
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPError as e:
                raise RuntimeError(f"Azure chat/completions error: {e.response.text}") from e

            data = r.json()
            usage = data.get("usage", {}) or {}

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return txt, usage

            txt, _ = _first_text(data.get("choices", []))
            return txt, usage

        return await self._retry.run(_call)

    async def _image_azure_generate(
        self,
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
    ) -> ImageGenerationResult:
        assert self._client is not None

        if not self.base_url or not self.azure_deployment:
            raise RuntimeError(
                "Azure generate_image requires base_url=<resource endpoint> and azure_deployment=<deployment name>"
            )

        api_version = azure_api_version or "2025-04-01-preview"
        url = _azure_images_generations_url(self.base_url, self.azure_deployment, api_version)

        headers = {"api-key": self.api_key, "Content-Type": "application/json"}

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

        async def _call():
            r = await self._client.post(url, headers=headers, json=body)
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Azure image generation error: {r.text}") from e

            data = r.json()
            imgs: list[GeneratedImage] = []
            for item in data.get("data", []) or []:
                imgs.append(
                    GeneratedImage(
                        b64=item.get("b64_json"),
                        url=item.get("url"),
                        mime_type=_guess_mime_from_format((output_format or "png").lower())
                        if item.get("b64_json")
                        else None,
                        revised_prompt=item.get("revised_prompt"),
                    )
                )

            return ImageGenerationResult(images=imgs, usage=data.get("usage", {}) or {}, raw=data)

        return await self._retry.run(_call)

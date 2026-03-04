"""Google Gemini methods (chat + image generation)."""

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
    _data_url_to_b64_and_mime,
    _is_data_url,
    _normalize_base_url_no_trailing_slash,
    _to_gemini_parts,
)


class _GeminiMixin:
    """Provider methods for Google Gemini."""

    async def _chat_gemini_generate_content(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        output_format: ChatOutputFormat,
        json_schema: dict[str, Any] | None,
        fail_on_unsupported: bool,
        tools: list[dict[str, Any]] | None = None,
        **kw: Any,
    ) -> tuple[str, dict[str, int]]:
        await self._ensure_client()
        assert self._client is not None

        temperature = kw.get("temperature", 0.5)
        top_p = kw.get("top_p", 1.0)

        if tools is not None and fail_on_unsupported:
            raise RuntimeError("Gemini tools/function calling not wired yet in this client")

        # Merge system messages into preamble
        system_parts: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                c = m.get("content")
                system_parts.append(c if isinstance(c, str) else str(c))
        system = "\n".join(system_parts)

        turns: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "user" if m.get("role") == "user" else "model"
            parts = _to_gemini_parts(m.get("content"))
            turns.append({"role": role, "parts": parts})

        if system:
            turns.insert(0, {"role": "user", "parts": [{"text": f"System instructions: {system}"}]})

        async def _call():
            gen_cfg: dict[str, Any] = {"temperature": temperature, "topP": top_p}

            # Gemini native structured outputs
            if output_format == "json_object":
                gen_cfg["responseMimeType"] = "application/json"
            elif output_format == "json_schema":
                if json_schema is None:
                    raise ValueError("output_format='json_schema' requires json_schema")
                gen_cfg["responseMimeType"] = "application/json"
                gen_cfg["responseJsonSchema"] = json_schema

            payload = {"contents": turns, "generationConfig": gen_cfg}

            r = await self._client.post(
                f"{self.base_url}/v1/models/{model}:generateContent?key={self.api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Gemini generateContent failed ({e.response.status_code}): {e.response.text}"
                ) from e

            data = r.json()
            um = data.get("usageMetadata") or {}
            usage = {
                "input_tokens": int(um.get("promptTokenCount", 0) or 0),
                "output_tokens": int(um.get("candidatesTokenCount", 0) or 0),
            }

            if output_format == "raw":
                txt = json.dumps(data, ensure_ascii=False)
                return txt, usage

            cand = (data.get("candidates") or [{}])[0]
            txt = "".join(p.get("text", "") for p in (cand.get("content", {}).get("parts") or []))
            return txt, usage

        return await self._retry.run(_call)

    async def _image_gemini_generate(
        self,
        prompt: str,
        *,
        model: str,
        input_images: list[str] | None,
        **kw: Any,
    ) -> ImageGenerationResult:
        assert self._client is not None

        base = (
            _normalize_base_url_no_trailing_slash(self.base_url)
            or "https://generativelanguage.googleapis.com"
        )
        url = f"{base}/v1beta/models/{model}:generateContent"

        parts: list[dict[str, Any]] = []
        if input_images:
            for img in input_images:
                if not _is_data_url(img):
                    raise ValueError("Gemini input_images must be data: URLs (base64) for now.")
                b64, mime = _data_url_to_b64_and_mime(img)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})

        parts.append({"text": prompt})

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
        }

        async def _call():
            r = await self._client.post(
                url,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            try:
                r.raise_for_status()
            except Exception as e:
                raise RuntimeError(f"Gemini image generation error: {r.text}") from e

            data = r.json()
            cand = (data.get("candidates") or [{}])[0]
            content = cand.get("content") or {}
            out_parts = content.get("parts") or []

            imgs: list[GeneratedImage] = []
            for p in out_parts:
                inline = p.get("inlineData") or p.get("inline_data")
                if inline and inline.get("data"):
                    mime = inline.get("mimeType") or inline.get("mime_type")
                    imgs.append(GeneratedImage(b64=inline["data"], mime_type=mime))

            um = data.get("usageMetadata") or {}
            usage = {
                "input_tokens": int(um.get("promptTokenCount", 0) or 0),
                "output_tokens": int(um.get("candidatesTokenCount", 0) or 0),
            }

            return ImageGenerationResult(images=imgs, usage=usage, raw=data)

        return await self._retry.run(_call)

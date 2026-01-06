import base64
from collections.abc import Sequence
import json
import re
from typing import Any, Literal

from aethergraph.services.llm.types import ImageInput

ChatOutputFormat = Literal["text", "json_object", "json_schema"]


def _is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:") and ";base64," in s


def _image_bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _data_url_to_b64_and_mime(url: str) -> tuple[str, str]:
    head, b64 = url.split(",", 1)
    mime = head.split(";")[0].split(":", 1)[1]
    return b64, mime


def _ensure_b64(img: ImageInput) -> tuple[str, str]:
    if img.b64 and img.mime_type:
        return img.b64, img.mime_type
    if img.url and _is_data_url(img.url):
        return _data_url_to_b64_and_mime(img.url)
    if img.data and img.mime_type:
        import base64

        return base64.b64encode(img.data).decode("ascii"), img.mime_type
    raise ValueError("ImageInput must have (b64+mime_type) or (data+mime_type) or a data: URL")


def _normalize_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Normalize many common message shapes into:
      {"role": "...", "parts": [{"type":"text","text":...} | {"type":"image","image": ImageInput}]}
    Supports:
      - {"role","content":"text"}
      - ChatCompletions multimodal image_url parts
      - Responses input_image parts
      - Anthropic image source blocks
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content")
        parts: list[dict[str, Any]] = []

        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")

                if t in ("text", "input_text", "output_text"):
                    parts.append({"type": "text", "text": p.get("text", "")})
                    continue

                if t == "image_url":
                    iu = p.get("image_url") or {}
                    url = iu.get("url") or p.get("url")
                    if isinstance(url, str):
                        parts.append({"type": "image", "image": ImageInput(url=url)})
                    continue

                if t == "input_image":
                    url = p.get("image_url")
                    if isinstance(url, str):
                        parts.append({"type": "image", "image": ImageInput(url=url)})
                    continue

                if t == "image":
                    src = p.get("source") or {}
                    if src.get("type") == "base64":
                        parts.append(
                            {
                                "type": "image",
                                "image": ImageInput(
                                    b64=src.get("data"), mime_type=src.get("media_type")
                                ),
                            }
                        )
                    elif src.get("type") == "url":
                        parts.append({"type": "image", "image": ImageInput(url=src.get("url"))})
                    continue

        out.append({"role": role, "parts": parts})
    return out


def _has_images(norm: Sequence[dict[str, Any]]) -> bool:
    return any(p.get("type") == "image" for m in norm for p in m.get("parts", []))


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_text(text: str) -> str:
    t = _strip_code_fences(text)
    if not t:
        return t
    if t[0] in "{[":
        return t
    m = re.search(r"(\{.*\}|\[.*\])", t, flags=re.DOTALL)
    return m.group(1).strip() if m else t


def _validate_json_schema(obj: Any, schema: dict[str, Any]) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception:
        return
    jsonschema.validate(instance=obj, schema=schema)


def _ensure_system_json_directive(
    messages: list[dict[str, Any]], *, schema: dict[str, Any] | None
) -> list[dict[str, Any]]:
    directive = "Return ONLY valid JSON. No markdown, no commentary."
    if schema is not None:
        directive += "\nThe JSON MUST conform to this JSON Schema:\n" + json.dumps(
            schema, ensure_ascii=False
        )
    return [{"role": "system", "content": directive}] + list(messages)


def _message_content_has_images(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image_url", "input_image", "image"):
                    return True
    return False


def _to_anthropic_blocks(content: Any) -> list[dict[str, Any]]:
    """
    Accept:
      - str -> [{"type":"text","text":...}]
      - OpenAI multimodal list -> convert data URLs -> anthropic base64 image blocks
      - Anthropic blocks -> passthrough
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text", "output_text"):
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif t == "image" and "source" in p:
                blocks.append(p)
            elif t in ("image_url", "input_image"):
                url = None
                if t == "image_url":
                    url = (p.get("image_url") or {}).get("url") or p.get("url")
                else:
                    url = p.get("image_url")
                if isinstance(url, str) and _is_data_url(url):
                    b64, mime = _data_url_to_b64_and_mime(url)
                    blocks.append(
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": b64},
                        }
                    )
                elif isinstance(url, str):
                    raise RuntimeError(
                        "Anthropic vision: provide data: URLs (base64) for images (no remote fetch in client)."
                    )
        return blocks

    return [{"type": "text", "text": str(content)}]


def _to_gemini_parts(content: Any) -> list[dict[str, Any]]:
    """
    Gemini REST generateContent supports:
      - {"text": "..."}
      - {"inline_data": {"mime_type": "...", "data": "<base64>"}}
    """
    if isinstance(content, str):
        return [{"text": content}]

    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text", "output_text"):
                parts.append({"text": p.get("text", "")})
            elif t in ("image_url", "input_image"):
                url = None
                if t == "image_url":
                    url = (p.get("image_url") or {}).get("url") or p.get("url")
                else:
                    url = p.get("image_url")
                if isinstance(url, str) and _is_data_url(url):
                    b64, mime = _data_url_to_b64_and_mime(url)
                    parts.append({"inline_data": {"mime_type": mime, "data": b64}})
                elif isinstance(url, str):
                    raise RuntimeError(
                        "Gemini vision: provide data: URLs (base64) or file_uri; remote http(s) URLs not accepted inline."
                    )
        return parts

    return [{"text": str(content)}]


def _normalize_openai_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Make OpenAI Responses input robust:
      - If content is str: keep as-is
      - If content is OpenAI chat multimodal parts (text/image_url): convert to input_text/input_image
      - If already input_text/input_image: passthrough
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        c = m.get("content")

        if isinstance(c, str):
            out.append({"role": role, "content": c})
            continue

        if isinstance(c, list):
            blocks: list[dict[str, Any]] = []
            for p in c:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t in ("input_text", "input_image"):
                    blocks.append(p)
                elif t in ("text", "output_text"):
                    blocks.append({"type": "input_text", "text": p.get("text", "")})
                elif t == "image_url":
                    url = (p.get("image_url") or {}).get("url") or p.get("url")
                    if isinstance(url, str):
                        blocks.append({"type": "input_image", "image_url": url})
                else:
                    # ignore unknown part types
                    pass
            out.append({"role": role, "content": blocks})
            continue

        out.append({"role": role, "content": str(c)})
    return out


def _normalize_base_url_no_trailing_slash(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _azure_images_generations_url(endpoint: str, deployment: str, api_version: str) -> str:
    # endpoint example: https://<resource>.openai.azure.com
    ep = _normalize_base_url_no_trailing_slash(endpoint)
    return f"{ep}/openai/deployments/{deployment}/images/generations?api-version={api_version}"


def _guess_mime_from_format(fmt: str) -> str:
    if fmt == "png":
        return "image/png"
    if fmt == "jpeg":
        return "image/jpeg"
    if fmt == "webp":
        return "image/webp"
    return "application/octet-stream"

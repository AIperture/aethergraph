from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

_DATA_URL_RE = re.compile(r"data:([^;,]+)(?:;[^,]*)?,[^\s\"']+")


def sanitize_content(value: Any) -> Any:
    """Normalize values for bounded persistence and redact embedded data URLs.

    This policy intentionally does not claim general secret or PII detection. It
    removes inline data payloads and replaces raw binary values with stable metadata.
    """

    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _DATA_URL_RE.sub(lambda match: f"[redacted data URL: {match.group(1)}]", value)
    if isinstance(value, bytes):
        return {
            "binary_bytes": len(value),
            "sha256": sha256(value).hexdigest(),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): sanitize_content(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_content(item) for item in value]
    if isinstance(value, set):
        normalized = [sanitize_content(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if hasattr(value, "model_dump"):
        return sanitize_content(value.model_dump())
    if hasattr(value, "dict"):
        return sanitize_content(value.dict())
    return repr(value)


def sanitize_text(value: str) -> str:
    sanitized = sanitize_content(value)
    if not isinstance(sanitized, str):
        raise TypeError("Text sanitization must produce a string")
    return sanitized


def canonical_json(value: Any) -> str:
    return json.dumps(
        sanitize_content(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

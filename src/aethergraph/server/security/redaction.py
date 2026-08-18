from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

_DATA_URL_RE = re.compile(r"data:([^;,]+)(?:;[^,]*)?,[^\s\"']+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+"
)
_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "access-token",
    "access_token",
    "auth-token",
    "auth_token",
    "bot-token",
    "bot_token",
    "password",
    "secret",
    "token",
}
MASKED_SECRET_SENTINEL = "****"
REDACTED_CREDENTIAL = "[redacted credential]"


def _sensitive_key(key: Any) -> bool:
    return str(key).strip().lower() in _SENSITIVE_KEYS


def sanitize_content(value: Any) -> Any:
    """Normalize persisted observations and remove embedded sensitive material.

    Intro:
        Applies the single server-owned persistence policy to nested values,
        credentials, inline data URLs, binary payloads, and model objects.

    Examples:
        Redact a credential field:
        ```python
        assert sanitize_content({"api_key": "secret"}) == {
            "api_key": "[redacted credential]"
        }
        ```

        Replace binary payloads with bounded metadata:
        ```python
        result = sanitize_content(b"payload")
        assert result["binary_bytes"] == 7
        ```

    Args:
        value: Arbitrary observation or diagnostic value.

    Returns:
        Any: JSON-compatible sanitized content or a stable representation.

    Notes:
        Credential redaction is key-aware for mappings and pattern-aware for
        strings. Returned values must still be bounded by the calling store.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        sanitized = _DATA_URL_RE.sub(lambda match: f"[redacted data URL: {match.group(1)}]", value)
        sanitized = _BEARER_RE.sub("Bearer [redacted credential]", sanitized)
        return _ASSIGNMENT_RE.sub(
            lambda match: f"{match.group(1)}={REDACTED_CREDENTIAL}", sanitized
        )
    if isinstance(value, bytes):
        return {"binary_bytes": len(value), "sha256": sha256(value).hexdigest()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): (
                REDACTED_CREDENTIAL
                if _sensitive_key(key) and item is not None
                else sanitize_content(item)
            )
            for key, item in value.items()
        }
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
    """Redact one text value through the canonical persistence policy.

    Intro:
        Applies the same string rules used for structured observation content.

    Examples:
        Redact a bearer token:
        ```python
        text = sanitize_text("Bearer token-value")
        assert "token-value" not in text
        ```

        Redact an inline data URL:
        ```python
        text = sanitize_text("data:text/plain;base64,c2VjcmV0")
        assert "c2VjcmV0" not in text
        ```

    Args:
        value: Raw text intended for persistence or diagnostics.

    Returns:
        str: Sanitized text.

    Notes:
        A non-string sanitizer result is treated as a programming error.
    """
    sanitized = sanitize_content(value)
    if not isinstance(sanitized, str):
        raise TypeError("Text sanitization must produce a string")
    return sanitized


def canonical_json(value: Any) -> str:
    """Serialize one value after canonical persistence redaction.

    Intro:
        Produces deterministic compact JSON for hashes, comparisons, and stores.

    Examples:
        Serialize a mapping deterministically:
        ```python
        assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
        ```

        Redact before serialization:
        ```python
        assert "secret" not in canonical_json({"token": "secret"})
        ```

    Args:
        value: Arbitrary value accepted by `sanitize_content`.

    Returns:
        str: Sorted compact JSON containing only sanitized content.

    Notes:
        This is the canonical serialization boundary for observation persistence.
    """
    return json.dumps(
        sanitize_content(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def mask_secret(value: str | None) -> str | None:
    """Return a stable UI-safe partial mask for one credential.

    Intro:
        Preserves a short prefix and suffix for local settings recognition while
        preventing the full credential from reaching API responses.

    Examples:
        Mask a long credential:
        ```python
        assert mask_secret("abcdefghijk") == "abcd****hijk"
        ```

        Preserve an absent value:
        ```python
        assert mask_secret(None) is None
        ```

    Args:
        value: Plain credential value or `None`.

    Returns:
        str | None: Masked display value, the sentinel for short values, or `None`.

    Notes:
        Masked values are display-only and must never be used for authentication.
    """
    if not value:
        return None
    if len(value) <= 8:
        return MASKED_SECRET_SENTINEL
    return value[:4] + MASKED_SECRET_SENTINEL + value[-4:]


def is_masked_secret(value: str | None) -> bool:
    """Identify values produced by the canonical UI credential masker.

    Intro:
        Lets settings mutations distinguish a retained display mask from newly
        supplied credential material.

    Examples:
        Recognize a masked value:
        ```python
        assert is_masked_secret("abcd****hijk")
        ```

        Recognize a new value:
        ```python
        assert not is_masked_secret("new-credential")
        ```

    Args:
        value: Candidate settings payload value.

    Returns:
        bool: `True` for absent or canonically masked values.

    Notes:
        The predicate detects the exact sentinel embedded by `mask_secret`.
    """
    return not value or MASKED_SECRET_SENTINEL in value

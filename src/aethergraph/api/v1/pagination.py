from __future__ import annotations

import base64
from dataclasses import dataclass
import json

from fastapi import HTTPException  # type: ignore

# ---------------------------------------------------------------------------
# Legacy offset-based helpers (still used by existing endpoints)
# ---------------------------------------------------------------------------


def decode_cursor(cursor: str | None) -> int:
    """
    Turn an opaque cursor string into an integer offset.

    Supports both legacy plain-integer cursors and the newer base64-JSON
    format (extracts the ``off`` field).
    """
    if not cursor:
        return 0

    # Try new format first
    info = _try_decode_b64_json(cursor)
    if info is not None:
        if "off" in info:
            return int(info["off"])
        # Keyset cursor passed to an offset-based endpoint – treat as page 0
        return 0

    # Legacy plain integer
    try:
        return int(cursor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid cursor") from e


def encode_cursor(offset: int) -> str:
    """Encode an integer offset as an opaque cursor string."""
    return _encode_b64_json({"v": 1, "off": offset})


# ---------------------------------------------------------------------------
# Keyset cursor helpers (for append-only tables with autoincrement id)
# ---------------------------------------------------------------------------


def encode_keyset_cursor(row_id: int) -> str:
    """Encode a keyset cursor pointing *after* the given row id."""
    return _encode_b64_json({"v": 1, "id": row_id})


def encode_keyset_before_cursor(row_id: int) -> str:
    """Encode a keyset cursor pointing *before* the given row id (backward pagination)."""
    return _encode_b64_json({"v": 1, "bid": row_id})


def encode_offset_cursor(offset: int) -> str:
    """Encode an offset cursor in the new base64-JSON format."""
    return _encode_b64_json({"v": 1, "off": offset})


# ---------------------------------------------------------------------------
# Unified decoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CursorInfo:
    """Decoded cursor payload."""

    kind: str  # "keyset" | "keyset_before" | "offset"
    value: int  # row_id (keyset/keyset_before) or offset (offset)


def decode_cursor_v2(cursor: str | None) -> CursorInfo | None:
    """
    Decode a cursor string into a :class:`CursorInfo`.

    Returns ``None`` when *cursor* is ``None`` or empty (first page).

    Handles four formats:
    1. Base64-JSON ``{"v":1,"id":<n>}``  → keyset (forward: id > n)
    2. Base64-JSON ``{"v":1,"bid":<n>}`` → keyset_before (backward: id < n)
    3. Base64-JSON ``{"v":1,"off":<n>}`` → offset
    4. Plain integer string (legacy)     → offset
    """
    if not cursor:
        return None

    info = _try_decode_b64_json(cursor)
    if info is not None:
        if "bid" in info:
            return CursorInfo(kind="keyset_before", value=int(info["bid"]))
        if "id" in info:
            return CursorInfo(kind="keyset", value=int(info["id"]))
        if "off" in info:
            return CursorInfo(kind="offset", value=int(info["off"]))

    # Legacy plain integer → offset
    try:
        return CursorInfo(kind="offset", value=int(cursor))
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid cursor") from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_b64_json(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _try_decode_b64_json(cursor: str) -> dict | None:
    try:
        # Re-pad base64
        padded = cursor + "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(padded)
        return json.loads(raw)
    except Exception:
        return None

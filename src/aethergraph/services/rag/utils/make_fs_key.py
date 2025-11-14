import base64
import hashlib
import re

# Windows forbidden characters and device names
_INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?\*\x00-\x1F]')
_RESERVED_WIN = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def make_fs_key(cid: str, max_len: int = 128) -> str:
    """
    Convert any logical corpus_id (may include ':', Unicode, etc.)
    into a portable filename segment: [a-zA-Z0-9._-] only, no trailing space/dot,
    not a reserved device name.
    """
    # 1) Keep a short human-friendly prefix if present (e.g., "proj", "sess", "run")
    if ":" in cid:
        prefix, rest = cid.split(":", 1)
    else:
        prefix, rest = "cid", cid

    # 2) Encode the rest to a compact, portable token (urlsafe base64 without padding)
    #    This avoids collisions from simple character replacement.
    token = base64.urlsafe_b64encode(rest.encode("utf-8")).decode("ascii").rstrip("=")

    # 3) Build candidate and sanitize any stray chars just in case
    key = f"{prefix}-{token}"
    key = _INVALID_CHARS_RE.sub("_", key).rstrip(" .")

    # 4) Avoid Windows reserved device names
    if key.upper() in _RESERVED_WIN:
        key = f"_{key}_"

    # 5) Enforce a reasonable max length (append a short hash if truncated)
    if len(key) > max_len:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        key = key[: max_len - 9] + "-" + h

    return key

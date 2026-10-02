"""Selected JSON helpers for the filesystem-free conformance provider."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import re

from aethergraph.storage.contracts import StorageIntegrityError


def plain(value):
    if is_dataclass(value):
        return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def select(value, path, match_key=None, match_value=None):
    if not isinstance(path, str) or not re.fullmatch(
        r"\$(?:\.[A-Za-z_][A-Za-z_0-9]*|\[\d+\])*", path
    ):
        raise ValueError("Invalid simple payload path")
    if (match_key is None) != (match_value is None) or (
        match_key is not None
        and (
            not isinstance(match_key, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", match_key)
            or not isinstance(match_value, str)
            or not match_value
        )
    ):
        raise ValueError("Array selection requires an exact simple key and string value")
    for key, index in re.findall(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[(\d+)\]", path):
        if key:
            value = value.get(key) if isinstance(value, Mapping) else None
        else:
            value = (
                value[int(index)]
                if isinstance(value, (list, tuple)) and int(index) < len(value)
                else None
            )
    if match_key is not None:
        rows = (
            [
                item
                for item in value
                if isinstance(item, Mapping) and item.get(match_key) == match_value
            ]
            if isinstance(value, (list, tuple))
            else []
        )
        if len(rows) > 1:
            raise StorageIntegrityError("Selected payload array identity is not unique")
        value = rows[0] if rows else None
    return value


def chunk(value, metadata, *, offset, limit, reason="not_retained"):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16384:
        raise ValueError("Invalid content chunk bounds")
    result = {
        **metadata,
        "offset": offset,
        "available": False,
        "text": "",
        "char_count": 0,
        "next_offset": offset,
        "has_more": False,
    }
    if value is None:
        return {**result, "unavailable_reason": reason}
    body = json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if offset > len(body):
        raise ValueError("Content offset is outside the selected section")
    text = body[offset : offset + limit]
    return {
        **result,
        "available": True,
        "text": text,
        "char_count": len(body),
        "next_offset": offset + len(text),
        "has_more": offset + len(text) < len(body),
        "content_revision": hashlib.sha256(body.encode()).hexdigest(),
    }

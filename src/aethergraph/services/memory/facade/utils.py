import hashlib
import json
import os
import re
import time
from typing import Any
import unicodedata

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_event_id(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = s.replace(" ", "-")
    s = _SAFE.sub("-", s)
    return s.strip("-") or "default"


def load_sticky(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_sticky(path: str, m: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

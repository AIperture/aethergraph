import hashlib
import json
import os
import re
import time
from typing import Any
import unicodedata

from aethergraph.contracts.services.memory import Event
from aethergraph.services.scope.scope import Scope, ScopeLevel

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_event_id(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def normalize_tags(tags: list[str] | None) -> list[str]:
    """
    Normalize tags by trimming whitespace, dropping empties, and de-duplicating
    while preserving first-seen order.
    """
    if not tags:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for tag in tags:
        if not tag:
            continue
        value = str(tag).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


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


def event_matches_level(evt: Event, scope: Scope | None, *, level: ScopeLevel) -> bool:
    """
    Generic filter for in-memory / persistence events.

    - "scope":  everything in this memory scope (timeline) – always True.
    - "session": events whose session_id matches current scope.session_id.
    - "run":    events whose run_id matches current scope.run_id.
    - "user":   events whose user/client matches current scope user.
    - "org":    events whose org_id matches current scope org.
    """
    if scope is None:
        # If we have no scope context, just return everything on the timeline.
        return True

    # scope-level: caller already used a scope-specific timeline_id
    if level == "scope":
        return True

    if level == "session":
        if not scope.session_id:
            return True  # nothing to constrain by; treat as global on this timeline
        return evt.session_id == scope.session_id

    if level == "run":
        if not scope.run_id:
            return True
        return evt.run_id == scope.run_id

    if level == "user":
        u = scope.user_id or scope.client_id
        if not u:
            return True
        # Support both user_id and client_id on events
        return (evt.user_id == u) or (evt.client_id == u)

    if level == "org":
        if not scope.org_id:
            return True
        return evt.org_id == scope.org_id

    # Fallback: be permissive
    return True

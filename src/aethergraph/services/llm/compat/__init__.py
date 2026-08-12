"""Legacy codecs and selectors retained at public compatibility boundaries."""

from .endpoint_selection import resolve_legacy_chat_adapter
from .profiles import chat_profile_from_legacy, embedding_profile_from_legacy

__all__ = [
    "chat_profile_from_legacy",
    "embedding_profile_from_legacy",
    "resolve_legacy_chat_adapter",
]

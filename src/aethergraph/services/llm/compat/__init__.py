"""Legacy codecs and selectors retained at public compatibility boundaries."""

from .endpoint_selection import resolve_legacy_chat_adapter
from .profiles import (
    chat_profile_from_legacy,
    embedding_profile_from_legacy,
    image_generation_profile_from_settings,
)

__all__ = [
    "chat_profile_from_legacy",
    "embedding_profile_from_legacy",
    "image_generation_profile_from_settings",
    "resolve_legacy_chat_adapter",
]

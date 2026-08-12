"""Legacy profile codecs retained at public configuration boundaries."""

from .profiles import chat_profile_from_legacy, embedding_profile_from_legacy

__all__ = [
    "chat_profile_from_legacy",
    "embedding_profile_from_legacy",
]

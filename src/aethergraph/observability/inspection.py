"""Provider-neutral identity and failures for canonical inspection."""

from __future__ import annotations

from dataclasses import dataclass


class ObservabilityUnavailableError(RuntimeError):
    """Signal that the canonical observation service is unavailable."""


class ObservabilityNotFoundError(LookupError):
    """Signal that a scoped canonical inspection record does not exist."""


class ObservabilityWorkspaceError(RuntimeError):
    """Signal that a manifested historical workspace cannot be opened."""


@dataclass(frozen=True)
class ObservabilityIdentity:
    """Authenticated identity applied to one canonical inspection reader."""

    mode: str = "local"
    user_id: str | None = None
    org_id: str | None = None


__all__ = [
    "ObservabilityIdentity",
    "ObservabilityNotFoundError",
    "ObservabilityUnavailableError",
    "ObservabilityWorkspaceError",
]

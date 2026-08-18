"""Stable public contracts for canonical Agent-state services."""

from __future__ import annotations

from typing import Literal

AgentStateBackend = Literal["hybrid", "memory", "local"]


class AgentStateConflictError(RuntimeError):
    """Report a stale enclosing Agent-state revision before commit."""

    def __init__(self, *, key: str, expected_revision: int, actual_revision: int) -> None:
        self.key = str(key)
        self.expected_revision = int(expected_revision)
        self.actual_revision = int(actual_revision)
        super().__init__(
            f"Agent state {self.key!r} revision changed: expected "
            f"{self.expected_revision}, actual {self.actual_revision}"
        )

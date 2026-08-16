"""Provider-neutral public memory failures."""

from __future__ import annotations


class StateSnapshotConflictError(RuntimeError):
    """Report a failed storage-level state snapshot revision comparison."""

    def __init__(self, *, key: str, expected_revision: int, actual_revision: int) -> None:
        self.key = str(key)
        self.expected_revision = int(expected_revision)
        self.actual_revision = int(actual_revision)
        super().__init__(
            f"State snapshot {self.key!r} revision changed: expected "
            f"{self.expected_revision}, actual {self.actual_revision}"
        )


__all__ = ["StateSnapshotConflictError"]

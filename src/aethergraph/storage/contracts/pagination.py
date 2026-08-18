"""Opaque cursor pagination shared by canonical storage protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")
MAX_STORAGE_PAGE_SIZE = 1_000


@dataclass(frozen=True, slots=True)
class PageRequest:
    """Bounded request for a stable provider-owned cursor page."""

    limit: int = 100
    cursor: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not 1 <= self.limit <= MAX_STORAGE_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_STORAGE_PAGE_SIZE}")
        if self.cursor is not None and (
            not isinstance(self.cursor, str) or not self.cursor.strip()
        ):
            raise ValueError("cursor must be a non-empty opaque string when supplied")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """Immutable page of records and an optional opaque continuation cursor."""

    items: tuple[T, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("items must be an immutable tuple")
        if self.next_cursor is not None and (
            not isinstance(self.next_cursor, str) or not self.next_cursor.strip()
        ):
            raise ValueError("next_cursor must be a non-empty opaque string when supplied")

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
    """Immutable records and provider-owned opaque continuation anchors.

    `next_cursor` means additional records existed at read time. A provider may
    supply `resume_cursor` at the current tail for later appends. `item_cursors`
    allow a response to stop after an item without losing the remaining page;
    they bind the same scope, filters and ordering as the page request.
    """

    items: tuple[T, ...]
    next_cursor: str | None = None
    resume_cursor: str | None = None
    item_cursors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise TypeError("items must be an immutable tuple")
        if self.next_cursor is not None and (
            not isinstance(self.next_cursor, str) or not self.next_cursor.strip()
        ):
            raise ValueError("next_cursor must be a non-empty opaque string when supplied")
        if self.resume_cursor is not None and (
            not isinstance(self.resume_cursor, str) or not self.resume_cursor.strip()
        ):
            raise ValueError("resume_cursor must be a non-empty opaque string when supplied")
        if not isinstance(self.item_cursors, tuple) or (
            self.item_cursors and len(self.item_cursors) != len(self.items)
        ):
            raise ValueError("item_cursors must be an immutable anchor tuple matching items")
        if any(not isinstance(value, str) or not value for value in self.item_cursors):
            raise ValueError("item_cursors must contain non-empty opaque strings")

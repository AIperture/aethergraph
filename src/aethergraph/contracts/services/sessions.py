from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from aethergraph.api.v1.schemas import Session
from aethergraph.core.runtime.run_types import SessionKind


class SessionStore(Protocol):
    async def create(
        self,
        *,
        kind: SessionKind,
        user_id: str | None = None,
        org_id: str | None = None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        """
        Create a new session and return it.
        """

    async def get(self, session_id: str) -> Session | None:
        """
        Get a session by its ID, or None if not found.
        """

    async def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Session]:
        """
        List sessions for a specific user, optionally filtered by kind.
        """

    async def touch(
        self,
        session_id: str,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        """
        Update session's updated_at (e.g., when new message/run occurs).
        No-op if session doesn't exist.
        """

    async def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        """
        Update session metadata, returning the updated session.
        No-op if session doesn't exist (returns None).
        """

    async def delete(self, session_id: str) -> None:
        """
        Delete a session by its ID.
        No-op if session doesn't exist.
        """

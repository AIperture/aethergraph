from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

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
    ) -> None:
        """
        Create a new session.
        """

    async def get(self, session_id: str) -> None:
        """
        Get a session by its ID.
        """

    async def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[None]:
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

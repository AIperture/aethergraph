from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from aethergraph.api.v1.schemas import Session
from aethergraph.core.runtime.run_types import SessionKind


class SessionStore(Protocol):
    async def create(
        self,
        *,
        session_id: str | None = None,
        kind: SessionKind,
        user_id: str | None = None,
        org_id: str | None = None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        """
        Create a session with an optional caller-owned identity.

        Repeating an exact explicit identity is idempotent. Reusing that
        identity with different ownership or source metadata must fail.
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
        title_source: Literal["manual", "auto"] | None = None,
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

    async def record_artifact(
        self,
        session_id: str,
        *,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one exact artifact occurrence for a session.

        The stable artifact identity makes retry behavior explicit while preserving
        the frozen no-op behavior for an absent session.

        Examples:
            Count an artifact:
                ```python
                await sessions.record_artifact("session-1", occurrence_id="occurrence-1")
                ```

            Retry the same occurrence:
                ```python
                await sessions.record_artifact("session-1", occurrence_id="occurrence-1")
                ```

        Args:
            session_id: Exact session identity to update.
            occurrence_id: Stable artifact occurrence idempotency identity.
            created_at: Optional artifact creation time; defaults to current UTC.

        Returns:
            None: The occurrence was counted, replayed, or its session was absent.

        Notes:
            Reusing one artifact identity with a different timestamp fails directly.
        """
        ...

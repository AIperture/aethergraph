import asyncio
from datetime import datetime, timezone
import uuid

from aethergraph.api.v1.schemas import Session
from aethergraph.core.runtime.run_types import SessionKind


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()  # TODO: confirm async lock is fine bc this will only be used inside uvicorn process with UI.

    async def create(
        self,
        *,
        kind: SessionKind,
        user_id: str | None,
        org_id: str | None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        async with self._lock:
            now = datetime.now(timezone.utc)
            session_id = f"sess_{uuid.uuid4().hex[:8]}"
            sess = Session(
                session_id=session_id,
                kind=kind,
                title=title,
                user_id=user_id,
                org_id=org_id,
                source=source,
                external_ref=external_ref,
                created_at=now,
                updated_at=now,
            )
            self._sessions[session_id] = sess
            return sess

    async def get(self, session_id: str) -> Session | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        async with self._lock:
            records = list(self._sessions.values())

            if user_id is not None:
                records = [s for s in records if s.user_id == user_id]
            if org_id is not None:
                records = [s for s in records if s.org_id == org_id]
            if kind is not None:
                records = [s for s in records if s.kind == kind]

            records.sort(key=lambda s: s.created_at, reverse=True)

            if offset:
                records = records[offset:]
            if limit:
                records = records[:limit]

            return records

    async def touch(
        self,
        session_id: str,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        async with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return
            sess.updated_at = updated_at or datetime.now(timezone.utc)

# aethergraph/storage/sqlite_session_store.py

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid

from aethergraph.api.v1.schemas import Session
from aethergraph.contracts.services.sessions import SessionStore
from aethergraph.core.runtime.run_types import SessionKind


def _dt_to_ts(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _parse_dt(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None
    if isinstance(val, int | float):
        try:
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        except Exception:
            return None
    return None


def _session_to_doc(sess: Session) -> dict[str, Any]:
    # Support both Pydantic v1 (.dict) and v2 (.model_dump)
    data = sess.model_dump() if hasattr(sess, "model_dump") else sess.dict()

    # Normalize datetimes to ISO for JSON
    for key in ("created_at", "updated_at", "last_artifact_at"):
        if isinstance(data.get(key), datetime):
            data[key] = data[key].isoformat()
    return data


def _doc_to_session(doc: dict[str, Any]) -> Session:
    # Convert ISO/ts back to datetime
    for key in ("created_at", "updated_at", "last_artifact_at"):
        if key in doc:
            parsed = _parse_dt(doc[key])
            if parsed is not None:
                doc[key] = parsed

    # Normalize kind if stored as str
    if "kind" in doc and isinstance(doc["kind"], str):
        try:
            doc["kind"] = SessionKind(doc["kind"])
        except Exception:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Unknown SessionKind stored in DB: {doc['kind']}")

    return Session(**doc)


class SQLiteSessionStoreSync:
    """
    SQLite-backed SessionStore.

    - Stores full Session as JSON in `data_json`
    - Promotes session_id, kind, user_id, org_id, created_at, updated_at,
      artifact_count, last_artifact_at to columns for fast listing / stats.
    """

    def __init__(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(
            str(path_obj),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")

        # Base table
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id        TEXT PRIMARY KEY,
                data_json         TEXT NOT NULL,
                kind              TEXT NOT NULL,
                user_id           TEXT,
                org_id            TEXT,
                created_at        REAL NOT NULL,
                updated_at        REAL NOT NULL,
                artifact_count    INTEGER NOT NULL DEFAULT 0,
                last_artifact_at  REAL
            )
            """
        )

        # Indices
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_org_updated ON sessions(org_id, updated_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_kind_updated ON sessions(kind, updated_at DESC)"
        )

        self._lock = threading.RLock()

    # -------- core helpers --------

    def _upsert(self, sess: Session) -> Session:
        doc = _session_to_doc(sess)
        payload = json.dumps(doc, ensure_ascii=False)

        created_ts = _dt_to_ts(sess.created_at)
        updated_ts = _dt_to_ts(sess.updated_at)
        last_art_ts = _dt_to_ts(sess.last_artifact_at)
        artifact_count = sess.artifact_count or 0

        kind_val = sess.kind.value if isinstance(sess.kind, SessionKind) else str(sess.kind)

        with self._lock:
            self._db.execute(
                """
                INSERT INTO sessions (
                    session_id, data_json,
                    kind, user_id, org_id,
                    created_at, updated_at,
                    artifact_count, last_artifact_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    data_json        = excluded.data_json,
                    kind             = excluded.kind,
                    user_id          = excluded.user_id,
                    org_id           = excluded.org_id,
                    created_at       = excluded.created_at,
                    updated_at       = excluded.updated_at,
                    artifact_count   = excluded.artifact_count,
                    last_artifact_at = excluded.last_artifact_at
                """,
                (
                    sess.session_id,
                    payload,
                    kind_val,
                    sess.user_id,
                    sess.org_id,
                    created_ts,
                    updated_ts,
                    artifact_count,
                    last_art_ts,
                ),
            )
        return sess

    # -------- SessionStore-style API (sync) --------

    def create(
        self,
        *,
        kind: SessionKind,
        user_id: str | None = None,
        org_id: str | None = None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        now = datetime.now(timezone.utc)
        sess = Session(
            session_id=str(uuid.uuid4()),
            kind=kind,
            title=title,
            user_id=user_id,
            org_id=org_id,
            source=source,
            external_ref=external_ref,
            created_at=now,
            updated_at=now,
            artifact_count=0,
            last_artifact_at=None,
        )
        return self._upsert(sess)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data_json FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        doc = json.loads(row[0])
        return _doc_to_session(doc)

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._db.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,),
            )

    def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        where: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if org_id is not None:
            where.append("org_id = ?")
            params.append(org_id)
        if kind is not None:
            where.append("kind = ?")
            params.append(kind.value if isinstance(kind, SessionKind) else str(kind))

        sql = "SELECT data_json FROM sessions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC"

        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        return [_doc_to_session(json.loads(r[0])) for r in rows]

    def touch(self, session_id: str, *, updated_at: datetime | None = None) -> None:
        sess = self.get(session_id)
        if not sess:
            return
        sess.updated_at = updated_at or datetime.now(timezone.utc)
        self._upsert(sess)

    def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        sess = self.get(session_id)
        if not sess:
            return None
        if title is not None:
            sess.title = title
        if external_ref is not None:
            sess.external_ref = external_ref
        sess.updated_at = datetime.now(timezone.utc)
        return self._upsert(sess)

    def record_artifact(
        self,
        session_id: str,
        *,
        created_at: datetime | None = None,
    ) -> None:
        """
        Optional API used by ArtifactFacade._record via getattr(..., 'record_artifact').
        Updates artifact_count + last_artifact_at + updated_at.
        """
        sess = self.get(session_id)
        if not sess:
            return

        ts = created_at or datetime.now(timezone.utc)

        sess.artifact_count = (sess.artifact_count or 0) + 1
        if sess.last_artifact_at is None or ts > sess.last_artifact_at:
            sess.last_artifact_at = ts

        # For UI, bump updated_at as well
        if ts > sess.updated_at:
            sess.updated_at = ts
        else:
            sess.updated_at = datetime.now(timezone.utc)

        self._upsert(sess)


class SQLiteSessionStore(SessionStore):
    """
    Async SessionStore implementation backed by SQLiteSessionStoreSync.
    """

    def __init__(self, path: str):
        self._sync = SQLiteSessionStoreSync(path)

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
        # Delegate to sync create (which already constructs Session correctly)
        return await asyncio.to_thread(
            self._sync.create,
            kind=kind,
            user_id=user_id,
            org_id=org_id,
            title=title,
            source=source,
            external_ref=external_ref,
        )

    async def get(self, session_id: str) -> Session | None:
        return await asyncio.to_thread(self._sync.get, session_id)

    async def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Session]:
        return await asyncio.to_thread(
            self._sync.list_for_user,
            user_id=user_id,
            org_id=org_id,
            kind=kind,
            limit=limit,
            offset=offset,
        )

    async def touch(
        self,
        session_id: str,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        await asyncio.to_thread(self._sync.touch, session_id, updated_at=updated_at)

    async def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        return await asyncio.to_thread(
            self._sync.update,
            session_id,
            title=title,
            external_ref=external_ref,
        )

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(self._sync.delete, session_id)

    async def record_artifact(
        self,
        session_id: str,
        *,
        created_at: datetime | None = None,
    ) -> None:
        """
        Optional method, called via getattr(..., 'record_artifact', None)
        from ArtifactFacade._record.
        """
        await asyncio.to_thread(
            self._sync.record_artifact,
            session_id,
            created_at=created_at,
        )

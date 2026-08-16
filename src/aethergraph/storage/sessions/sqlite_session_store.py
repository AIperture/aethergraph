# aethergraph/storage/sqlite_session_store.py

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime
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
        dt = dt.replace(tzinfo=UTC)
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
            return datetime.fromtimestamp(float(val), tz=UTC)
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
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS session_artifact_occurrences (
                session_id TEXT NOT NULL,
                occurrence_id TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                PRIMARY KEY(session_id, occurrence_id)
            )
            """
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
        session_id: str | None = None,
        kind: SessionKind,
        user_id: str | None = None,
        org_id: str | None = None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        with self._lock:
            if session_id is not None:
                existing = self.get(session_id)
                if existing is not None:
                    expected = (kind, user_id, org_id, source, external_ref)
                    actual = (
                        existing.kind,
                        existing.user_id,
                        existing.org_id,
                        existing.source,
                        existing.external_ref,
                    )
                    if actual != expected:
                        raise ValueError(f"Session identity collision: {session_id}")
                    return existing
            now = datetime.now(UTC)
            sess = Session(
                session_id=session_id or str(uuid.uuid4()),
                kind=kind,
                title=title,
                title_source="manual" if (title or "").strip() else None,
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
            self._db.execute("BEGIN IMMEDIATE")
            try:
                self._db.execute(
                    "DELETE FROM session_artifact_occurrences WHERE session_id = ?",
                    (session_id,),
                )
                self._db.execute(
                    "DELETE FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise

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
        sess.updated_at = updated_at or datetime.now(UTC)
        self._upsert(sess)

    def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        title_source: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        sess = self.get(session_id)
        if not sess:
            return None
        if title is not None:
            sess.title = title
            sess.title_source = title_source or "manual"
        if external_ref is not None:
            sess.external_ref = external_ref
        sess.updated_at = datetime.now(UTC)
        return self._upsert(sess)

    def record_artifact(
        self,
        session_id: str,
        *,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Atomically count one stable SQLite session artifact occurrence.

        Receipt insertion and the public session counters commit together under one
        immediate write transaction.

        Examples:
            Count an artifact:
                ```python
                sessions.record_artifact("session-1", occurrence_id="occurrence-1")
                ```

            Replay the receipt:
                ```python
                sessions.record_artifact("session-1", occurrence_id="occurrence-1")
                ```

        Args:
            session_id: Exact session identity to update.
            occurrence_id: Stable artifact occurrence identity.
            created_at: Optional artifact creation time; defaults to current UTC.

        Returns:
            None: The occurrence was counted, replayed, or its session was absent.

        Notes:
            Reusing an artifact identity with another timestamp raises `ValueError`.
        """
        if not isinstance(occurrence_id, str) or not occurrence_id.strip():
            raise ValueError("occurrence_id must be a non-empty string")
        occurred_at = created_at or datetime.now(UTC)
        occurred_ts = _dt_to_ts(occurred_at)
        assert occurred_ts is not None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT data_json FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return
                receipt = self._db.execute(
                    """
                    SELECT occurred_at FROM session_artifact_occurrences
                    WHERE session_id = ? AND occurrence_id = ?
                    """,
                    (session_id, occurrence_id),
                ).fetchone()
                if receipt is not None:
                    if float(receipt[0]) != occurred_ts:
                        raise ValueError("Session artifact occurrence identity conflicts")
                    self._db.execute("COMMIT")
                    return
                self._db.execute(
                    """
                    INSERT INTO session_artifact_occurrences(
                        session_id, occurrence_id, occurred_at
                    ) VALUES (?, ?, ?)
                    """,
                    (session_id, occurrence_id, occurred_ts),
                )
                session = _doc_to_session(json.loads(row[0]))
                session.artifact_count = (session.artifact_count or 0) + 1
                session.last_artifact_at = max(
                    occurred_at,
                    session.last_artifact_at or occurred_at,
                )
                session.updated_at = max(session.updated_at, occurred_at)
                self._upsert(session)
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise


class SQLiteSessionStore(SessionStore):
    """
    Async SessionStore implementation backed by SQLiteSessionStoreSync.
    """

    def __init__(self, path: str):
        self._sync = SQLiteSessionStoreSync(path)

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
        # Delegate to sync create (which already constructs Session correctly)
        return await asyncio.to_thread(
            self._sync.create,
            session_id=session_id,
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
        title_source: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        return await asyncio.to_thread(
            self._sync.update,
            session_id,
            title=title,
            title_source=title_source,
            external_ref=external_ref,
        )

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(self._sync.delete, session_id)

    async def record_artifact(
        self,
        session_id: str,
        *,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one stable SQLite session artifact occurrence asynchronously.

        The async boundary delegates the exact identity and timestamp to the
        transactional synchronous store.

        Examples:
            Count an artifact:
                ```python
                await sessions.record_artifact("session-1", occurrence_id="occurrence-1")
                ```

            Replay the receipt:
                ```python
                await sessions.record_artifact("session-1", occurrence_id="occurrence-1")
                ```

        Args:
            session_id: Exact session identity to update.
            occurrence_id: Stable artifact occurrence identity.
            created_at: Optional artifact creation time; defaults to current UTC.

        Returns:
            None: The occurrence was counted, replayed, or its session was absent.

        Notes:
            Collision and transaction errors propagate from the synchronous store.
        """
        await asyncio.to_thread(
            self._sync.record_artifact,
            session_id,
            occurrence_id=occurrence_id,
            created_at=created_at,
        )

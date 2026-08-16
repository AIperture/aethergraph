from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
import uuid

from aethergraph.api.v1.schemas import Session
from aethergraph.contracts.services.sessions import SessionStore
from aethergraph.contracts.storage.doc_store import (
    DocStore,  # wherever your DocStore Protocol lives
)
from aethergraph.core.runtime.run_types import SessionKind


def _encode_dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _decode_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _session_to_doc(s: Session) -> dict[str, Any]:
    d = s.model_dump() if hasattr(s, "model_dump") else dict(s)  # pydantic v2/v1 tolerant
    d["created_at"] = _encode_dt(getattr(s, "created_at", None))
    d["updated_at"] = _encode_dt(getattr(s, "updated_at", None))
    return d


def _doc_to_session(doc: dict[str, Any]) -> Session:
    doc = dict(doc)
    doc["created_at"] = _decode_dt(doc.get("created_at"))
    doc["updated_at"] = _decode_dt(doc.get("updated_at"))
    return Session(**doc)


class DocSessionStore(SessionStore):
    """
    SessionStore backed by an arbitrary DocStore.

    - Uses doc IDs like "<prefix><session_id>" (prefix defaults to "session:").
    - Persists Session as JSON-friendly dicts (ISO datetimes).
    - Supports FS-backed or SQLite-backed DocStore transparently.

    The only requirement is that the underlying DocStore implements `list()`
    if you want list_for_user() to work.
    """

    def __init__(self, doc_store: DocStore, *, prefix: str = "session:") -> None:
        self._ds = doc_store
        self._prefix = prefix
        self._lock = asyncio.Lock()

    def _doc_id(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def create(
        self,
        *,
        session_id: str | None = None,
        kind: SessionKind,
        user_id: str | None,
        org_id: str | None,
        title: str | None = None,
        source: str = "webui",
        external_ref: str | None = None,
    ) -> Session:
        now = datetime.now(UTC)
        resolved_session_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"
        sess = Session(
            session_id=resolved_session_id,
            kind=kind,
            title=title,
            title_source="manual" if (title or "").strip() else None,
            user_id=user_id,
            org_id=org_id,
            source=source,
            external_ref=external_ref,
            created_at=now,
            updated_at=now,
        )

        async with self._lock:
            existing_doc = await self._ds.get(self._doc_id(resolved_session_id))
            if existing_doc is not None:
                existing = _doc_to_session(existing_doc)
                expected = (kind, user_id, org_id, source, external_ref)
                actual = (
                    existing.kind,
                    existing.user_id,
                    existing.org_id,
                    existing.source,
                    existing.external_ref,
                )
                if actual != expected:
                    raise ValueError(f"Session identity collision: {resolved_session_id}")
                return existing
            await self._ds.put(self._doc_id(resolved_session_id), _session_to_doc(sess))
        return sess

    async def get(self, session_id: str) -> Session | None:
        doc_id = self._doc_id(session_id)
        async with self._lock:
            doc = await self._ds.get(doc_id)
        if not doc:
            return None
        return _doc_to_session(doc)

    async def list_for_user(
        self,
        *,
        user_id: str | None,
        org_id: str | None = None,
        kind: SessionKind | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Session]:
        # Same tradeoff as DocRunStore.list(): scan all, filter in Python
        if not hasattr(self._ds, "list"):
            raise RuntimeError(
                "Underlying DocStore does not implement list(); "
                "cannot support SessionStore.list_for_user()."
            )

        async with self._lock:
            doc_ids: list[str] = await self._ds.list()  # type: ignore[attr-defined]
            doc_ids = [d for d in doc_ids if d.startswith(self._prefix)]

            records: list[Session] = []
            for doc_id in doc_ids:
                doc = await self._ds.get(doc_id)
                if not doc:
                    continue
                sess = _doc_to_session(doc)

                if user_id is not None and sess.user_id != user_id:
                    continue
                if org_id is not None and sess.org_id != org_id:
                    continue
                if kind is not None and sess.kind != kind:
                    continue

                records.append(sess)

        records.sort(key=lambda s: s.created_at, reverse=True)

        if offset > 0:
            records = records[offset:]
        if limit is not None:
            records = records[:limit]
        return records

    async def touch(
        self,
        session_id: str,
        *,
        updated_at: datetime | None = None,
    ) -> None:
        doc_id = self._doc_id(session_id)
        async with self._lock:
            doc = await self._ds.get(doc_id)
            if doc is None:
                return
            doc["updated_at"] = _encode_dt(updated_at or datetime.now(UTC))
            await self._ds.put(doc_id, doc)

    async def update(
        self,
        session_id: str,
        *,
        title: str | None = None,
        title_source: str | None = None,
        external_ref: str | None = None,
    ) -> Session | None:
        doc_id = self._doc_id(session_id)
        async with self._lock:
            doc = await self._ds.get(doc_id)
            if doc is None:
                return None

            if title is not None:
                doc["title"] = title
                doc["title_source"] = title_source or "manual"
            if external_ref is not None:
                doc["external_ref"] = external_ref

            # Always bump updated_at
            doc["updated_at"] = _encode_dt(datetime.now(UTC))

            await self._ds.put(doc_id, doc)

        return _doc_to_session(doc)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            await self._ds.delete(self._doc_id(session_id))

    async def record_artifact(
        self,
        session_id: str,
        *,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one stable document-backed session artifact occurrence.

        Receipt state stays nested in the session document and is excluded from the
        public Session projection.

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
            Reusing an artifact identity with another timestamp raises `ValueError`.
        """
        if not isinstance(occurrence_id, str) or not occurrence_id.strip():
            raise ValueError("occurrence_id must be a non-empty string")
        occurred_at = created_at or datetime.now(UTC)
        occurred_text = _encode_dt(occurred_at)
        doc_id = self._doc_id(session_id)
        async with self._lock:
            doc = await self._ds.get(doc_id)
            if doc is None:
                return
            receipts = doc.setdefault("_artifact_occurrences", {})
            if not isinstance(receipts, dict):
                raise ValueError("Session artifact receipts are malformed")
            previous = receipts.get(occurrence_id)
            if previous is not None:
                if previous != occurred_text:
                    raise ValueError("Session artifact occurrence identity conflicts")
                return
            receipts[occurrence_id] = occurred_text
            doc["artifact_count"] = int(doc.get("artifact_count") or 0) + 1
            last_artifact_at = _decode_dt(doc.get("last_artifact_at"))
            doc["last_artifact_at"] = _encode_dt(max(occurred_at, last_artifact_at or occurred_at))
            updated_at = _decode_dt(doc.get("updated_at"))
            doc["updated_at"] = _encode_dt(max(occurred_at, updated_at or occurred_at))
            await self._ds.put(doc_id, doc)

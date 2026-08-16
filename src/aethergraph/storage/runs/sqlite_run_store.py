# aethergraph/storage/sqlite_run_store.py

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from aethergraph.contracts.services.runs import RunStore
from aethergraph.core.runtime.run_types import RunRecord, RunStatus


def _dt_to_ts(dt: datetime | None) -> float | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        # assume UTC if naive
        return dt.replace(tzinfo=UTC).timestamp()
    return dt.timestamp()


def _encode_run(record: RunRecord) -> dict[str, Any]:
    """Convert RunRecord -> plain dict with JSON-safe types."""
    if is_dataclass(record):  # noqa: SIM108
        data = asdict(record)
    else:
        # fallback; should not really happen
        data = dict(record.__dict__)

    for k, v in list(data.items()):
        if isinstance(v, datetime):
            data[k] = v.isoformat()
    return data


def _decode_run(data: dict[str, Any]) -> RunRecord:
    """Convert dict from JSON back into RunRecord."""

    # Best-effort datetime parsing for common fields
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
        return None

    for key in (
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "first_artifact_at",
        "last_artifact_at",
        "result_updated_at",
    ):
        if key in data:
            parsed = _parse_dt(data[key])
            if parsed is not None:
                data[key] = parsed

    return RunRecord(**data)


class SQLiteRunStoreSync:
    """
    SQLite-backed RunStore.

    - Stores full RunRecord as JSON in `data_json`
    - Promotes a few fields to columns for fast filtering:
        run_id, graph_id, status, user_id, org_id, session_id,
        started_at, finished_at
    """

    def __init__(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(
            str(path_obj),
            check_same_thread=False,
            isolation_level=None,  # autocommit
        )
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")

        # Base table
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id       TEXT PRIMARY KEY,
                data_json    TEXT NOT NULL,
                graph_id     TEXT,
                status       TEXT,
                user_id      TEXT,
                org_id       TEXT,
                session_id   TEXT,
                started_at   REAL,
                finished_at  REAL
            )
            """
        )

        # Indices for common queries
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_graph_started ON runs(graph_id, started_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_status_started ON runs(status, started_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_user_started ON runs(user_id, started_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_org_started ON runs(org_id, started_at DESC)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_session_started ON runs(session_id, started_at DESC)"
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS run_artifact_occurrences (
                run_id TEXT NOT NULL,
                occurrence_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                occurred_at REAL NOT NULL,
                PRIMARY KEY(run_id, occurrence_id)
            )
            """
        )

        self._lock = threading.RLock()

    # --- core ops ---

    def create(self, record: RunRecord) -> None:
        data = _encode_run(record)
        payload = json.dumps(data, ensure_ascii=False)
        started_ts = _dt_to_ts(getattr(record, "started_at", None))
        finished_ts = _dt_to_ts(getattr(record, "finished_at", None))

        with self._lock:
            self._db.execute(
                """
                INSERT INTO runs (
                    run_id, data_json,
                    graph_id, status,
                    user_id, org_id, session_id,
                    started_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    payload,
                    getattr(record, "graph_id", None),
                    record.status.value
                    if isinstance(record.status, RunStatus)
                    else str(record.status),
                    getattr(record, "user_id", None),
                    getattr(record, "org_id", None),
                    getattr(record, "session_id", None),
                    started_ts,
                    finished_ts,
                ),
            )

    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: datetime | None = None,
        error: str | None = None,
        meta_update: dict[str, Any] | None = None,
        field_updates: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            row = self._db.execute(
                "SELECT data_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                return

            data = json.loads(row[0])
            data["status"] = status.value if isinstance(status, RunStatus) else str(status)
            if finished_at is not None:
                data["finished_at"] = finished_at.isoformat()
            if error is not None:
                data["error"] = error
            if meta_update:
                meta = dict(data.get("meta") or {})
                meta.update(meta_update)
                data["meta"] = meta
            if field_updates:
                for key, value in field_updates.items():
                    if isinstance(value, datetime):
                        data[key] = value.isoformat()
                    else:
                        data[key] = value

            payload = json.dumps(data, ensure_ascii=False)
            finished_ts = _dt_to_ts(finished_at)

            self._db.execute(
                """
                UPDATE runs
                SET data_json = ?, status = ?, finished_at = ?
                WHERE run_id = ?
                """,
                (payload, status.value, finished_ts, run_id),
            )

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data_json FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        return _decode_run(data)

    def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RunRecord]:
        """
        List runs ordered by started_at DESC.

        NOTE: session_id is optional; you can ignore it if you want to keep
        the signature 100% identical to your current RunStore, or add it
        and update RunManager accordingly.
        """
        where: list[str] = []
        params: list[Any] = []

        if graph_id is not None:
            where.append("graph_id = ?")
            params.append(graph_id)

        if status is not None:
            where.append("status = ?")
            status_val = status.value if isinstance(status, RunStatus) else str(status)
            params.append(status_val)

        if org_id is not None:
            where.append("org_id = ?")
            params.append(org_id)

        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)

        if session_id is not None:
            where.append("session_id = ?")
            params.append(session_id)

        sql = "SELECT data_json FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY started_at DESC"

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        return [_decode_run(json.loads(r[0])) for r in rows]

    def record_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        occurrence_id: str,
        created_at: datetime | None = None,
        max_recent: int = 10,
    ) -> None:
        """Atomically count one stable SQLite run artifact occurrence.

        The receipt keeps content and occurrence identities distinct while its run
        counters and recent-artifact preview update in the same transaction.

        Examples:
            Count an occurrence:
                ```python
                runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

            Replay the occurrence:
                ```python
                runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

        Args:
            run_id: Exact run identity to update.
            artifact_id: Stable content artifact identity.
            occurrence_id: Stable occurrence idempotency identity.
            created_at: Optional occurrence time; defaults to current UTC.
            max_recent: Positive bound for retained content artifact identities.

        Returns:
            None: The occurrence was counted, replayed, or its run was absent.

        Notes:
            Reusing an occurrence with different content or time raises `ValueError`.
        """
        _artifact_identity(artifact_id, occurrence_id)
        if isinstance(max_recent, bool) or not isinstance(max_recent, int) or max_recent < 1:
            raise ValueError("max_recent must be a positive integer")
        occurred_at = created_at or datetime.now(UTC)
        occurred_ts = _dt_to_ts(occurred_at)
        assert occurred_ts is not None
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute(
                    "SELECT data_json FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    self._db.execute("COMMIT")
                    return
                receipt = self._db.execute(
                    """
                    SELECT artifact_id, occurred_at FROM run_artifact_occurrences
                    WHERE run_id = ? AND occurrence_id = ?
                    """,
                    (run_id, occurrence_id),
                ).fetchone()
                if receipt is not None:
                    if str(receipt[0]) != artifact_id or float(receipt[1]) != occurred_ts:
                        raise ValueError("Run artifact occurrence identity conflicts")
                    self._db.execute("COMMIT")
                    return
                self._db.execute(
                    """
                    INSERT INTO run_artifact_occurrences(
                        run_id, occurrence_id, artifact_id, occurred_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (run_id, occurrence_id, artifact_id, occurred_ts),
                )
                record = _decode_run(json.loads(row[0]))
                record.artifact_count = (record.artifact_count or 0) + 1
                record.first_artifact_at = min(
                    occurred_at,
                    record.first_artifact_at or occurred_at,
                )
                record.last_artifact_at = max(
                    occurred_at,
                    record.last_artifact_at or occurred_at,
                )
                recent = list(record.recent_artifact_ids or [])
                recent.append(artifact_id)
                record.recent_artifact_ids = recent[-max_recent:]
                payload = json.dumps(_encode_run(record), ensure_ascii=False)
                self._db.execute(
                    "UPDATE runs SET data_json = ? WHERE run_id = ?",
                    (payload, run_id),
                )
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise


class SQLiteRunStore(RunStore):
    """
    Async RunStore implementation that delegates to SQLiteRunStoreSync
    using asyncio.to_thread for I/O.
    """

    def __init__(self, path: str):
        self._sync = SQLiteRunStoreSync(path)

    async def create(self, record: RunRecord) -> None:
        await asyncio.to_thread(self._sync.create, record)

    async def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        finished_at: datetime | None = None,
        error: str | None = None,
        meta_update: dict[str, Any] | None = None,
        field_updates: dict[str, Any] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._sync.update_status,
            run_id,
            status,
            finished_at=finished_at,
            error=error,
            meta_update=meta_update,
            field_updates=field_updates,
        )

    async def get(self, run_id: str) -> RunRecord | None:
        return await asyncio.to_thread(self._sync.get, run_id)

    async def list(
        self,
        *,
        graph_id: str | None = None,
        status: RunStatus | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        # If you decide to expose session_id here, add it and thread it down.
    ) -> list[RunRecord]:
        # For now we only use graph_id/status; session_id can be added later
        return await asyncio.to_thread(
            self._sync.list,
            graph_id=graph_id,
            status=status,
            user_id=user_id,
            org_id=org_id,
            session_id=session_id,
            limit=limit,
            offset=offset,
        )

    async def record_artifact(
        self,
        run_id: str,
        *,
        artifact_id: str,
        occurrence_id: str,
        created_at: datetime | None = None,
    ) -> None:
        """Count one stable SQLite run artifact occurrence asynchronously.

        The async boundary delegates both content and occurrence identities to the
        transactional synchronous store.

        Examples:
            Count an occurrence:
                ```python
                await runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

            Replay the occurrence:
                ```python
                await runs.record_artifact(
                    "run-1", artifact_id="artifact-1", occurrence_id="occurrence-1"
                )
                ```

        Args:
            run_id: Exact run identity to update.
            artifact_id: Stable content artifact identity.
            occurrence_id: Stable occurrence idempotency identity.
            created_at: Optional occurrence time; defaults to current UTC.

        Returns:
            None: The occurrence was counted, replayed, or its run was absent.

        Notes:
            Collision and transaction errors propagate from the synchronous store.
        """
        await asyncio.to_thread(
            self._sync.record_artifact,
            run_id,
            artifact_id=artifact_id,
            occurrence_id=occurrence_id,
            created_at=created_at,
        )


def _artifact_identity(artifact_id: str, occurrence_id: str) -> None:
    for name, value in (("artifact_id", artifact_id), ("occurrence_id", occurrence_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

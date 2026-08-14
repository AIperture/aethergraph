from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from aethergraph.contracts.storage.trigger_store import TriggerStore
from aethergraph.services.triggers.scheduling import _advance_after_claim, _normalize_utc
from aethergraph.services.triggers.types import TriggerClaim, TriggerRecord


def _timestamp(value: datetime | None) -> float | None:
    return _normalize_utc(value).timestamp() if value is not None else None


def _fire_id(trigger_id: str, scheduled_for: datetime) -> str:
    occurrence = f"{trigger_id}|{_normalize_utc(scheduled_for).isoformat()}"
    digest = hashlib.sha256(occurrence.encode("utf-8")).hexdigest()[:24]
    return f"trigfire-{digest}"


class SQLiteTriggerStore(TriggerStore):
    """Persist trigger definitions and occurrence leases in one SQLite authority."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS triggers (
                    trigger_id TEXT PRIMARY KEY,
                    data_json TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    event_key TEXT,
                    next_fire_at REAL,
                    org_id TEXT,
                    user_id TEXT,
                    client_id TEXT,
                    graph_id TEXT,
                    updated_at REAL NOT NULL
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_triggers_due
                ON triggers(active, kind, next_fire_at)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_triggers_event_scope
                ON triggers(active, kind, event_key, org_id, user_id, client_id)
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS trigger_fires (
                    fire_id TEXT PRIMARY KEY,
                    trigger_id TEXT NOT NULL,
                    scheduled_for REAL NOT NULL,
                    worker_id TEXT,
                    status TEXT NOT NULL,
                    lease_until REAL,
                    attempts INTEGER NOT NULL,
                    retry_at REAL,
                    run_id TEXT,
                    last_error TEXT,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(trigger_id) REFERENCES triggers(trigger_id) ON DELETE CASCADE
                )
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_trigger_fires_retry
                ON trigger_fires(status, retry_at, lease_until)
                """)

    @staticmethod
    def _record(row: sqlite3.Row) -> TriggerRecord:
        return TriggerRecord.from_dict(json.loads(str(row["data_json"])))

    @staticmethod
    def _claim(
        fire_row: sqlite3.Row,
        trigger: TriggerRecord,
        *,
        reclaimed: bool,
    ) -> TriggerClaim:
        return TriggerClaim(
            fire_id=str(fire_row["fire_id"]),
            trigger=trigger,
            scheduled_for=datetime.fromtimestamp(float(fire_row["scheduled_for"]), UTC),
            worker_id=str(fire_row["worker_id"]),
            lease_until=datetime.fromtimestamp(float(fire_row["lease_until"]), UTC),
            attempts=int(fire_row["attempts"]),
            reclaimed=reclaimed,
        )

    @staticmethod
    def _write_record(connection: sqlite3.Connection, trig: TriggerRecord) -> None:
        now_ts = datetime.now(UTC).timestamp()
        connection.execute(
            """
            INSERT INTO triggers (
                trigger_id, data_json, active, kind, event_key, next_fire_at,
                org_id, user_id, client_id, graph_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trigger_id) DO UPDATE SET
                data_json = excluded.data_json,
                active = excluded.active,
                kind = excluded.kind,
                event_key = excluded.event_key,
                next_fire_at = excluded.next_fire_at,
                org_id = excluded.org_id,
                user_id = excluded.user_id,
                client_id = excluded.client_id,
                graph_id = excluded.graph_id,
                updated_at = excluded.updated_at
            """,
            (
                trig.trigger_id,
                json.dumps(trig.to_dict(), ensure_ascii=False, sort_keys=True),
                int(trig.active),
                trig.kind,
                trig.event_key,
                _timestamp(trig.next_fire_at),
                trig.org_id,
                trig.user_id,
                trig.client_id,
                trig.graph_id,
                now_ts,
            ),
        )

    async def create(self, trig: TriggerRecord) -> None:
        await asyncio.to_thread(self._create, trig)

    def _create(self, trig: TriggerRecord) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            exists = connection.execute(
                "SELECT 1 FROM triggers WHERE trigger_id = ?", (trig.trigger_id,)
            ).fetchone()
            if exists is not None:
                raise ValueError(f"Trigger already exists: {trig.trigger_id}")
            self._write_record(connection, trig)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def update(self, trig: TriggerRecord) -> None:
        await asyncio.to_thread(self._update, trig)

    def _update(self, trig: TriggerRecord) -> None:
        with self._connect() as connection:
            self._write_record(connection, trig)

    async def get(self, trigger_id: str) -> TriggerRecord | None:
        return await asyncio.to_thread(self._get, trigger_id)

    def _get(self, trigger_id: str) -> TriggerRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT data_json FROM triggers WHERE trigger_id = ?", (trigger_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    async def delete(self, trigger_id: str) -> None:
        await asyncio.to_thread(self._delete, trigger_id)

    def _delete(self, trigger_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM triggers WHERE trigger_id = ?", (trigger_id,))

    async def list_all(
        self,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        kind: str | None = None,
        active: bool | None = None,
    ) -> list[TriggerRecord]:
        return await asyncio.to_thread(
            self._list_all,
            org_id=org_id,
            user_id=user_id,
            client_id=client_id,
            graph_id=graph_id,
            kind=kind,
            active=active,
        )

    def _list_all(self, **filters: Any) -> list[TriggerRecord]:
        where: list[str] = []
        params: list[Any] = []
        for field in ("org_id", "client_id", "graph_id", "kind"):
            value = filters.get(field)
            if value is not None:
                where.append(f"{field} = ?")
                params.append(value)
        if filters.get("user_id") is not None:
            where.append("(user_id = ? OR client_id = ?)")
            params.extend([filters["user_id"], filters["user_id"]])
        if filters.get("active") is not None:
            where.append("active = ?")
            params.append(int(filters["active"]))
        sql = "SELECT data_json FROM triggers"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_at DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._record(row) for row in rows]

    async def list_by_event_key(
        self,
        event_key: str,
        *,
        org_id: str | None = None,
        user_id: str | None = None,
        client_id: str | None = None,
    ) -> list[TriggerRecord]:
        if org_id is None and user_id is None and client_id is None:
            raise ValueError("Event trigger reads require an explicit tenant scope")
        return await asyncio.to_thread(
            self._list_by_event_key,
            event_key,
            org_id=org_id,
            user_id=user_id,
            client_id=client_id,
        )

    def _list_by_event_key(self, event_key: str, **scope: Any) -> list[TriggerRecord]:
        where = ["active = 1", "kind = 'event'", "event_key = ?"]
        params: list[Any] = [event_key]
        if scope.get("org_id") is not None:
            where.append("org_id = ?")
            params.append(scope["org_id"])
        if scope.get("user_id") is not None:
            where.append("(user_id = ? OR client_id = ?)")
            params.extend([scope["user_id"], scope["user_id"]])
        if scope.get("client_id") is not None:
            where.append("client_id = ?")
            params.append(scope["client_id"])
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT data_json FROM triggers WHERE " + " AND ".join(where), params
            ).fetchall()
        return [self._record(row) for row in rows]

    async def claim_due(
        self,
        now: datetime,
        *,
        worker_id: str,
        lease_until: datetime,
        limit: int,
        skip_missed_before: datetime | None = None,
    ) -> list[TriggerClaim]:
        """Atomically claim due trigger occurrences across workers.

        Intro:
            Leases retryable occurrences and advances newly claimed schedules in
            one SQLite write transaction.

        Examples:
            Claim due work:
            ```python
            claims = await store.claim_due(
                now, worker_id="worker-a", lease_until=lease_until, limit=100
            )
            ```

            Skip pre-start misses for non-catch-up schedules:
            ```python
            claims = await store.claim_due(
                now,
                worker_id="worker-a",
                lease_until=lease_until,
                limit=100,
                skip_missed_before=started_at,
            )
            ```

        Args:
            now: Current UTC scan instant.
            worker_id: Unique owner of returned leases.
            lease_until: UTC expiry for returned leases.
            limit: Maximum claims returned.
            skip_missed_before: Startup boundary for explicit missed-run skipping.

        Returns:
            list[TriggerClaim]: Worker-owned occurrences, including reclaimed leases.

        Notes:
            SQLite `BEGIN IMMEDIATE` prevents competing workers from claiming the
            same stable fire identifier.
        """
        return await asyncio.to_thread(
            self._claim_due,
            _normalize_utc(now),
            worker_id,
            _normalize_utc(lease_until),
            limit,
            _normalize_utc(skip_missed_before) if skip_missed_before else None,
        )

    def _claim_due(
        self,
        now: datetime,
        worker_id: str,
        lease_until: datetime,
        limit: int,
        skip_missed_before: datetime | None,
    ) -> list[TriggerClaim]:
        if limit <= 0:
            return []
        now_ts = now.timestamp()
        lease_ts = lease_until.timestamp()
        claims: list[TriggerClaim] = []
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            retry_rows = connection.execute(
                """
                SELECT f.*, t.data_json
                FROM trigger_fires AS f
                JOIN triggers AS t ON t.trigger_id = f.trigger_id
                WHERE (f.status = 'retry' AND f.retry_at <= ?)
                   OR (f.status = 'leased' AND f.lease_until <= ?)
                ORDER BY f.scheduled_for ASC
                LIMIT ?
                """,
                (now_ts, now_ts, limit),
            ).fetchall()
            for row in retry_rows:
                connection.execute(
                    """
                    UPDATE trigger_fires
                    SET status = 'leased', worker_id = ?, lease_until = ?,
                        retry_at = NULL, attempts = attempts + 1, updated_at = ?
                    WHERE fire_id = ?
                    """,
                    (worker_id, lease_ts, now_ts, row["fire_id"]),
                )
                refreshed = connection.execute(
                    "SELECT * FROM trigger_fires WHERE fire_id = ?", (row["fire_id"],)
                ).fetchone()
                claims.append(self._claim(refreshed, self._record(row), reclaimed=True))

            remaining = limit - len(claims)
            if remaining > 0:
                due_rows = connection.execute(
                    """
                    SELECT * FROM triggers
                    WHERE active = 1 AND kind != 'event'
                      AND next_fire_at IS NOT NULL AND next_fire_at <= ?
                    ORDER BY next_fire_at ASC
                    LIMIT ?
                    """,
                    (now_ts, remaining),
                ).fetchall()
                for row in due_rows:
                    trig = self._record(row)
                    scheduled_for = datetime.fromtimestamp(float(row["next_fire_at"]), UTC)
                    fire_id = _fire_id(trig.trigger_id, scheduled_for)
                    if (
                        skip_missed_before is not None
                        and scheduled_for < skip_missed_before
                        and not trig.catch_up_missed
                    ):
                        if trig.kind == "one_shot":
                            trig.active = False
                            trig.next_fire_at = None
                        else:
                            trig.next_fire_at = _advance_after_claim(
                                trig, scheduled_for=scheduled_for, now=skip_missed_before
                            )
                        connection.execute(
                            """
                            INSERT OR IGNORE INTO trigger_fires (
                                fire_id, trigger_id, scheduled_for, worker_id, status,
                                lease_until, attempts, retry_at, run_id, last_error, updated_at
                            ) VALUES (?, ?, ?, NULL, 'skipped_missed', NULL, 0, NULL, NULL, NULL, ?)
                            """,
                            (fire_id, trig.trigger_id, scheduled_for.timestamp(), now_ts),
                        )
                        self._write_record(connection, trig)
                        continue

                    inserted = connection.execute(
                        """
                        INSERT OR IGNORE INTO trigger_fires (
                            fire_id, trigger_id, scheduled_for, worker_id, status,
                            lease_until, attempts, retry_at, run_id, last_error, updated_at
                        ) VALUES (?, ?, ?, ?, 'leased', ?, 1, NULL, NULL, NULL, ?)
                        """,
                        (
                            fire_id,
                            trig.trigger_id,
                            scheduled_for.timestamp(),
                            worker_id,
                            lease_ts,
                            now_ts,
                        ),
                    )
                    if inserted.rowcount != 1:
                        continue
                    if trig.kind == "one_shot":
                        trig.active = False
                        trig.next_fire_at = None
                    else:
                        trig.next_fire_at = _advance_after_claim(
                            trig, scheduled_for=scheduled_for, now=now
                        )
                    self._write_record(connection, trig)
                    fire_row = connection.execute(
                        "SELECT * FROM trigger_fires WHERE fire_id = ?", (fire_id,)
                    ).fetchone()
                    claims.append(self._claim(fire_row, trig, reclaimed=False))
            connection.commit()
            return claims
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def complete_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        run_id: str,
        completed_at: datetime,
    ) -> bool:
        return await asyncio.to_thread(
            self._complete_claim, fire_id, worker_id, run_id, _normalize_utc(completed_at)
        )

    def _complete_claim(
        self, fire_id: str, worker_id: str, run_id: str, completed_at: datetime
    ) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT trigger_id FROM trigger_fires WHERE fire_id = ?", (fire_id,)
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            cursor = connection.execute(
                """
                UPDATE trigger_fires
                SET status = 'delivered', worker_id = NULL, lease_until = NULL,
                    retry_at = NULL, run_id = ?, last_error = NULL, updated_at = ?
                WHERE fire_id = ? AND worker_id = ? AND status = 'leased'
                """,
                (run_id, completed_at.timestamp(), fire_id, worker_id),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return False
            trigger_row = connection.execute(
                "SELECT data_json FROM triggers WHERE trigger_id = ?", (row["trigger_id"],)
            ).fetchone()
            if trigger_row is not None:
                trig = self._record(trigger_row)
                trig.last_fired_at = completed_at
                self._write_record(connection, trig)
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def fail_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        error: str,
        retry_at: datetime,
    ) -> bool:
        return await asyncio.to_thread(
            self._fail_claim, fire_id, worker_id, error, _normalize_utc(retry_at)
        )

    def _fail_claim(self, fire_id: str, worker_id: str, error: str, retry_at: datetime) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE trigger_fires
                SET status = 'retry', worker_id = NULL, lease_until = NULL,
                    retry_at = ?, last_error = ?, updated_at = ?
                WHERE fire_id = ? AND worker_id = ? AND status = 'leased'
                """,
                (
                    retry_at.timestamp(),
                    error[:2000],
                    datetime.now(UTC).timestamp(),
                    fire_id,
                    worker_id,
                ),
            )
            return cursor.rowcount == 1

    async def skip_claim(
        self,
        fire_id: str,
        *,
        worker_id: str,
        reason: str,
        completed_at: datetime,
    ) -> bool:
        return await asyncio.to_thread(
            self._skip_claim, fire_id, worker_id, reason, _normalize_utc(completed_at)
        )

    def _skip_claim(
        self, fire_id: str, worker_id: str, reason: str, completed_at: datetime
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE trigger_fires
                SET status = ?, worker_id = NULL, lease_until = NULL,
                    retry_at = NULL, last_error = NULL, updated_at = ?
                WHERE fire_id = ? AND worker_id = ? AND status = 'leased'
                """,
                (f"skipped_{reason}", completed_at.timestamp(), fire_id, worker_id),
            )
            return cursor.rowcount == 1

    async def get_claim(self, fire_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_claim, fire_id)

    def _get_claim(self, fire_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM trigger_fires WHERE fire_id = ?", (fire_id,)
            ).fetchone()
        return dict(row) if row is not None else None

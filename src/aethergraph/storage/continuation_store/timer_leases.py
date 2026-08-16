from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sqlite3

from aethergraph.services.continuations.timer_lease import TimerLease, TimerLeaseStatus


class SQLiteContinuationTimerLeaseStore:
    """Persist atomic continuation-timer claims in SQLite."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS continuation_timer_leases_v2 (
                    fire_id TEXT PRIMARY KEY,
                    continuation_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    scheduled_for REAL NOT NULL,
                    worker_id TEXT,
                    status TEXT NOT NULL,
                    lease_until REAL,
                    attempts INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    next_attempt_at REAL,
                    last_error TEXT,
                    finished_at REAL,
                    updated_at REAL NOT NULL
                )
                """)
            columns = {
                str(row["name"])
                for row in connection.execute(
                    "PRAGMA table_info(continuation_timer_leases_v2)"
                ).fetchall()
            }
            if "revision" not in columns:
                connection.execute(
                    """
                    ALTER TABLE continuation_timer_leases_v2
                    ADD COLUMN revision INTEGER NOT NULL DEFAULT 1
                    """
                )
            if "finished_at" not in columns:
                connection.execute(
                    "ALTER TABLE continuation_timer_leases_v2 ADD COLUMN finished_at REAL"
                )
                connection.execute(
                    """
                    UPDATE continuation_timer_leases_v2
                    SET finished_at = updated_at
                    WHERE status IN ('delivered', 'dead_letter')
                    """
                )
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_continuation_timer_retry_v2
                ON continuation_timer_leases_v2(status, next_attempt_at, lease_until)
                """)

    @staticmethod
    def _from_row(row: sqlite3.Row, *, reclaimed: bool = False) -> TimerLease:
        return TimerLease(
            fire_id=str(row["fire_id"]),
            continuation_id=str(row["continuation_id"]),
            run_id=str(row["run_id"]),
            node_id=str(row["node_id"]),
            scheduled_for=datetime.fromtimestamp(float(row["scheduled_for"]), tz=UTC),
            worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
            status=TimerLeaseStatus(str(row["status"])),
            lease_until=(
                datetime.fromtimestamp(float(row["lease_until"]), tz=UTC)
                if row["lease_until"] is not None
                else None
            ),
            attempts=int(row["attempts"]),
            revision=int(row["revision"]),
            updated_at=datetime.fromtimestamp(float(row["updated_at"]), tz=UTC),
            next_attempt_at=(
                datetime.fromtimestamp(float(row["next_attempt_at"]), tz=UTC)
                if row["next_attempt_at"] is not None
                else None
            ),
            last_error=str(row["last_error"]) if row["last_error"] is not None else None,
            finished_at=(
                datetime.fromtimestamp(float(row["finished_at"]), tz=UTC)
                if row["finished_at"] is not None
                else None
            ),
            reclaimed=reclaimed,
        )

    async def claim(
        self,
        *,
        fire_id: str,
        continuation_id: str,
        run_id: str,
        node_id: str,
        scheduled_for: datetime,
        worker_id: str,
        now: datetime,
        lease_until: datetime,
    ) -> TimerLease | None:
        """Atomically claim one due continuation fire.

        Intro:
            Creates a durable lease or reclaims an expired lease in one SQLite
            write transaction. Delivered, dead-lettered, backoff-delayed, and
            currently leased fires are not returned.

        Examples:
            Claim a new fire:
            ```python
            lease = await store.claim(
                fire_id="fire-1",
                continuation_id="cont-1",
                run_id="run-1",
                node_id="wait",
                scheduled_for=scheduled_for,
                worker_id="worker-a",
                now=now,
                lease_until=lease_until,
            )
            ```

            Detect a competing worker:
            ```python
            if await store.claim(
                fire_id="fire-1",
                continuation_id="cont-1",
                run_id="run-1",
                node_id="wait",
                scheduled_for=scheduled_for,
                worker_id="worker-b",
                now=now,
                lease_until=lease_until,
            ) is None:
                print("already claimed")
            ```

        Args:
            fire_id: Stable identity for one scheduled continuation occurrence.
            continuation_id: Stable continuation identity.
            run_id: Exact durable run identity.
            node_id: Exact waiting node identity.
            scheduled_for: Exact UTC occurrence time.
            worker_id: Timer worker attempting delivery.
            now: Injected current UTC time.
            lease_until: UTC time after which another worker may reclaim the fire.

        Returns:
            TimerLease | None: Claimed lease, or `None` when the fire is not
            currently eligible. `TimerLease.reclaimed` identifies stale-lease recovery.

        Notes:
            SQLite `BEGIN IMMEDIATE` serializes competing claims across processes.
        """
        now_ts = now.timestamp()
        lease_until_ts = lease_until.timestamp()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM continuation_timer_leases_v2 WHERE fire_id = ?",
                (fire_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO continuation_timer_leases_v2 (
                        fire_id, continuation_id, run_id, node_id, scheduled_for,
                        worker_id, status,
                        lease_until, attempts, revision, next_attempt_at, last_error,
                        finished_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'leased', ?, 1, 1, NULL, NULL, NULL, ?)
                    """,
                    (
                        fire_id,
                        continuation_id,
                        run_id,
                        node_id,
                        scheduled_for.timestamp(),
                        worker_id,
                        lease_until_ts,
                        now_ts,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM continuation_timer_leases_v2 WHERE fire_id = ?",
                    (fire_id,),
                ).fetchone()
                connection.commit()
                return self._from_row(row)

            status = str(row["status"])
            identity = (
                str(row["continuation_id"]),
                str(row["run_id"]),
                str(row["node_id"]),
                float(row["scheduled_for"]),
            )
            requested = (
                continuation_id,
                run_id,
                node_id,
                scheduled_for.timestamp(),
            )
            if identity != requested:
                connection.rollback()
                raise ValueError("Timer fire identity conflicts with its durable receipt")
            current_lease_until = row["lease_until"]
            next_attempt_at = row["next_attempt_at"]
            if status in {"delivered", "dead_letter"}:
                connection.rollback()
                return None
            if next_attempt_at is not None and float(next_attempt_at) > now_ts:
                connection.rollback()
                return None
            if (
                status == "leased"
                and current_lease_until is not None
                and float(current_lease_until) > now_ts
            ):
                connection.rollback()
                return None

            reclaimed = status == "leased" and current_lease_until is not None
            connection.execute(
                """
                UPDATE continuation_timer_leases_v2
                SET worker_id = ?, status = 'leased', lease_until = ?,
                    attempts = attempts + 1, revision = revision + 1,
                    next_attempt_at = NULL, finished_at = NULL, updated_at = ?
                WHERE fire_id = ?
                """,
                (worker_id, lease_until_ts, now_ts, fire_id),
            )
            row = connection.execute(
                "SELECT * FROM continuation_timer_leases_v2 WHERE fire_id = ?",
                (fire_id,),
            ).fetchone()
            connection.commit()
            return self._from_row(row, reclaimed=reclaimed)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def complete(self, lease: TimerLease, *, now: datetime) -> bool:
        """Mark one worker-owned timer fire delivered.

        Intro:
            Transitions a leased fire to the durable terminal `delivered` state
            only when the caller still owns that lease.

        Examples:
            Complete a successful delivery:
            ```python
            changed = await store.complete(lease, now=now)
            ```

            Detect stale ownership:
            ```python
            if not await store.complete(stale, now=now):
                print("lease no longer owned")
            ```

        Args:
            lease: Exact worker-owned lease and expected revision.
            now: Injected current UTC time.

        Returns:
            bool: `True` when the leased row transitioned to delivered.

        Notes:
            Delivered rows remain as durable deduplication receipts.
        """
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE continuation_timer_leases_v2
                SET status = 'delivered', worker_id = NULL, lease_until = NULL,
                    next_attempt_at = NULL, last_error = NULL, finished_at = ?,
                    revision = revision + 1, updated_at = ?
                WHERE fire_id = ? AND worker_id = ? AND status = 'leased'
                  AND revision = ?
                """,
                (
                    now.timestamp(),
                    now.timestamp(),
                    lease.fire_id,
                    lease.worker_id,
                    lease.revision,
                ),
            )
            return cursor.rowcount == 1

    async def record_failure(
        self,
        lease: TimerLease,
        *,
        now: datetime,
        next_attempt_at: datetime | None,
        error: str,
        dead_letter: bool,
    ) -> bool:
        """Persist retry or dead-letter state for one failed delivery.

        Intro:
            Releases a worker-owned lease into bounded retry backoff or a durable
            terminal dead-letter state without consuming the continuation itself.

        Examples:
            Schedule a retry:
            ```python
            await store.record_failure(
                lease,
                now=now,
                next_attempt_at=retry_at,
                error="scheduler unavailable",
                dead_letter=False,
            )
            ```

            Dead-letter an exhausted fire:
            ```python
            await store.record_failure(
                lease,
                now=now,
                next_attempt_at=None,
                error="retry limit reached",
                dead_letter=True,
            )
            ```

        Args:
            lease: Exact worker-owned lease and expected revision.
            now: Injected current UTC time.
            next_attempt_at: Next eligible UTC delivery time, or `None` for dead letter.
            error: Bounded failure description.
            dead_letter: Whether to terminalize instead of retrying.

        Returns:
            bool: `True` when the worker-owned lease was updated.

        Notes:
            The continuation store is not mutated on delivery failure.
        """
        status = "dead_letter" if dead_letter else "retry"
        next_ts = next_attempt_at.timestamp() if next_attempt_at is not None else None
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE continuation_timer_leases_v2
                SET status = ?, worker_id = NULL, lease_until = NULL,
                    next_attempt_at = ?, last_error = ?, finished_at = ?,
                    revision = revision + 1, updated_at = ?
                WHERE fire_id = ? AND worker_id = ? AND status = 'leased'
                  AND revision = ?
                """,
                (
                    status,
                    next_ts,
                    error[:1000],
                    now.timestamp() if dead_letter else None,
                    now.timestamp(),
                    lease.fire_id,
                    lease.worker_id,
                    lease.revision,
                ),
            )
            return cursor.rowcount == 1

    async def get(self, run_id: str, node_id: str, fire_id: str) -> TimerLease | None:
        """Read one durable timer lease or receipt.

        Intro:
            Returns the current persisted state for diagnostics and tests without
            changing lease ownership.

        Examples:
            Inspect a delivered fire:
            ```python
            receipt = await store.get("run-1", "node-1", "fire-1")
            assert receipt is not None and receipt.status == "delivered"
            ```

            Detect an unknown fire:
            ```python
            assert await store.get("run-1", "node-1", "missing") is None
            ```

        Args:
            run_id: Exact durable run identity.
            node_id: Exact waiting node identity.
            fire_id: Stable scheduled occurrence identity.

        Returns:
            TimerLease | None: Persisted state, or `None` when unknown.

        Notes:
            Scope mismatch returns `None`; `reclaimed` is `False` for ordinary reads.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM continuation_timer_leases_v2
                WHERE fire_id = ? AND run_id = ? AND node_id = ?
                """,
                (fire_id, run_id, node_id),
            ).fetchone()
        return self._from_row(row) if row is not None else None

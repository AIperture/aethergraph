from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

"""
This is not used in the main codebase; only used by async wrapper SqliteEventLog.
"""


class SQLiteEventLogSync:
    def __init__(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(
            str(path_obj),
            check_same_thread=False,
            isolation_level=None,
        )
        self._lock = threading.RLock()
        self._initialize_db()

    def _initialize_db(self) -> None:
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                scope_id  TEXT,
                kind      TEXT,
                tags_json TEXT,
                payload   TEXT NOT NULL,
                -- new tenant / dimension columns
                user_id   TEXT,
                org_id    TEXT,
                run_id    TEXT,
                session_id TEXT
            )
            """
        )
        # Migration for existing DBs
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(events)").fetchall()}
        if "user_id" not in cols:
            self._db.execute("ALTER TABLE events ADD COLUMN user_id TEXT")
        if "org_id" not in cols:
            self._db.execute("ALTER TABLE events ADD COLUMN org_id TEXT")
        if "run_id" not in cols:
            self._db.execute("ALTER TABLE events ADD COLUMN run_id TEXT")
        if "session_id" not in cols:
            self._db.execute("ALTER TABLE events ADD COLUMN session_id TEXT")

        # Existing indexes
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_scope ON events(scope_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_kind  ON events(kind)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(ts)")

        # tenant-aware indexes
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_org_ts ON events(org_id, ts)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts)")

        # Compound index for efficient keyset pagination on scoped event queries
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_scope_kind_id ON events(scope_id, kind, id)"
        )

    def append(self, evt: dict) -> None:
        row = dict(evt)
        partition_scope_id = row.pop("_partition_scope_id", row.get("scope_id"))

        ts = row.get("ts")
        if isinstance(ts, datetime):
            ts = ts.timestamp()
        elif isinstance(ts, int | float):
            ts = float(ts)
        elif isinstance(ts, str):
            # Handle ISO 8601 timestamps like '2025-11-27T19:48:09.758687+00:00' or ...Z
            try:
                s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ts = dt.timestamp()
            except Exception:
                # Fallback: current time if we can't parse
                ts = time.time()

        if ts is None:
            ts = time.time()

        scope_id = partition_scope_id
        kind = row.get("kind")
        tags = row.get("tags") or []
        tags_json = json.dumps(tags, ensure_ascii=False)

        # tenant & run dims (not all events will have these fields. Chat events can just use session_id to retrieve info after optional authentication)
        user_id = row.get("user_id")
        org_id = row.get("org_id")
        run_id = row.get("run_id")
        session_id = row.get("session_id")

        # Optionally overwrite the ts in the payload to the normalized float
        row["ts"] = ts
        payload = json.dumps(row, ensure_ascii=False)

        with self._lock:
            self._db.execute(
                """
                INSERT INTO events (ts, scope_id, kind, tags_json, payload, user_id, org_id, run_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, scope_id, kind, tags_json, payload, user_id, org_id, run_id, session_id),
            )

    def query(
        self,
        *,
        scope_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
        offset: int = 0,
        user_id: str | None = None,
        org_id: str | None = None,
        after_id: int | None = None,
        before_id: int | None = None,
    ) -> list[dict]:
        where: list[str] = []
        params: list[Any] = []

        if after_id is not None:
            where.append("id > ?")
            params.append(after_id)

        if before_id is not None:
            where.append("id < ?")
            params.append(before_id)

        if scope_id is not None:
            where.append("scope_id = ?")
            params.append(scope_id)

        if since is not None:
            where.append("ts >= ?")
            params.append(since.timestamp())

        if until is not None:
            where.append("ts <= ?")
            params.append(until.timestamp())

        if kinds:
            where.append(f"kind IN ({', '.join('?' for _ in kinds)})")
            params.extend(kinds)

        # Tenant-level filters for metering
        if user_id is not None:
            where.append("user_id = ?")
            params.append(user_id)
        if org_id is not None:
            where.append("org_id = ?")
            params.append(org_id)

        # When keyset cursor is used, order by id for stable pagination.
        # before_id uses DESC order (fetching older events), reversed at the end.
        # Otherwise keep legacy ts ordering.
        if before_id is not None:
            order_col = "id"
            order_dir = "DESC"
        elif after_id is not None:
            order_col = "id"
            order_dir = "ASC"
        else:
            order_col = "ts"
            order_dir = "ASC"

        need_python_tag_filter = bool(tags)

        # Build SQL – push LIMIT/OFFSET to SQL when no Python-side tag filtering
        sql = "SELECT id, payload, tags_json FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_col} {order_dir}"

        if not need_python_tag_filter:
            if offset:
                sql += f" LIMIT {limit if limit is not None else -1} OFFSET {offset}"
            elif limit is not None:
                sql += f" LIMIT {limit}"

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        tags_set = set(tags or [])
        filtered: list[dict] = []
        for row_id, payload_str, tags_json_str in rows:
            evt = json.loads(payload_str)
            if need_python_tag_filter:
                row_tags = set(json.loads(tags_json_str) or [])
                if not row_tags.issuperset(tags_set):
                    continue
            # Inject row id so callers can build keyset cursors
            evt["_row_id"] = row_id
            filtered.append(evt)

        # Apply offset/limit in Python only when tag filtering is active
        if need_python_tag_filter:
            if offset:
                filtered = filtered[offset:]
            if limit is not None:
                filtered = filtered[:limit]

        # When fetching backwards (before_id), results come in DESC order.
        # Reverse to return in chronological (ASC) order.
        if before_id is not None:
            filtered.reverse()

        return filtered

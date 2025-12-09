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
    """
    Append-only event log on SQLite.

    Each row:
      - id        INTEGER PRIMARY KEY AUTOINCREMENT
      - ts        REAL (seconds since epoch)
      - scope_id  TEXT
      - kind      TEXT
      - tags_json TEXT (JSON list[str])
      - payload   TEXT (JSON dict, full event)
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
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL NOT NULL,
                scope_id  TEXT,
                kind      TEXT,
                tags_json TEXT,
                payload   TEXT NOT NULL
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_scope ON events(scope_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_kind  ON events(kind)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(ts)")
        self._lock = threading.RLock()

    def append(self, evt: dict) -> None:
        row = dict(evt)

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

        scope_id = row.get("scope_id")
        kind = row.get("kind")
        tags = row.get("tags") or []
        tags_json = json.dumps(tags, ensure_ascii=False)

        # Optionally overwrite the ts in the payload to the normalized float
        row["ts"] = ts
        payload = json.dumps(row, ensure_ascii=False)

        with self._lock:
            self._db.execute(
                """
                INSERT INTO events (ts, scope_id, kind, tags_json, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, scope_id, kind, tags_json, payload),
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
        offset: int = 0,  # 🔹 NEW
    ) -> list[dict]:
        # NOTE: This pushes scope/time/kind filters and ts ordering into SQL,
        # then:
        #   - Loads all matching rows into Python
        #   - Applies tag filtering
        #   - Applies offset + limit on the filtered list
        #
        # For large event volumes, we'd want to:
        #   - Normalize tags into a separate table or use better JSON indexing, and
        #   - Do tag filtering + LIMIT/OFFSET (or keyset) in SQL instead of Python.
        where: list[str] = []
        params: list[Any] = []

        if scope_id is not None:
            where.append("scope_id = ?")
            params.append(scope_id)

        if since is not None:
            where.append("ts >= ?")
            params.append(since.timestamp())

        if until is not None:
            where.append("ts <= ?")
            params.append(until.timestamp())

        if kinds is not None and len(kinds) > 0:
            where.append(f"kind IN ({', '.join('?' for _ in kinds)})")
            params.extend(kinds)

        sql = "SELECT payload, tags_json FROM events"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts ASC"

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        tags_set = set(tags or [])
        filtered: list[dict] = []
        for payload_str, tags_json in rows:
            evt = json.loads(payload_str)
            if tags:
                row_tags = set(json.loads(tags_json) or [])
                if not row_tags.issuperset(tags_set):
                    continue
            filtered.append(evt)

        # Apply offset + limit AFTER all filters
        if offset:
            filtered = filtered[offset:]
        if limit is not None:
            filtered = filtered[:limit]

        return filtered

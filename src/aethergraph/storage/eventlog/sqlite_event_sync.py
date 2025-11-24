from __future__ import annotations

from datetime import datetime
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
        if ts is None:
            ts = time.time()

        scope_id = row.get("scope_id")
        kind = row.get("kind")
        tags = row.get("tags") or []
        tags_json = json.dumps(tags, ensure_ascii=False)
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
    ) -> list[dict]:
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
        if limit is not None:
            sql += f" LIMIT {int(limit)}"

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        out: list[dict] = []
        tags_set = set(tags or [])
        for payload_str, tags_json in rows:
            evt = json.loads(payload_str)
            if tags:
                row_tags = set(json.loads(tags_json) or [])
                if not row_tags.issuperset(tags_set):
                    continue
            out.append(evt)
        return out

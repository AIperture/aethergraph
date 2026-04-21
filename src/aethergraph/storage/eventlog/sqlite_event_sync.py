from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


class SQLiteEventLogSync:
    def __init__(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path_obj), check_same_thread=False, isolation_level=None)
        self._lock = threading.RLock()
        self._initialize_db()

    def _initialize_db(self) -> None:
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                scope_id TEXT,
                kind TEXT,
                tags_json TEXT,
                payload TEXT NOT NULL,
                user_id TEXT,
                org_id TEXT,
                run_id TEXT,
                session_id TEXT,
                client_id TEXT,
                agent_id TEXT,
                graph_id TEXT,
                node_id TEXT,
                topic TEXT,
                tool TEXT,
                severity INTEGER,
                signal REAL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS event_tags (
                event_row_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                UNIQUE(event_row_id, tag)
            )
            """
        )
        cols = {row[1] for row in self._db.execute("PRAGMA table_info(events)").fetchall()}
        for name, type_sql in (
            ("user_id", "TEXT"),
            ("org_id", "TEXT"),
            ("run_id", "TEXT"),
            ("session_id", "TEXT"),
            ("client_id", "TEXT"),
            ("agent_id", "TEXT"),
            ("graph_id", "TEXT"),
            ("node_id", "TEXT"),
            ("topic", "TEXT"),
            ("tool", "TEXT"),
            ("severity", "INTEGER"),
            ("signal", "REAL"),
        ):
            if name not in cols:
                self._db.execute(f"ALTER TABLE events ADD COLUMN {name} {type_sql}")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_scope ON events(scope_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_org_ts ON events(org_id, ts)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_run_ts ON events(run_id, ts)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_session_ts ON events(session_id, ts)"
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_events_agent_ts ON events(agent_id, ts)")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_scope_kind_id ON events(scope_id, kind, id)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_scope_tool_ts ON events(scope_id, tool, ts)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_scope_topic_ts ON events(scope_id, topic, ts)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_tags_tag_row ON event_tags(tag, event_row_id)"
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
            try:
                s = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                ts = dt.timestamp()
            except Exception:
                ts = time.time()
        if ts is None:
            ts = time.time()
        row["ts"] = ts
        tags = [str(tag) for tag in (row.get("tags") or [])]
        payload = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._db.execute(
                """
                INSERT INTO events (
                    ts, scope_id, kind, tags_json, payload,
                    user_id, org_id, run_id, session_id, client_id,
                    agent_id, graph_id, node_id, topic, tool, severity, signal
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    partition_scope_id,
                    row.get("kind"),
                    json.dumps(tags, ensure_ascii=False),
                    payload,
                    row.get("user_id"),
                    row.get("org_id"),
                    row.get("run_id"),
                    row.get("session_id"),
                    row.get("client_id"),
                    row.get("agent_id"),
                    row.get("graph_id"),
                    row.get("node_id"),
                    row.get("topic"),
                    row.get("tool"),
                    row.get("severity"),
                    row.get("signal"),
                ),
            )
            row_id = self._db.execute("SELECT last_insert_rowid()").fetchone()[0]
            if tags:
                self._db.executemany(
                    "INSERT OR IGNORE INTO event_tags (event_row_id, tag) VALUES (?, ?)",
                    [(row_id, tag) for tag in tags],
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
        client_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        after_id: int | None = None,
        before_id: int | None = None,
    ) -> list[dict]:
        where: list[str] = []
        join_params: list[Any] = []
        where_params: list[Any] = []
        if after_id is not None:
            where.append("events.id > ?")
            where_params.append(after_id)
        if before_id is not None:
            where.append("events.id < ?")
            where_params.append(before_id)
        if scope_id is not None:
            where.append("events.scope_id = ?")
            where_params.append(scope_id)
        if since is not None:
            where.append("events.ts >= ?")
            where_params.append(since.timestamp())
        if until is not None:
            where.append("events.ts <= ?")
            where_params.append(until.timestamp())
        if kinds:
            where.append(f"events.kind IN ({', '.join('?' for _ in kinds)})")
            where_params.extend(kinds)
        for column, value in (
            ("user_id", user_id),
            ("org_id", org_id),
            ("client_id", client_id),
            ("session_id", session_id),
            ("run_id", run_id),
            ("agent_id", agent_id),
            ("graph_id", graph_id),
            ("node_id", node_id),
            ("topic", topic),
            ("tool", tool),
        ):
            if value is not None:
                where.append(f"events.{column} = ?")
                where_params.append(value)
        order_col, order_dir = (
            ("id", "DESC")
            if before_id is not None
            else ("id", "ASC")
            if after_id is not None
            else ("ts", "ASC")
        )
        sql = "SELECT DISTINCT events.id, events.payload FROM events"
        if tags:
            for idx, tag in enumerate(tags):
                alias = f"et{idx}"
                sql += f" JOIN event_tags {alias} ON {alias}.event_row_id = events.id AND {alias}.tag = ?"
                join_params.append(tag)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY events.{order_col} {order_dir}"
        if offset:
            sql += f" LIMIT {limit if limit is not None else -1} OFFSET {offset}"
        elif limit is not None:
            sql += f" LIMIT {limit}"
        params = [*join_params, *where_params]
        with self._lock:
            rows = self._db.execute(sql, params).fetchall()
        out: list[dict] = []
        for row_id, payload_str in rows:
            evt = json.loads(payload_str)
            evt["_row_id"] = row_id
            out.append(evt)
        if before_id is not None:
            out.reverse()
        return out

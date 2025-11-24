from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

"""
SQLite Key-Value Store with TTL (synchronous). Only used by async wrapper SQLiteKV.
"""


class SQLiteKVSync:
    """
    Durable KV with TTL (JSON values), thread-safe via RLock.
    """

    def __init__(self, path: str, *, prefix: str = ""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
              k TEXT PRIMARY KEY,
              v TEXT,
              expire_at REAL
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS kv_exp_idx ON kv(expire_at);")
        self._lock = threading.RLock()
        self._prefix = prefix

    def _k(self, k: str) -> str:
        return f"{self._prefix}{k}" if self._prefix else k

    def get(self, key: str, default: Any = None) -> Any:
        k = self._k(key)
        with self._lock:
            row = self._db.execute("SELECT v, expire_at FROM kv WHERE k=?", (k,)).fetchone()
            if not row:
                return default
            v_txt, exp = row
            if exp and exp < time.time():
                self._db.execute("DELETE FROM kv WHERE k=?", (k,))
                return default
            try:
                return json.loads(v_txt)
            except Exception:
                return default

    def set(self, key: str, value: Any, ttl_s: int | None = None) -> None:
        k = self._k(key)
        exp = time.time() + ttl_s if ttl_s is not None else None
        v_txt = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._db.execute(
                """
                INSERT INTO kv (k, v, expire_at) VALUES (?, ?, ?)
                ON CONFLICT(k) DO UPDATE SET v=excluded.v, expire_at=excluded.expire_at
                """,
                (k, v_txt, exp),
            )

    def delete(self, key: str) -> None:
        k = self._k(key)
        with self._lock:
            self._db.execute("DELETE FROM kv WHERE k=?", (k,))

    def mget(self, keys: list[str]) -> list[Any]:
        return [self.get(k) for k in keys]

    def mset(self, kv: dict[str, Any], ttl_s: int | None = None) -> None:
        for k, v in kv.items():
            self.set(k, v, ttl_s=ttl_s)

    def expire(self, key: str, ttl_s: int) -> None:
        k = self._k(key)
        exp = time.time() + ttl_s
        with self._lock:
            self._db.execute("UPDATE kv SET expire_at=? WHERE k=?", (exp, k))

    def purge_expired(self, limit: int = 1000) -> int:
        now = time.time()
        with self._lock:
            rows = self._db.execute(
                "SELECT k FROM kv WHERE expire_at IS NOT NULL AND expire_at < ? LIMIT ?",
                (now, limit),
            ).fetchall()
            keys = [r[0] for r in rows]
            if not keys:
                return 0
            self._db.executemany("DELETE FROM kv WHERE k=?", [(k,) for k in keys])
        return len(keys)

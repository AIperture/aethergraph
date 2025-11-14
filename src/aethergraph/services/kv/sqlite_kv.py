from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any


class SQLiteKV:
    """
    Durable KV with TTL (JSON values).
    Thread-safe via RLock; async callers can await these methods safely.
    """

    def __init__(self, path: str, *, prefix: str = ""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS kv (
              k TEXT PRIMARY KEY,
              v TEXT,
              expire_at REAL
            )
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS kv_exp_idx ON kv(expire_at);")
        self._lock = threading.RLock()
        self._prefix = prefix

    def _k(self, k: str) -> str:
        return f"{self._prefix}{k}" if self._prefix else k

    async def get(self, key: str, default: Any = None) -> Any:
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

    async def set(self, key: str, value: Any, *, ttl_s: int | None = None) -> None:
        k = self._k(key)
        with self._lock:
            exp = (time.time() + ttl_s) if ttl_s else None
            v_txt = json.dumps(value)
            self._db.execute(
                "INSERT INTO kv(k,v,expire_at) VALUES(?,?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v, expire_at=excluded.expire_at",
                (k, v_txt, exp),
            )

    async def delete(self, key: str) -> None:
        k = self._k(key)
        with self._lock:
            self._db.execute("DELETE FROM kv WHERE k=?", (k,))

    async def list_append_unique(
        self, key: str, items: list, *, id_key: str = "id", ttl_s: int | None = None
    ) -> list:
        k = self._k(key)
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
            cur = []
            if row and row[0]:
                try:
                    cur = json.loads(row[0])
                except Exception:
                    cur = []
            seen = {x.get(id_key) for x in cur if isinstance(x, dict)}
            cur.extend([x for x in items if isinstance(x, dict) and x.get(id_key) not in seen])
            exp = (time.time() + ttl_s) if ttl_s else None
            self._db.execute(
                "INSERT INTO kv(k,v,expire_at) VALUES(?,?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v, expire_at=excluded.expire_at",
                (k, json.dumps(cur), exp),
            )
            return cur

    async def list_pop_all(self, key: str) -> list:
        k = self._k(key)
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
            self._db.execute("DELETE FROM kv WHERE k=?", (k,))
            if not row or not row[0]:
                return []
            try:
                val = json.loads(row[0])
                return list(val) if isinstance(val, list) else []
            except Exception:
                return []

    # Optional helpers
    async def mget(self, keys: list[str]) -> list[Any]:
        out = []
        for k in keys:
            out.append(await self.get(k))
        return out

    async def mset(self, kv: dict[str, Any], *, ttl_s: int | None = None) -> None:
        for k, v in kv.items():
            await self.set(k, v, ttl_s=ttl_s)

    async def expire(self, key: str, ttl_s: int) -> None:
        k = self._k(key)
        with self._lock:
            self._db.execute("UPDATE kv SET expire_at=? WHERE k=?", (time.time() + ttl_s, k))

    async def purge_expired(self, limit: int = 1000) -> int:
        with self._lock:
            now = time.time()
            # sqlite lacks DELETE .. LIMIT in older versions; do it in two steps
            rows = self._db.execute(
                "SELECT k FROM kv WHERE expire_at IS NOT NULL AND expire_at < ? LIMIT ?",
                (now, limit),
            ).fetchall()
            for (k,) in rows:
                self._db.execute("DELETE FROM kv WHERE k=?", (k,))
            return len(rows)

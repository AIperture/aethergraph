from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

"""
This is not used in the main codebase; only used by async wrapper SqliteDocStore.
"""


class SQLiteDocStoreSync:
    """
    Durable document store on SQLite.

    - Single connection per instance.
    - Thread-safe via RLock.
    - Values are JSON-serialized dicts.
    """

    def __init__(self, path: str):
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._db = sqlite3.connect(
            str(path_obj),
            check_same_thread=False,  # allow multi-thread access (guarded by RLock)
            isolation_level=None,  # autocommit
        )
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id     TEXT PRIMARY KEY,
                data_json  TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._lock = threading.RLock()

    def put(self, doc_id: str, doc: dict[str, Any]) -> None:
        payload = json.dumps(doc, ensure_ascii=False)
        now = time.time()
        # TEMP: tiny backoff to avoid rare SQLite stalls under continuations.
        # This is a hacky workaround.
        # It happens when following conditions align:
        # 1) continuation store using sqlite doc to save
        # 2) the continuation is created under if/else or for loop nodes
        # NOTE: this bug only appears in SQLite and not in other DBs.
        # Remove once we move continuations to Postgres or refactor SQLite usage.
        time.sleep(0.01)
        with self._lock:
            try:
                self._db.execute(
                    """
                    INSERT INTO docs (doc_id, data_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        data_json = excluded.data_json,
                        updated_at = excluded.updated_at
                    """,
                    (doc_id, payload, now),
                )
            except sqlite3.Error as e:
                print("🍓 SQLiteDocStoreSync ERROR during put:", doc_id, repr(e))
                raise

    def get(self, doc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data_json FROM docs WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))

    def list(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT doc_id FROM docs").fetchall()
        return [r[0] for r in rows]

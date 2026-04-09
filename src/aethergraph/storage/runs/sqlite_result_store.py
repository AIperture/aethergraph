from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3
import threading

from aethergraph.contracts.services.runs import RunResultStore
from aethergraph.core.runtime.run_types import RunResult
from aethergraph.storage.runs.result_store import _decode_result, _encode_result


class SQLiteRunResultStoreSync:
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
            CREATE TABLE IF NOT EXISTS run_results (
                run_id      TEXT PRIMARY KEY,
                graph_id    TEXT,
                session_id  TEXT,
                status      TEXT,
                created_at  TEXT,
                updated_at  TEXT,
                data_json   TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_run_results_graph_updated ON run_results(graph_id, updated_at DESC)"
        )
        self._lock = threading.RLock()

    def save(self, run_id: str, result: RunResult) -> None:
        data = _encode_result(result)
        payload = json.dumps(data, ensure_ascii=False)
        with self._lock:
            self._db.execute(
                """
                INSERT INTO run_results (
                    run_id, graph_id, session_id, status, created_at, updated_at, data_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    graph_id=excluded.graph_id,
                    session_id=excluded.session_id,
                    status=excluded.status,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    data_json=excluded.data_json
                """,
                (
                    run_id,
                    result.graph_id,
                    result.session_id,
                    result.status.value,
                    result.created_at.isoformat(),
                    result.updated_at.isoformat(),
                    payload,
                ),
            )

    def get(self, run_id: str) -> RunResult | None:
        with self._lock:
            row = self._db.execute(
                "SELECT data_json FROM run_results WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return _decode_result(json.loads(row[0]))

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))


class SQLiteRunResultStore(RunResultStore):
    def __init__(self, path: str):
        self._sync = SQLiteRunResultStoreSync(path)

    async def save(self, run_id: str, result: RunResult) -> None:
        await asyncio.to_thread(self._sync.save, run_id, result)

    async def get(self, run_id: str) -> RunResult | None:
        return await asyncio.to_thread(self._sync.get, run_id)

    async def delete(self, run_id: str) -> None:
        await asyncio.to_thread(self._sync.delete, run_id)

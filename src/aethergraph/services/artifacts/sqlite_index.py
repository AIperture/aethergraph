# aethergraph/artifacts/index_sqlite.py
from __future__ import annotations
import asyncio
import json, sqlite3
from typing import Optional, Literal, Dict, List
from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.services.artifacts.jsonl_index import JsonlArtifactIndexSync

class SqliteArtifactIndexSync:
    """ SQLite-based artifact index for medium to large scale use cases.
        Suitable for larger scale (millions of artifacts) with indexing.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            uri TEXT NOT NULL,
            kind TEXT NOT NULL,
            bytes INTEGER,
            sha256 TEXT,
            mime TEXT,
            run_id TEXT,
            graph_id TEXT,
            node_id TEXT,
            tool_name TEXT,
            tool_version TEXT,
            created_at TEXT,
            labels TEXT,
            metrics TEXT,
            params TEXT,
            preview_uri TEXT,
            pinned INTEGER DEFAULT 0
        )""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kind ON artifacts(kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_run ON artifacts(run_id)")
        con.commit()
        con.close()

    def upsert(self, a: Artifact) -> None:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        INSERT INTO artifacts
        (artifact_id, uri, kind, bytes, sha256, mime, run_id, graph_id, node_id,
         tool_name, tool_version, created_at, labels, metrics, params, preview_uri, pinned)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(artifact_id) DO UPDATE SET
            uri=excluded.uri, kind=excluded.kind, bytes=excluded.bytes,
            sha256=excluded.sha256, mime=excluded.mime, run_id=excluded.run_id,
            graph_id=excluded.graph_id, node_id=excluded.node_id,
            tool_name=excluded.tool_name, tool_version=excluded.tool_version,
            created_at=excluded.created_at, labels=excluded.labels,
            metrics=excluded.metrics, params=excluded.params,
            preview_uri=excluded.preview_uri, pinned=excluded.pinned
        """, (
            a.artifact_id, a.uri, a.kind, a.bytes, a.sha256, a.mime,
            a.run_id, a.graph_id, a.node_id, a.tool_name, a.tool_version,
            a.created_at,
            json.dumps(a.labels), json.dumps(a.metrics), json.dumps(a.params),
            a.preview_uri, 1 if a.pinned else 0
        ))
        con.commit(); con.close()

    def list_for_run(self, run_id: str) -> List[Artifact]:
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        cur.execute("SELECT * FROM artifacts WHERE run_id=? ORDER BY created_at", (run_id,))
        rows = cur.fetchall(); con.close()
        return [self._row_to_artifact(r) for r in rows]

    def search(self, *, kind: Optional[str]=None, labels: Optional[Dict[str,str]]=None,
               metric: Optional[str]=None, mode: Optional[Literal["max","min"]]=None) -> List[Artifact]:
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        q = "SELECT * FROM artifacts WHERE 1=1"; args=[]
        if kind: q += " AND kind=?"; args.append(kind)
        # naive label filter: all requested label kv must be contained in labels json
        if labels:
            for k,v in labels.items():
                q += " AND json_extract(labels, ?) = ?"
                args += (f'$.{k}', v)
        cur.execute(q, args)
        rows = [self._row_to_artifact(r) for r in cur.fetchall()]
        con.close()
        if metric and mode and rows:
            rows = [r for r in rows if r.metrics and metric in r.metrics]
            reverse = (mode=="max")
            rows.sort(key=lambda a: a.metrics.get(metric, float("-inf")), reverse=reverse)
        return rows

    def best(self, *, kind: str, metric: str, mode: Literal["max","min"], filters: Optional[Dict[str,str]]=None) -> Optional[Artifact]:
        rows = self.search(kind=kind, labels=filters, metric=metric, mode=mode)
        return rows[0] if rows else None

    def pin(self, artifact_id: str, pinned: bool=True) -> None:
        con = sqlite3.connect(self.db_path); cur = con.cursor()
        cur.execute("UPDATE artifacts SET pinned=? WHERE artifact_id=?", (1 if pinned else 0, artifact_id))
        con.commit(); con.close()

    def _row_to_artifact(self, r) -> Artifact:
        (artifact_id, uri, kind, bytes_, sha256, mime, run_id, graph_id, node_id,
         tool_name, tool_version, created_at, labels, metrics, params, preview_uri, pinned) = r
        return Artifact(
            artifact_id=artifact_id, uri=uri, kind=kind, bytes=bytes_, sha256=sha256, mime=mime,
            run_id=run_id, graph_id=graph_id, node_id=node_id, tool_name=tool_name, tool_version=tool_version,
            created_at=created_at, labels=json.loads(labels or "{}"), metrics=json.loads(metrics or "{}"),
            params=json.loads(params or "{}"), preview_uri=preview_uri, pinned=bool(pinned)
        )



class SqliteArtifactIndex:  # implements AsyncArtifactIndex
    def __init__(self, path: str, occurrences_path: str | None = None):
        self._sync = JsonlArtifactIndexSync(path, occurrences_path)

    async def upsert(self, a: Artifact) -> None:
        await asyncio.to_thread(self._sync.upsert, a)

    async def list_for_run(self, run_id: str) -> list[Artifact]:
        return await asyncio.to_thread(self._sync.list_for_run, run_id)

    async def search(self, **kw) -> list[Artifact]:
        return await asyncio.to_thread(self._sync.search, **kw)

    async def best(self, **kw) -> Optional[Artifact]:
        return await asyncio.to_thread(self._sync.best, **kw)

    async def pin(self, artifact_id: str, pinned: bool=True) -> None:
        await asyncio.to_thread(self._sync.pin, artifact_id, pinned)

    async def record_occurrence(self, a: Artifact, extra_labels: dict | None = None):
        await asyncio.to_thread(self._sync.record_occurrence, a, extra_labels)
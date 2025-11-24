import os

from aethergraph.config.config import AppSettings, ContinuationStoreSettings
from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.kv import AsyncKV
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog
from aethergraph.services.continuations.stores.fs_store import FSContinuationStore
from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore
from aethergraph.storage.artifacts.artifact_index_jsonl import JsonlArtifactIndex
from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex
from aethergraph.storage.continuation_store.kvdoc_cont import KVDocContinuationStore
from aethergraph.storage.docstore.fs_doc import FSDocStore
from aethergraph.storage.docstore.sqlite_doc import SqliteDocStore
from aethergraph.storage.eventlog.fs_event import FSEventLog
from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog
from aethergraph.storage.graph_state_store.state_store import GraphStateStoreImpl


def build_artifact_store(cfg: AppSettings) -> AsyncArtifactStore:
    """
    Decide which artifact store backend to use based on AppSettings.storage.artifacts.
    """
    art_cfg = cfg.storage.artifacts
    root = os.path.abspath(cfg.root)

    if art_cfg.backend == "fs":
        from aethergraph.storage.artifacts.fs_cas import FSArtifactStore

        base_dir = os.path.join(root, art_cfg.fs.base_dir)
        return FSArtifactStore(base_dir=base_dir)

    if art_cfg.backend == "s3":
        from aethergraph.storage.artifacts.s3_cas import (
            S3ArtifactStore,  # late import to avoid boto3 dependency if unused
        )

        if not art_cfg.s3.bucket:
            raise ValueError("S3 backend selected, but STORAGE__ARTIFACTS__S3__BUCKET is empty")

        staging_dir = art_cfg.s3.staging_dir
        if not staging_dir:
            staging_dir = os.path.join(root, ".aethergraph_tmp", "artifacts")
        return S3ArtifactStore(
            bucket=art_cfg.s3.bucket,
            prefix=art_cfg.s3.prefix,
            staging_dir=staging_dir,
        )

    raise ValueError(f"Unknown artifacts backend: {art_cfg.backend!r}")


def build_artifact_index(cfg: AppSettings) -> AsyncArtifactIndex:
    idx_cfg = cfg.storage.artifact_index
    root = os.path.abspath(cfg.root)

    if idx_cfg.backend == "jsonl":
        path = os.path.join(root, idx_cfg.jsonl.path)
        occ = (
            os.path.join(root, idx_cfg.jsonl.occurrences_path)
            if idx_cfg.jsonl.occurrences_path
            else None
        )
        return JsonlArtifactIndex(path=path, occurrences_path=occ)

    if idx_cfg.backend == "sqlite":
        path = os.path.join(root, idx_cfg.sqlite.path)
        return SqliteArtifactIndex(path=path)

    raise ValueError(f"Unknown artifact index backend: {idx_cfg.backend!r}")


def build_graph_state_store(cfg: AppSettings) -> GraphStateStoreImpl:
    gs_cfg = cfg.storage.graph_state

    if gs_cfg.backend == "fs":
        base = os.path.join(cfg.root, gs_cfg.fs_root)
        docs = FSDocStore(os.path.join(base, "docs"))
        log = FSEventLog(os.path.join(base, "events"))
    elif gs_cfg.backend == "sqlite":
        db_path = os.path.join(cfg.root, gs_cfg.sqlite_path)
        docs = SqliteDocStore(db_path)
        log = SqliteEventLog(db_path)
    else:
        raise ValueError(f"Unknown graph_state backend: {gs_cfg.backend!r}")

    return GraphStateStoreImpl(doc_store=docs, event_log=log)


def _secret_bytes(secret_key: str) -> bytes:
    # simple default; support hex/env later if needed
    return secret_key.encode("utf-8")


def build_continuation_store(
    cfg: ContinuationStoreSettings,
    *,
    doc_store: DocStore | None = None,
    kv: AsyncKV | None = None,
    event_log: EventLog | None = None,
) -> AsyncContinuationStore:
    """
    Factory for continuation store.

    For backend="kvdoc", doc_store and kv *must* be provided.
    """

    secret = _secret_bytes(cfg.secret_key)

    if cfg.backend == "memory":
        return InMemoryContinuationStore(secret=secret)

    if cfg.backend == "fs":
        # Keep old FS behavior for people who rely on file layout for debugging.
        return FSContinuationStore(root=cfg.root, secret=secret)

    if cfg.backend == "kvdoc":
        if doc_store is None or kv is None:
            raise ValueError("KVDoc continuation backend requires doc_store and kv instances.")
        return KVDocContinuationStore(
            doc_store=doc_store,
            kv=kv,
            event_log=event_log,
            secret=secret,
            namespace=cfg.namespace,
        )

    raise ValueError(f"Unknown continuation backend: {cfg.backend}")

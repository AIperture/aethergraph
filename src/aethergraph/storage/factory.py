import os
from pathlib import Path

from aethergraph.config.config import AppSettings, ContinuationStoreSettings
from aethergraph.contracts.services.continuations import AsyncContinuationStore
from aethergraph.contracts.services.kv import AsyncKV
from aethergraph.contracts.services.memory import HotLog, Indices, Persistence
from aethergraph.contracts.services.state_stores import GraphStateStore
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.doc_store import DocStore
from aethergraph.contracts.storage.event_log import EventLog


def build_doc_store(cfg: AppSettings) -> DocStore:
    """
    Global DocStore factory, used by:
      - Memory persistence (EventLogPersistence)
      - RAG
      - Continuations (if you choose to share it)
      - Anything else that wants "document-ish" JSON blobs.
    """
    root = Path(cfg.root).resolve()
    dc = cfg.storage.docs

    if dc.backend == "sqlite":
        from aethergraph.storage.docstore.sqlite_doc import SqliteDocStore

        path = root / dc.sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteDocStore(path=str(path))

    if dc.backend == "fs":
        from aethergraph.storage.docstore.fs_doc import FSDocStore

        doc_root = root / dc.fs_dir
        doc_root.mkdir(parents=True, exist_ok=True)
        return FSDocStore(root=str(doc_root))

    raise ValueError(f"Unknown DocStore backend: {dc.backend!r}")


def build_event_log(cfg: AppSettings, service_name: str | None = None) -> EventLog | None:
    """
    Global EventLog factory.
    Used by:
      - GraphStateStore (if you want)
      - Memory (EventLogPersistence)
      - Continuations audit (optional)
    """
    root = Path(cfg.root).resolve()
    ec = cfg.storage.eventlog

    if ec.backend == "none":
        return None

    if ec.backend == "sqlite":
        from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog

        # If you use a different DB file per service, you get isolation between services,
        # but lose global querying and may have more files to manage.
        # If you use a single DB file, all services share the same event log table(s).
        path = root / ec.sqlite_path  # You could do: root / f"{service_name}_{ec.sqlite_path}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteEventLog(path=str(path))

    if ec.backend == "fs":
        from aethergraph.storage.eventlog.fs_event import FSEventLog

        ev_root = root / ec.fs_dir if not service_name else root / ec.fs_dir / service_name
        ev_root.mkdir(parents=True, exist_ok=True)
        return FSEventLog(root=str(ev_root))

    raise ValueError(f"Unknown EventLog backend: {ec.backend!r}")


def build_kv_store(cfg: AppSettings, *, extra_prefix: str = "") -> AsyncKV:
    """
    Global KV factory.

    extra_prefix lets subsystems (memory, continuations, etc.) add their own
    namespace on top of the global storage.kv.prefix.
    """
    root = Path(cfg.root).resolve()
    kc = cfg.storage.kv

    full_prefix = f"{kc.prefix}{extra_prefix}"

    if kc.backend == "sqlite":
        from aethergraph.storage.kv.sqlite_kv import SqliteKV

        path = root / kc.sqlite_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return SqliteKV(path=str(path), prefix=full_prefix)

    if kc.backend == "inmem":
        from aethergraph.storage.kv.inmem_kv import InMemoryKV

        return InMemoryKV(prefix=full_prefix)

    raise ValueError(f"Unknown KV backend: {kc.backend!r}")


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
        from aethergraph.storage.artifacts.artifact_index_jsonl import JsonlArtifactIndex

        path = os.path.join(root, idx_cfg.jsonl.path)
        occ = (
            os.path.join(root, idx_cfg.jsonl.occurrences_path)
            if idx_cfg.jsonl.occurrences_path
            else None
        )
        return JsonlArtifactIndex(path=path, occurrences_path=occ)

    if idx_cfg.backend == "sqlite":
        from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex

        path = os.path.join(root, idx_cfg.sqlite.path)
        return SqliteArtifactIndex(path=path)

    raise ValueError(f"Unknown artifact index backend: {idx_cfg.backend!r}")


def build_graph_state_store(cfg: AppSettings) -> GraphStateStore:
    from aethergraph.storage.graph_state_store.state_store import GraphStateStoreImpl

    gs_cfg = cfg.storage.graph_state

    if gs_cfg.backend == "fs":
        from aethergraph.storage.docstore.fs_doc import FSDocStore
        from aethergraph.storage.eventlog.fs_event import FSEventLog

        base = os.path.join(cfg.root, gs_cfg.fs_root)
        docs = FSDocStore(os.path.join(base, "docs"))
        log = FSEventLog(os.path.join(base, "events"))
    elif gs_cfg.backend == "sqlite":
        from aethergraph.storage.docstore.sqlite_doc import SqliteDocStore
        from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog

        db_path = os.path.join(cfg.root, gs_cfg.sqlite_path)
        docs = SqliteDocStore(db_path)
        log = SqliteEventLog(db_path)
    else:
        raise ValueError(f"Unknown graph_state backend: {gs_cfg.backend!r}")

    return GraphStateStoreImpl(doc_store=docs, event_log=log)


def _secret_bytes(secret_key: str) -> bytes:
    # simple default; support hex/env later if needed
    return secret_key.encode("utf-8")


def _build_kvdoc_cont_store(
    root: Path,
    cfg: ContinuationStoreSettings,
    secret: bytes,
) -> AsyncContinuationStore:
    kvdoc = cfg.kvdoc
    from aethergraph.storage.continuation_store.kvdoc_cont import KVDocContinuationStore

    # ---- DocStore ----
    if kvdoc.doc_store_backend == "sqlite":
        from aethergraph.storage.docstore.sqlite_doc import SqliteDocStore

        doc_path = root / kvdoc.sqlite_doc_store_path
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_store: DocStore = SqliteDocStore(path=str(doc_path))
    elif kvdoc.doc_store_backend == "fs":
        from aethergraph.storage.docstore.fs_doc import FSDocStore

        doc_dir = root / kvdoc.fs_doc_store_dir
        doc_dir.mkdir(parents=True, exist_ok=True)
        doc_store = FSDocStore(root=str(doc_dir))
    else:
        raise ValueError(f"Unknown doc_store_backend: {kvdoc.doc_store_backend}")

    # ---- KV ----
    if kvdoc.kv_backend == "sqlite":
        from aethergraph.storage.kv.sqlite_kv import SqliteKV

        kv_path = root / kvdoc.sqlite_kv_path
        kv_path.parent.mkdir(parents=True, exist_ok=True)
        kv: AsyncKV = SqliteKV(path=str(kv_path), prefix=f"{cfg.namespace}:")
    elif kvdoc.kv_backend == "inmem":
        from aethergraph.storage.kv.inmem_kv import InMemoryKV

        kv = InMemoryKV(prefix=f"{cfg.namespace}:")
    else:
        raise ValueError(f"Unknown kv_backend: {kvdoc.kv_backend}")

    # ---- EventLog (optional) ----
    event_log: EventLog | None
    if kvdoc.eventlog_backend == "none":
        event_log = None
    elif kvdoc.eventlog_backend == "sqlite":
        from aethergraph.storage.eventlog.sqlite_event import SqliteEventLog

        ev_path = root / kvdoc.sqlite_eventlog_path
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        event_log = SqliteEventLog(path=str(ev_path))
    elif kvdoc.eventlog_backend == "fs":
        from aethergraph.storage.eventlog.fs_event import FSEventLog

        ev_dir = root / kvdoc.fs_eventlog_dir
        ev_dir.mkdir(parents=True, exist_ok=True)
        event_log = FSEventLog(root=str(ev_dir))
    else:
        raise ValueError(f"Unknown eventlog_backend: {kvdoc.eventlog_backend}")

    return KVDocContinuationStore(
        doc_store=doc_store,
        kv=kv,
        event_log=event_log,
        secret=secret,
        namespace=cfg.namespace,
    )


def build_continuation_store(cfg: AppSettings) -> AsyncContinuationStore:
    """
    High-level factory used by your runtime builder.

    Mirrors `build_artifact_store(cfg)` in style.
    """
    root = Path(cfg.root).resolve()
    cont_cfg: ContinuationStoreSettings = cfg.storage.continuation
    secret = _secret_bytes(cont_cfg.secret_key)

    if cont_cfg.backend == "memory":
        from aethergraph.services.continuations.stores.inmem_store import InMemoryContinuationStore

        return InMemoryContinuationStore(secret=secret)

    if cont_cfg.backend == "fs":
        from aethergraph.services.continuations.stores.fs_store import FSContinuationStore

        # Keep old FS behavior for people who rely on on-disk layout.
        fs_root = root / cont_cfg.fs.root
        fs_root.parent.mkdir(parents=True, exist_ok=True)
        return FSContinuationStore(root=fs_root, secret=secret)

    if cont_cfg.backend == "kvdoc":
        return _build_kvdoc_cont_store(root, cont_cfg, secret)

    raise ValueError(f"Unknown continuation backend: {cont_cfg.backend}")


def build_vector_index(cfg: AppSettings):
    """
    Build a VectorIndex based on cfg.storage.vector_index.
    """
    vcfg = cfg.storage.vector_index
    root = os.path.abspath(cfg.root)

    if vcfg.backend == "sqlite":
        from aethergraph.storage.vector_index.sqlite_index import SQLiteVectorIndex

        index_root = os.path.join(root, vcfg.sqlite.dir)
        return SQLiteVectorIndex(root=index_root)

    if vcfg.backend == "faiss":
        from aethergraph.storage.vector_index.faiss_index import FAISSVectorIndex

        index_root = os.path.join(root, vcfg.faiss.dir)
        return FAISSVectorIndex(root=index_root, dim=vcfg.faiss.dim)

    if vcfg.backend == "chroma":
        try:
            import chromadb
        except ImportError as e:
            chromadb = None  # type: ignore
            raise RuntimeError("Chroma backend requires `chromadb` to be installed.") from e
        from aethergraph.storage.vector_index.chroma_index import ChromaVectorIndex

        if chromadb is None:
            raise RuntimeError(
                "Chroma backend selected, but `chromadb` is not installed. "
                "Install it with `pip install chromadb`."
            )
        persist_dir = os.path.join(root, vcfg.chroma.persist_dir)
        client = chromadb.PersistentClient(path=persist_dir)
        return ChromaVectorIndex(
            client=client,
            collection_prefix=vcfg.chroma.collection_prefix,
        )

    raise ValueError(f"Unknown vector index backend: {vcfg.backend!r}")


def build_memory_persistence(cfg: AppSettings) -> Persistence:
    mp = cfg.storage.memory.persistence
    root = cfg.root

    if mp.backend == "fs":
        from aethergraph.storage.memory.fs_persist import FSPersistence

        return FSPersistence(base_dir=root)

    if mp.backend == "eventlog":
        from aethergraph.storage.memory.event_persist import EventLogPersistence

        docs = build_doc_store(cfg)
        log = build_event_log(cfg)
        if log is None:
            raise ValueError("memory.persistence.backend=eventlog requires a non-none EventLog")
        return EventLogPersistence(
            log=log,
            docs=docs,
            uri_prefix=mp.uri_prefix,
        )

    raise ValueError(f"Unknown memory persistence backend: {mp.backend!r}")


def build_memory_hotlog(cfg: AppSettings) -> HotLog:
    from aethergraph.storage.memory.hotlog import KVHotLog

    kv = build_kv_store(cfg, extra_prefix="mem:hot:")
    return KVHotLog(kv=kv)


def build_memory_indices(cfg: AppSettings) -> Indices:
    from aethergraph.storage.memory.indices import KVIndices

    kv = build_kv_store(cfg, extra_prefix="mem:idx:")
    return KVIndices(kv=kv, hot_ttl_s=cfg.storage.memory.indices.ttl_s)

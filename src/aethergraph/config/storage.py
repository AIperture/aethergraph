from typing import Literal

from pydantic import BaseModel, Field

# --- Per-backend settings ---


# --- Artifact storage backends ---
class FSArtifactStoreSettings(BaseModel):
    # Interpreted relative to AppSettings.root in the factory
    base_dir: str = "artifacts"  # => <root>/artifacts by default


class S3ArtifactStoreSettings(BaseModel):
    bucket: str = ""  # must be set via env when backend="s3"
    prefix: str = "artifacts"  # e.g. "aethergraph/artifacts"
    # local temp dir; if empty, factory can default to something under root
    staging_dir: str = "./.aethergraph_tmp/artifacts"


class ArtifactStorageSettings(BaseModel):
    # which backend to use for artifacts
    backend: Literal["fs", "s3"] = "fs"

    fs: FSArtifactStoreSettings = FSArtifactStoreSettings()
    s3: S3ArtifactStoreSettings = S3ArtifactStoreSettings()


class JsonlArtifactIndexSettings(BaseModel):
    # Relative to AppSettings.root; we’ll join in the factory
    path: str = "artifacts/index.jsonl"
    occurrences_path: str | None = None  # default: <stem>_occurrences.jsonl


class SqliteArtifactIndexSettings(BaseModel):
    path: str = "artifacts/index.sqlite"


class ArtifactIndexSettings(BaseModel):
    backend: Literal["jsonl", "sqlite"] = "jsonl"
    jsonl: JsonlArtifactIndexSettings = JsonlArtifactIndexSettings()
    sqlite: SqliteArtifactIndexSettings = SqliteArtifactIndexSettings()


# --- Graph State Storage ---
class GraphStateStorageSettings(BaseModel):
    backend: Literal["fs", "sqlite"] = "sqlite"

    # FS backend
    fs_root: str = "graph_state"  # under AppSettings.root
    # SQLite backend
    sqlite_path: str = "graph_state/graph_state.db"  # relative to AppSettings.root


# --- Continuation Store ---
class KVDocContinuationStoreSettings(BaseModel):
    # DocStore backend type for continuations
    doc_store_backend: Literal["sqlite", "fs"] = "sqlite"
    sqlite_doc_store_path: str = "continuations/cont_doc_store.db"  # relative to AppSettings.root
    fs_doc_store_dir: str = "continuations/cont_doc_store"  # relative to AppSettings.root

    # AsyncKV backend type for token + correlator indexes
    kv_backend: Literal["sqlite", "inmem"] = "sqlite"
    sqlite_kv_path: str = "continuations/cont_kv_store.db"  # relative to AppSettings.root

    # EventLog backend for continuation audit (optional)
    eventlog_backend: Literal["none", "sqlite", "fs"] = "fs"
    sqlite_eventlog_path: str = "continuations/cont_events.db"  # relative to AppSettings.root
    fs_eventlog_dir: str = "continuations/cont_events"  # relative to AppSettings.root


class FSContinuationStoreSettings(BaseModel):
    # Where to store the old filesystem layout (runs/index/...).
    # Interpreted relative to AppSettings.root.
    root: str = "continuations/cont_fs_store"


class MemoryContinuationStoreSettings(BaseModel):
    # Placeholder for future options, e.g. max entries, debug flags, etc.
    enabled: bool = True


class ContinuationStoreSettings(BaseModel):
    # Which backend to use:
    #   - "fs":    keep existing FSContinuationStore
    #   - "kvdoc": KVDocContinuationStore (DocStore + AsyncKV + EventLog)
    #   - "memory": in-memory (for tests/dev)
    backend: Literal["fs", "kvdoc", "memory"] = "kvdoc"

    # Namespacing for DocStore ids / KV keys
    namespace: str = "cont"

    # Secret for HMAC token generation; override via env.
    secret_key: str = Field(
        default="change-me",
        description="Secret key for continuation HMAC tokens; set via AETHERGRAPH_CONT__SECRET_KEY.",
    )

    fs: FSContinuationStoreSettings = FSContinuationStoreSettings()
    kvdoc: KVDocContinuationStoreSettings = KVDocContinuationStoreSettings()
    memory: MemoryContinuationStoreSettings = MemoryContinuationStoreSettings()


class StorageSettings(BaseModel):
    artifacts: ArtifactStorageSettings = ArtifactStorageSettings()
    artifact_index: ArtifactIndexSettings = ArtifactIndexSettings()
    graph_state: GraphStateStorageSettings = GraphStateStorageSettings()
    continuation: ContinuationStoreSettings = ContinuationStoreSettings()

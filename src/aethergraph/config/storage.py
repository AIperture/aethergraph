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
    backend: Literal["fs", "sqlite"] = "fs"

    # FS backend
    fs_root: str = "graph_state"  # under AppSettings.root
    # SQLite backend
    sqlite_path: str = "graph_state.db"  # relative to AppSettings.root


# --- Continuation Store ---
class ContinuationStoreSettings(BaseModel):
    # Which backend to use:
    #   - "fs": keep existing FSContinuationStore
    #   - "kvdoc": KVDocContinuationStore (DocStore + AsyncKV + EventLog)
    #   - "memory": in-memory (for tests/dev)
    backend: Literal["fs", "kvdoc", "memory"] = "kvdoc"

    # Root directory for FS backend (and default sqlite paths if you derive them)
    root: str = "./aethergraph_data/cont"

    # Namespacing for KV/Doc keys / ids
    namespace: str = "cont"

    # Secret for HMAC token generation; override via env.
    secret_key: str = Field(
        default="change-me",
        description="Secret key for continuation HMAC tokens; set via AETHERGRAPH_CONT__SECRET_KEY.",
    )


class StorageSettings(BaseModel):
    artifacts: ArtifactStorageSettings = ArtifactStorageSettings()
    artifact_index: ArtifactIndexSettings = ArtifactIndexSettings()
    graph_state: GraphStateStorageSettings = GraphStateStorageSettings()
    continuation: ContinuationStoreSettings = ContinuationStoreSettings()

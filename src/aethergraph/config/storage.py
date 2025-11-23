from typing import Literal

from pydantic import BaseModel

# --- Per-backend settings ---


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


class StorageSettings(BaseModel):
    artifacts: ArtifactStorageSettings = ArtifactStorageSettings()
    artifact_index: ArtifactIndexSettings = ArtifactIndexSettings()

import os

from aethergraph.config.config import AppSettings
from aethergraph.contracts.storage.artifact_index import AsyncArtifactIndex
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.storage.artifacts.artifact_index_jsonl import JsonlArtifactIndex
from aethergraph.storage.artifacts.artifact_index_sqlite import SqliteArtifactIndex


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

import os

from aethergraph.storage.artifacts.cas_store import CASArtifactStore
from aethergraph.storage.blob.fs_blob import FSBlobStore


class FSArtifactStore(CASArtifactStore):
    # Initialize with a base directory for storing artifacts

    def __init__(self, base_dir: str):
        base_dir = os.path.abspath(base_dir)
        blob = FSBlobStore(os.path.join(base_dir, "blobs"))
        staging_dir = os.path.join(base_dir, "staging")
        super().__init__(blob=blob, staging_dir=staging_dir)

    # TODO: Add any FS-specific optimizations if needed
    # Optionally override load_artifact_dir to return actual local dir path if uri is file://cas/trees/...
    # and implement FS-only "pretty" symlinks.

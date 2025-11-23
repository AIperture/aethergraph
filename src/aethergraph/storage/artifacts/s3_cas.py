from aethergraph.storage.artifacts.cas_store import CASArtifactStore
from aethergraph.storage.blob.s3_blob import S3BlobStore


class S3ArtifactStore(CASArtifactStore):
    # Initialize with S3 bucket and optional prefix for storing artifacts
    def __init__(self, bucket: str, prefix: str, staging_dir: str):
        blob = S3BlobStore(bucket=bucket, prefix=prefix)
        super().__init__(blob=blob, staging_dir=staging_dir)

    # TODO: Optionally add any S3-specific optimizations if needed
    # - parse tree_sha from uri
    # - download files listed in manifest.json into a local temp dir
    # - return that path

from typing import Protocol

"""
Blob store interface for storing and retrieving binary large objects (blobs).
Typical implementations include:
- FSBlobStore: File system-based blob store for persistent storage
- S3BlobStore: Amazon S3-based blob store for cloud storage

This is usually used for storing large binary data such as images, audio, or other media files.
For example:
- Artifact Store for saving generated images or files
"""


class BlobStore(Protocol):
    @property
    def base_uri(self) -> str:  # e.g. file:///..., s3://bucket/prefix
        ...

    async def put_bytes(
        self,
        data: bytes,
        *,
        key: str | None = None,
        ext: str | None = None,
        mime: str | None = None,
    ) -> str:
        """Store bytes under an optional key; return full blob URI."""
        ...

    async def put_file(
        self,
        path: str,
        *,
        key: str | None = None,
        mime: str | None = None,
    ) -> str:
        """Store a local file; return full blob URI."""
        ...

    async def load_bytes(self, uri: str) -> bytes: ...

    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str: ...

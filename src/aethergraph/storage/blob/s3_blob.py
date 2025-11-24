from __future__ import annotations

import os

import boto3
from botocore.config import Config

from aethergraph.contracts.storage.blob_store import BlobStore
from aethergraph.services.artifacts.utils import to_thread

"""
S3BlobStore: BlobStore implementation using AWS S3 as the backend.
NOTE: This is a stub implementation; not fully tested.
"""


class S3BlobStore(BlobStore):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")

        # Create S3 client with default config; can be customized as needed
        self.s3_client = boto3.client(
            "s3", config=Config(signature_version="s3v4", max_pool_connections=50)
        )

    @property
    def base_uri(self) -> str:
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix}"
        return f"s3://{self.bucket}"

    def _resolve_key(self, key: str | None, ext: str | None) -> str:
        if key is None:
            import uuid

            key = uuid.uuid4().hex + (ext or "")
        if self.prefix:
            return f"{self.prefix}/{key.lstrip('/')}"
        return key.lstrip("/")

    async def put_bytes(
        self,
        data: bytes,
        *,
        key: str | None = None,
        ext: str | None = None,
        mime: str | None = None,
    ) -> str:
        key = self._resolve_key(key, ext)

        def _upload():
            extra = {}
            if mime:
                extra["ContentType"] = mime
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                **extra,
            )
            return f"s3://{self.bucket}/{key}"

        return await to_thread(_upload)

    async def put_file(
        self,
        path: str,
        *,
        key: str | None = None,
        mime: str | None = None,
    ) -> str:
        ext = os.path.splitext(path)[1]
        key = self._resolve_key(key, ext)

        def _upload_file():
            extra = {}
            if mime:
                extra["ContentType"] = mime
            self._client.upload_file(
                Filename=os.path.abspath(path),
                Bucket=self.bucket,
                Key=key,
                ExtraArgs=extra or None,
            )
            return f"s3://{self.bucket}/{key}"

        return await to_thread(_upload_file)

    async def load_bytes(self, uri: str) -> bytes:
        # assume s3://bucket/prefix...
        # you can parse, or just compute key from uri by stripping base_uri
        if uri.startswith("s3://"):
            _, rest = uri.split("://", 1)
            bucket, key = rest.split("/", 1)
        else:
            bucket = self.bucket
            key = uri

        def _download():
            obj = self._client.get_object(Bucket=bucket, Key=key)
            return obj["Body"].read()

        return await to_thread(_download)

    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        data = await self.load_bytes(uri)
        return data.decode(encoding, errors)

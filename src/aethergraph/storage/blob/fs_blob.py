from __future__ import annotations

import os
import shutil

from aethergraph.contracts.storage.blob_store import BlobStore
from aethergraph.storage.fs_utils import _from_uri_or_path, _to_file_uri, to_thread


class FSBlobStore(BlobStore):
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    @property
    def base_uri(self) -> str:
        return _to_file_uri(self.base_dir)

    def _resolve_key(self, key: str | None, ext: str | None) -> str:
        if key is None:
            # fall back to some random-ish name under "blobs/"
            import uuid

            name = uuid.uuid4().hex + (ext or "")
            key = os.path.join("blobs", name)
        return key

    async def put_bytes(
        self,
        data: bytes,
        *,
        key: str | None = None,
        ext: str | None = None,
        mime: str | None = None,
    ) -> str:
        key = self._resolve_key(key, ext)
        path = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def _write():
            with open(path, "wb") as f:
                f.write(data)
            return _to_file_uri(path)

        return await to_thread(_write)

    async def put_file(
        self,
        path: str,
        *,
        key: str | None = None,
        mime: str | None = None,
    ) -> str:
        ext = os.path.splitext(path)[1]
        key = self._resolve_key(key, ext)
        dst = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        def _move():
            shutil.move(os.path.abspath(path), dst)
            return _to_file_uri(dst)

        return await to_thread(_move)

    async def load_bytes(self, uri: str) -> bytes:
        path = _from_uri_or_path(uri)

        def _read():
            with open(path, "rb") as f:
                return f.read()

        return await to_thread(_read)

    async def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        data = await self.load_bytes(uri)
        return data.decode(encoding, errors)

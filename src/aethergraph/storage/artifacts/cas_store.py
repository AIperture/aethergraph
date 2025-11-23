from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO

from aethergraph.contracts.services.artifacts import Artifact
from aethergraph.contracts.storage.artifact_store import AsyncArtifactStore
from aethergraph.contracts.storage.blob_store import BlobStore

from .utils import (
    _now_iso,
    _sha256_file,
    _tree_manifest_and_hash,
    to_thread,
)

logger = logging.getLogger("aethergraph.services.artifacts.cas_store")


class CASArtifactStore(AsyncArtifactStore):
    """
    Content-addressed artifact store built on top of a BlobStore.

    - Uses local staging_dir for temp files/dirs.
    - Stores blobs via BlobStore with keys derived from SHA-256 hashes.
    - Persists minimal manifest/metadata as blobs too (for directories).
    """

    def __init__(self, blob: BlobStore, staging_dir: str):
        self._blob = blob
        self._staging_dir = os.path.abspath(staging_dir)
        os.makedirs(self._staging_dir, exist_ok=True)
        self.last_artifact: Artifact | None = None

    @property
    def base_uri(self) -> str:
        return self._blob.base_uri

    # ---------- staging utils ----------
    async def plan_staging_path(self, planned_ext: str = "") -> str:
        def _mk():
            fd, p = tempfile.mkstemp(suffix=planned_ext, dir=self._staging_dir)
            os.close(fd)
            return p

        return await to_thread(_mk)

    async def plan_staging_dir(self, suffix: str = "") -> str:
        def _mkd():
            return tempfile.mkdtemp(prefix="dir_", suffix=suffix, dir=self._staging_dir)

        return await to_thread(_mkd)

    # ---------- basic save / ingest ----------
    async def save_file(
        self,
        *,
        path: str,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        suggested_uri: str | None = None,  # NOTE: only metadata / pretty; impl may ignore
        pin: bool = False,
        labels: dict | None = None,
        metrics: dict | None = None,
        preview_uri: str | None = None,  # NOTE: only metadata / pretty; impl may ignore
    ) -> Artifact:
        sha, nbytes = await to_thread(_sha256_file, path)
        ext = os.path.splitext(path)[1]
        key = os.path.join("cas", "blobs", f"{sha}{ext}")

        blob_uri = await self._blob.put_file(path, key=key, mime=None)

        a = Artifact(
            artifact_id=sha,
            uri=blob_uri,
            kind=kind,
            bytes=nbytes,
            sha256=sha,
            mime=None,  # callers can fill in if desired
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            tool_name=tool_name,
            tool_version=tool_version,
            created_at=_now_iso(),
            labels=labels or {},
            metrics=metrics or {},
            preview_uri=preview_uri,
            pinned=pin,
        )
        self.last_artifact = a
        return a

    # ---------- streaming writer ----------
    @asynccontextmanager
    async def open_writer(
        self,
        *,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        planned_ext: str | None = None,
        pin: bool = False,
    ) -> AsyncIterator[Any]:
        staged_path = await self.plan_staging_path(planned_ext or "")

        class _Writer:
            """Helper class for streaming writes to a temp file."""

            def __init__(self, path: str, f: BinaryIO):
                self.tmp_path = path
                self._f = f
                self._labels: dict[str, str] = {}
                self._metrics: dict[str, float] = {}
                self.artifact: Artifact | None = None  # filled after finalize

            def write(self, chunk: bytes) -> None:
                self._f.write(chunk)

            def add_labels(self, labels: dict[str, str]) -> None:
                self._labels.update(labels or {})

            def add_metrics(self, metrics: dict[str, float]) -> None:
                self._metrics.update(metrics or {})

        writer: _Writer | None = None

        try:
            # Ruff-friendly: file is opened via a context manager, and kept
            # open for the duration of the user’s writes.
            with open(staged_path, "wb") as f:
                writer = _Writer(staged_path, f)
                # Yield to caller; they can await inside and call writer.write(...)
                yield writer
            # <-- file is closed here when the with-block exits

            # Now ingest the staged file into CAS and create the Artifact
            if writer is not None:
                a = await self.ingest_staged_file(
                    staged_path=staged_path,
                    kind=kind,
                    run_id=run_id,
                    graph_id=graph_id,
                    node_id=node_id,
                    tool_name=tool_name,
                    tool_version=tool_version,
                    pin=pin,
                    labels=writer._labels,
                    metrics=writer._metrics,
                )
                writer.artifact = a

        except Exception:
            # Best-effort cleanup of staged file on error
            try:
                if os.path.exists(staged_path):
                    os.remove(staged_path)
            finally:
                raise

    async def ingest_staged_file(
        self,
        *,
        staged_path: str,
        kind: str,
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        pin: bool = False,
        labels: dict | None = None,
        metrics: dict | None = None,
        preview_uri: str | None = None,
        suggested_uri: str | None = None,
    ) -> Artifact:
        # just delegate to save_file (same semantics)
        a = await self.save_file(
            path=staged_path,
            kind=kind,
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            tool_name=tool_name,
            tool_version=tool_version,
            suggested_uri=suggested_uri,
            pin=pin,
            labels=labels,
            metrics=metrics,
            preview_uri=preview_uri,
        )
        try:
            os.remove(staged_path)
        except Exception:
            logger.warning("ingest_staged_file: failed to delete staged file %s", staged_path)
        return a

    async def ingest_directory(
        self,
        *,
        staged_dir: str,
        kind: str = "dataset",
        run_id: str,
        graph_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        index_children: bool = False,  # TODO: use later for per-file artifacts
        pin: bool = False,
        labels: dict | None = None,
        metrics: dict | None = None,
        suggested_uri: str | None = None,
        archive: bool = False,
        archive_name: str = "bundle.tar.gz",
        cleanup: bool = True,
        store: str | None = None,  # "archive" | "manifest"
    ) -> Artifact:
        if not os.path.isdir(staged_dir):
            raise ValueError(f"ingest_directory: not a directory: {staged_dir}")

        if store is None:
            store = "archive" if archive else "manifest"

        manifest_entries, tree_sha = await to_thread(
            _tree_manifest_and_hash, staged_dir, include, exclude
        )

        # manifest blob
        manifest_key = os.path.join("cas", "trees", tree_sha, "manifest.json")

        def _dump_manifest() -> bytes:
            return json.dumps(
                {
                    "files": manifest_entries,
                    "created_at": _now_iso(),
                    "tool_name": tool_name,
                    "tool_version": tool_version,
                },
                indent=2,
            ).encode("utf-8")

        # manifest URI -> use it in future if needed
        _ = await self._blob.put_bytes(
            _dump_manifest(),
            key=manifest_key,
            ext=".json",
            mime="application/json",
        )

        archive_uri: str | None = None
        if store == "archive":
            # build tar.gz locally, then upload
            archive_path = os.path.join(self._staging_dir, f"{tree_sha}.tar.gz")

            def _make_tar():
                import tarfile

                with tarfile.open(archive_path, mode="w:gz") as tar:
                    for e in sorted(manifest_entries, key=lambda x: x["path"]):
                        abs_file = os.path.join(staged_dir, e["path"])
                        tar.add(abs_file, arcname=e["path"])
                return archive_path

            archive_path = await to_thread(_make_tar)
            archive_key = os.path.join("cas", "trees", tree_sha, archive_name)
            archive_uri = await self._blob.put_file(
                archive_path,
                key=archive_key,
                mime="application/gzip",
            )

        elif store == "manifest":
            if cleanup:
                # we will delete staged_dir; only OK if user accepts that artifacts
                # are now represented by manifest (+ optional archive)
                pass
        else:
            raise ValueError(f"unknown store mode: {store}")

        # Directory "handle" URI: base_uri + prefix
        dir_prefix = os.path.join("cas", "trees", tree_sha)
        # NOTE: we don't require an actual object at dir_prefix; it's a logical handle.
        dir_uri = self.base_uri.rstrip("/") + "/" + dir_prefix.replace(os.sep, "/")

        total_bytes = sum(e["bytes"] for e in manifest_entries)
        a = Artifact(
            artifact_id=tree_sha,
            uri=dir_uri,
            kind=kind,
            bytes=total_bytes,
            sha256=tree_sha,
            mime="application/vnd.aethergraph.bundle+dir",
            run_id=run_id,
            graph_id=graph_id,
            node_id=node_id,
            tool_name=tool_name,
            tool_version=tool_version,
            created_at=_now_iso(),
            labels=labels or {},
            metrics=metrics or {},
            preview_uri=archive_uri,
            pinned=pin,
        )
        self.last_artifact = a

        if cleanup:
            try:
                shutil.rmtree(staged_dir, ignore_errors=True)
            except Exception:
                logger.warning("ingest_directory: failed to cleanup staged dir %s", staged_dir)

        return a

    # ---------- load ----------
    async def load_bytes(self, uri):
        return await self._blob.load_bytes(uri)

    async def load_text(self, uri: str, *, encoding: str = "utf-8", errors: str = "strict") -> str:
        return await self._blob.load_text(uri, encoding=encoding, errors=errors)

    async def load_artifact_bytes(self, uri: str) -> bytes:
        return await self._blob.load_bytes(uri)

    async def load_artifact_dir(self, uri):
        """
        Normalize a directory artifact to a local path.

        FS backend can simply return the directory; S3 backend
        will download files described by manifest into a temp dir.
        For now, implement generic: if it's already a file:// path,
        just return as-is; otherwise, ArtifactFacade can add a helper
        `as_local_dir(artifact)` that handles S3 download.
        """
        return uri

    async def load_artifact(self, uri):
        # Compatibility: if direcotry URI, return as-is, else load blob content
        if uri.endswith("/"):
            # directory handle URI
            return await self.load_artifact_dir(uri)
        # else, blob URI
        return await self._blob.load_bytes(uri)

    # ---------- cleanup ----------
    async def cleanup_tmp(self, max_age_hours: int = 24) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).timestamp()

        def _cleanup():
            for p in Path(self._staging_dir).rglob("*"):
                try:
                    age_h = (now - p.stat().st_mtime) / 3600.0
                    if age_h > max_age_hours:
                        if p.is_file():
                            p.unlink(missing_ok=True)
                        else:
                            shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass

        await to_thread(_cleanup)

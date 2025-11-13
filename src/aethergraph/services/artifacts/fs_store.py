import asyncio
from contextlib import contextmanager
import datetime
import json
import mimetypes
from nturl2path import url2pathname
from typing import Optional, Any, BinaryIO, Union
import os
import shutil
import tempfile
from urllib.parse import unquote, urlparse
from pathlib import Path
from urllib.request import url2pathname
from urllib.parse import urlparse, unquote

from .utils import to_thread, _now_iso, _sha256_file, _content_addr_path, _content_addr_dir_path, _tree_manifest_and_hash, _write_json, _maybe_cleanup_tmp_parent 
from aethergraph.contracts.services.artifacts import Artifact

def _to_file_uri(path_str: str) -> str:
    """Canonical RFC-8089 file URI (file:///C:/..., forward slashes)."""
    return Path(path_str).resolve().as_uri()

def _from_uri_or_path(s: str) -> Path:
    """Robustly turn a file:// URI or plain path into a local Path."""
    if "://" not in s:
        return Path(s)
    u = urlparse(s)
    if (u.scheme or "").lower() != "file":
        raise ValueError(f"Unsupported URI scheme: {u.scheme}")
    if u.netloc:
        raw = f"//{u.netloc}{u.path}"   # UNC: file://server/share/...
    else:
        raw = u.path                    # Local drive: file:///C:/...
    return Path(url2pathname(unquote(raw)))


def _normalize_pretty_path(base_dir_for_pretty: str, suggested_uri: str) -> str:
    """ Normalize a suggested_uri into a local filesystem path for pretty linking. 
    Args:
        base_dir_for_pretty (str): Base directory to resolve relative paths against.
        suggested_uri (str): The suggested URI, which may be a file:// URI or a relative/absolute path.
    Returns:
        str: The normalized local filesystem path.
        
    Example:
        - suggested_uri = "file://./outputs/my_artifact.txt" -> "./outputs/my_artifact.txt
        - suggested_uri = "./outputs/my_artifact.txt" -> "./outputs/my_artifact.txt
        - suggested_uri = "/var/data/my_artifact.txt" -> "/var/data/my_artifact.txt
    NOTE: 
        Only used for local filesystem paths. For other URI schemes, additional handling would be needed.
    """
    p = _from_uri_or_path(suggested_uri)
    if not p.is_absolute():
        p = Path(base_dir_for_pretty) / p
    return str(p.resolve())


class _Writer:
    """ Helper class for streaming writes to a temp file. """
    def __init__(self, tmp_dir: str, planned_ext: Optional[str]):
        self.tmp_dir = tmp_dir
        suffix = planned_ext or ""
        file_dir, self.tmp_path = tempfile.mkstemp(suffix=suffix, dir=tmp_dir)
        os.close(file_dir)
        self._f = open(self.tmp_path, "wb")
        self._labels = {}; self._metrics = {}

    def write(self, chunk: bytes): 
        self._f.write(chunk)

    def add_labels(self, labels: dict): 
        self._labels.update(labels or {})
    
    def add_metrics(self, metrics: dict): 
        self._metrics.update(metrics or {})

    def close(self): 
        if not self._f.closed: 
            self._f.close()


class _Reader:
    """ Helper class for reading from a file. """
    def __init__(self, path: str, f: BinaryIO):
        self._path, self._f = path, f
    def read(self, n: int = -1) -> bytes:
        return self._f.read(n)
    def as_local_path(self) -> str:
        return self._path

class FileArtifactStoreSync:
    """
    Synchronous file-based artifact store.
    """ 
    def __init__(self, base_dir: str):
        # base directory for content-addressed storage
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

        # temporary staging area for in-progress writes
        self._tmp_root = os.path.join(self.base_dir, "_tmp")
        os.makedirs(self._tmp_root, exist_ok=True)

        self.last_artifact: Optional[Artifact] = None

    @property
    def base_uri(self) -> str:
        return _to_file_uri(self.base_dir)
    
    def tmp_path(self, suffix: str = "") -> str:
        """ Return a temporary path for external tools to write to. """
        os.makedirs(self._tmp_root, exist_ok=True)
        fd, p = tempfile.mkstemp(suffix=suffix, dir=self._tmp_root)
        os.close(fd)
        return p

    def save_file(
            self, path: str, *, kind: str, run_id: str, graph_id: str, node_id: str,
            tool_name: str, tool_version: str, suggested_uri: Optional[str] = None,
            pin: bool = False, labels: Optional[dict] = None, metrics: Optional[dict] = None,
            preview_uri: Optional[str] = None, cleanup: bool = True
    ) -> Artifact:
        """
        Save a file into content-addressed storage and return an Artifact record.   
        Args:
            path (str): The file path to save.
            kind (str): The kind of artifact.
            run_id (str): The run ID.
            graph_id (str): The graph ID.
            node_id (str): The node ID.
            tool_name (str): The tool name.
            tool_version (str): The tool version.
            suggested_uri (Optional[str]): A suggested URI for the artifact.
            pin (bool): Whether to pin the artifact.
            labels (Optional[dict]): Labels to attach to the artifact.
            metrics (Optional[dict]): Metrics to attach to the artifact.
            preview_uri (Optional[str]): A preview URI for the artifact.

        Returns:
            Artifact: The created Artifact object.
        
            It computes the SHA-256 hash of the file, moves it into a content-addressed storage
            structure, and optionally creates a "pretty" symlink if a suggested URI is provided. 
        """
        sha, nbytes = _sha256_file(path)  # compute hash + size
        ext = os.path.splitext(path)[1]
        target = _content_addr_path(self.base_dir, sha, ext)

        if not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)

            src_path = os.path.abspath(path)
            src_parent = os.path.dirname(src_path)

            if cleanup:
                shutil.move(src_path, target)
            else:
                shutil.copy2(src_path, target)
            # 🔐 Only clean up if the source file lived DIRECTLY under _tmp (mkstemp case).
            # If it was inside a staged dir like _tmp/dir_xxx, DO NOT prune that dir here.
            if cleanup and os.path.normcase(os.path.abspath(src_parent)) == os.path.normcase(os.path.abspath(self._tmp_root)):
                _maybe_cleanup_tmp_parent(self._tmp_root, src_path)

        mime, _ = mimetypes.guess_type(target)
        uri = _to_file_uri(target)

        # optional "pretty" mirror path (symlink) if suggested
        if suggested_uri:
            pretty = _normalize_pretty_path(self.base_dir, suggested_uri)
            os.makedirs(os.path.dirname(pretty), exist_ok=True)
            if not os.path.exists(pretty):
                try:
                    os.symlink(target, pretty)
                except OSError:
                    shutil.copy2(target, pretty)
        

        # ✅ Remove this unconditional cleanup (it could wipe staged dirs):
        # _maybe_cleanup_tmp_parent(self._tmp_root, path)

        # Ensure _tmp exists for future staging even if it was emptied
        os.makedirs(self._tmp_root, exist_ok=True)

        a = Artifact(
            artifact_id=sha, uri=uri, kind=kind, bytes=nbytes, sha256=sha, mime=mime,
            run_id=run_id, graph_id=graph_id, node_id=node_id, tool_name=tool_name, tool_version=tool_version,
            created_at=_now_iso(), labels=labels or {}, metrics=metrics or {}, preview_uri=preview_uri, pinned=pin
        )
        self.last_artifact = a
        return a
    
    def save_text(
            self, payload: str, *, suggested_uri: Optional[str] = None
    ):
        """ Save a text payload as an artifact. """
        staged_path = self.tmp_path(suffix=".txt")
        with open(staged_path, "w", encoding="utf-8") as f:
            f.write(payload)
        a = self.save_file(
            path=staged_path, kind="text", run_id="ad-hoc", graph_id="ad-hoc", node_id="ad-hoc",
            tool_name="fs_store.save_text", tool_version="0.1.0",
            suggested_uri=suggested_uri, cleanup=True
        )
        return a
        
    def save_json(self, payload: str, *, suggested_uri: Optional[str] = None):
        """ Save a JSON payload as an artifact. """
        import json
        staged_path = self.tmp_path(suffix=".json")
        with open(staged_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        a = self.save_file(
            path=staged_path, kind="json", run_id="ad-hoc", graph_id="ad-hoc", node_id="ad-hoc",
            tool_name="fs_store.save_json", tool_version="0.1.0",
            suggested_uri=suggested_uri, cleanup=True
        )
        return a
        
    @contextmanager
    def open_writer(
        self, *, kind: str, run_id: str, graph_id: str, node_id: str,
        tool_name: str, tool_version: str, planned_ext: Optional[str] = None,
        pin: bool = False
    ):
        """ Context manager that yields a streaming ArtifactWriter. """
        w = _Writer(self._tmp_root, planned_ext)
        try:
            yield w
            w.close()
            sha, nbytes = _sha256_file(w.tmp_path)
            target = _content_addr_path(self.base_dir, sha, planned_ext)
            if not os.path.exists(target):
                shutil.move(w.tmp_path, target)
                _maybe_cleanup_tmp_parent(self._tmp_root, w.tmp_path)
            else:
                os.remove(w.tmp_path)  # already present => dedup
                _maybe_cleanup_tmp_parent(self._tmp_root, w.tmp_path)
            mime, _ = mimetypes.guess_type(target)
            a = Artifact(
                artifact_id=sha, uri=_to_file_uri(target), kind=kind,
                bytes=nbytes, sha256=sha, mime=mime, run_id=run_id, graph_id=graph_id,
                node_id=node_id, tool_name=tool_name, tool_version=tool_version,
                created_at=_now_iso(), labels=w._labels, metrics=w._metrics, pinned=pin
            )
            # stash on the writer so caller can grab it after context exits
            w._artifact = a
            self.last_artifact = a
        except Exception:
            try:
                w.close()
                if os.path.exists(w.tmp_path): os.remove(w.tmp_path)
            finally:
                raise

    @contextmanager
    def open_reader(self, uri: str):
        """ Context manager that yields an ArtifactReader for a given URI. """
        path = _from_uri_or_path(uri)
        f = open(path, "rb")
        try:
            yield _Reader(path, f)
        finally:
            f.close()

    # --------------------- advanced flow for external tools ------------------
    def plan_staging_path(self, planned_ext: str = "") -> str:
        """Return a temp path that an external tool can write to directly."""
        os.makedirs(self._tmp_root, exist_ok=True)  # ensure _tmp exists
        return self.tmp_path(suffix=planned_ext)
    
    def ingest_staged_file(
        self, staged_path: str, *, kind: str, run_id: str, graph_id: str, node_id: str,
        tool_name: str, tool_version: str, pin: bool=False, labels: Optional[dict]=None,
        metrics: Optional[dict]=None, preview_uri: Optional[str]=None,
        suggested_uri: Optional[str]=None
    ) -> Artifact:
        """Turn a staged file into a content-addressed artifact + (optional) pretty link."""
        return self.save_file(
            path=staged_path, kind=kind, run_id=run_id, graph_id=graph_id, node_id=node_id,
            tool_name=tool_name, tool_version=tool_version, suggested_uri=suggested_uri,
            pin=pin, labels=labels, metrics=metrics, preview_uri=preview_uri
        )
    

    def plan_staging_dir(self, suffix: str = "") -> str:
        """Return an empty directory path that an external tool can write into."""
        # make a unique folder under _tmp
        d = tempfile.mkdtemp(prefix="dir_", suffix=suffix, dir=self._tmp_root)
        return d

    def ingest_directory(
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
        index_children: bool = False,
        pin: bool = False,
        labels: Optional[dict] = None,
        metrics: Optional[dict] = None,
        suggested_uri: Optional[str] = None,
        archive: bool = False,
        archive_name: str = "bundle.tar.gz",
        cleanup: bool = True,
        store: str | None = None,  # NEW: "archive" | "copy" | "manifest"; None -> derive from 'archive'
    ) -> Artifact:
        if not os.path.isdir(staged_dir):
            raise ValueError(f"ingest_directory: not a directory: {staged_dir}")

        if store is None:
            store = "archive" if archive else "manifest"  # previous default was manifest-only

        manifest_entries, tree_sha = _tree_manifest_and_hash(staged_dir, include, exclude)
        cas_dir = _content_addr_dir_path(self.base_dir, tree_sha)
        manifest_path = os.path.join(cas_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            _write_json(manifest_path, {
                "files": manifest_entries,
                "created_at": _now_iso(),
                "tool_name": tool_name,
                "tool_version": tool_version,
            })

        archive_uri = None
        if store == "archive":
            archive_path = os.path.join(cas_dir, archive_name)
            if not os.path.exists(archive_path):
                import tarfile
                with tarfile.open(archive_path, mode="w:gz") as tar:
                    for e in sorted(manifest_entries, key=lambda x: x["path"]):
                        abs_file = os.path.join(staged_dir, e["path"])
                        tar.add(abs_file, arcname=e["path"])
            archive_uri = _to_file_uri(archive_path)

        elif store == "copy":
            dst_root = os.path.join(cas_dir, "tree")
            for e in manifest_entries:
                src = os.path.join(staged_dir, e["path"])
                dst = os.path.join(dst_root, e["path"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

        elif store == "manifest":
            if cleanup:
                raise ValueError("store='manifest' with cleanup=True would lose bytes; set cleanup=False or use store='archive'/'copy'.")
        else:
            raise ValueError(f"unknown store mode: {store}")

        # Pretty link: try to symlink the whole directory to CAS dir (best UX)
        if suggested_uri:
            pretty_dir = _normalize_pretty_path(self.base_dir, suggested_uri)
            parent = os.path.dirname(pretty_dir)
            os.makedirs(parent, exist_ok=True)
            try:
                # prefer a directory symlink if pretty path doesn't exist
                if not os.path.lexists(pretty_dir):  # avoid overwriting
                    os.symlink(cas_dir, pretty_dir, target_is_directory=True)
            except OSError:
                # Fallback: ensure dir exists and link/copy small files inside
                os.makedirs(pretty_dir, exist_ok=True)
                pm = os.path.join(pretty_dir, "manifest.json")
                if not os.path.exists(pm):
                    try: os.symlink(manifest_path, pm)
                    except OSError: shutil.copy2(manifest_path, pm)
                if store == "archive" and archive_uri:
                    pa = os.path.join(pretty_dir, archive_name)
                    if not os.path.exists(pa):
                        src = archive_uri[len("file://"):]
                        try: os.symlink(src, pa)
                        except OSError: shutil.copy2(src, pa)
                elif store == "copy":
                    # copy small files (under 1MB) for convenience
                    for e in manifest_entries:
                        if e["bytes"] <= 1024*1024:
                            src = os.path.join(cas_dir, "tree", e["path"])
                            dst = os.path.join(pretty_dir, e["path"])
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            if not os.path.exists(dst):
                                try: os.symlink(src, dst)
                                except OSError: shutil.copy2(src, dst)

        total_bytes = sum(e["bytes"] for e in manifest_entries)
        a = Artifact(
            artifact_id=tree_sha,
            uri=_to_file_uri(cas_dir),
            kind=kind,
            bytes=total_bytes,
            sha256=tree_sha,
            mime="application/vnd.aethergraph.bundle+dir",
            run_id=run_id, graph_id=graph_id, node_id=node_id,
            tool_name=tool_name, tool_version=tool_version,
            created_at=_now_iso(),
            labels=labels or {}, metrics=metrics or {}, preview_uri=archive_uri, pinned=pin
        )

        self.last_artifact = a

        if cleanup and store in ("archive", "copy"):
            try: shutil.rmtree(staged_dir, ignore_errors=True)
            except Exception: pass

        return a
    

    def cleanup_tmp(self, max_age_hours: int = 24):
        now = datetime.now(datetime.timezone.utc).timestamp()
        for p in Path(self._tmp_root).rglob("*"):
            try:
                age_h = (now - p.stat().st_mtime) / 3600.0
                if age_h > max_age_hours:
                    if p.is_file(): p.unlink(missing_ok=True)
                    else: shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass

    def load_bytes(self, uri: str) -> bytes:
        path = _from_uri_or_path(uri)
        if os.path.isdir(path):
            raise IsADirectoryError(f"Expected file, got directory: {path}")
        with open(path, "rb") as f:
            return f.read()

    def load_text(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> str:
        data = self.load_bytes(uri)
        return data.decode(encoding, errors)

    def load_json(
        self,
        uri: str,
        *,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> Any:
        text = self.load_text(uri, encoding=encoding, errors=errors)
        return json.loads(text)
    
        
    def load_artifact(self, uri: str) -> Union[str, bytes]:
        """Load an artifact by URI.

        - If it's a directory, return the directory path as a string.
        - If it's a file, return the file contents as bytes.
        """
        path = _from_uri_or_path(uri)
        if os.path.isdir(path):
            return path
        with open(path, "rb") as f:
            return f.read()

    def load_artifact_bytes(self, uri: str) -> bytes:
        """Load a file artifact and return its bytes.

        Raises:
            IsADirectoryError: if the URI points to a directory.
        """
        path = _from_uri_or_path(uri)
        if os.path.isdir(path):
            raise IsADirectoryError(f"Expected file, got directory: {path}")
        with open(path, "rb") as f:
            return   f.read()

    def load_artifact_dir(self, uri: str) -> str:
        """Return the path when the artifact is a directory."""
        path = _from_uri_or_path(uri)
        if not os.path.isdir(path):
            raise NotADirectoryError(f"Expected directory, got file: {path}")
        return path
    
class FSArtifactStore:  # implements AsyncArtifactStore
    def __init__(self, base_dir: str):
        self._sync = FileArtifactStoreSync(base_dir)

    @property
    def base_uri(self) -> str:
        return self._sync.base_uri
    
    def tmp_path(self, suffix: str = "") -> str:
        return self._sync.tmp_path(suffix=suffix)

    async def save_file(self, **kw) -> Any:
        return await to_thread(self._sync.save_file, **kw)
    
    async def save_text(self, **kw) -> Any:
        return await to_thread(self._sync.save_text, **kw)

    async def save_json(self, **kw) -> Any:
        return await to_thread(self._sync.save_json, **kw)
    
    async def open_writer(self, **kw):
        # Wrap the sync contextmanager so 'with' usage in Facade stays the same.
        # Return the sync contextmanager directly; user code runs inside with-block
        # but all disk ops inside are already sync and cheap; or expose an async CM.
        return self._sync.open_writer(**kw)

    async def plan_staging_path(self, planned_ext: str = "") -> str:
        return await to_thread(self._sync.plan_staging_path, planned_ext)

    async def ingest_staged_file(self, **kw) -> Any:
        return await to_thread(self._sync.ingest_staged_file, **kw)

    async def plan_staging_dir(self, suffix: str = "") -> str:
        return await to_thread(self._sync.plan_staging_dir, suffix)

    async def ingest_directory(self, **kw) -> Any:
        return await to_thread(self._sync.ingest_directory, **kw)

    async def load_bytes(self, uri: str) -> bytes:
        return await to_thread(self._sync.load_bytes, uri)
    
    async def load_text(self, uri: str, *, encoding: str = "utf-8", errors: str = "strict") -> str:
        return await to_thread(self._sync.load_text, uri, encoding=encoding, errors=errors)
    
    async def load_json(self, uri: str, *, encoding: str = "utf-8", errors: str = "strict") -> Any:
        return await to_thread(self._sync.load_json, uri, encoding=encoding, errors=errors)
    
    async def load_artifact(self, uri: str):
        return await to_thread(self._sync.load_artifact, uri)

    async def load_artifact_bytes(self, uri: str) -> bytes:
        return await to_thread(self._sync.load_artifact_bytes, uri)

    async def load_artifact_dir(self, uri: str) -> str:
        return await to_thread(self._sync.load_artifact_dir, uri)

    async def cleanup_tmp(self, max_age_hours: int = 24):
        return await to_thread(self._sync.cleanup_tmp, max_age_hours)

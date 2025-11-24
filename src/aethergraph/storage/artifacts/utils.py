import asyncio
from datetime import datetime, timezone
from fnmatch import fnmatch
import hashlib
import json
import os
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def to_thread(fn, *a, **k):
    return await asyncio.to_thread(fn, *a, **k)


# ----- helpers ----- NOTE: we have multiple copies of these in different places, consider centralizing -----
def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str, chunk=1024 * 1024) -> tuple[str, int]:
    """Return (sha256 hex, size in bytes) of a file."""
    h = hashlib.sha256()
    total = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            total += len(b)
    return h.hexdigest(), total


def _content_addr_path(base_dir: str, sha256: str, ext: str | None) -> str:
    """Return a content-addressed path under base_dir for a given sha256 and optional extension.
    Creates subdirectories as needed.

    It works as follows:
    - Takes the first 4 characters of the sha256 hash to create two levels of subdirectories.
    - The first two characters form the first subdirectory (sub1).
    - The next two characters form the second subdirectory (sub2).
    - The full sha256 hash, optionally followed by the provided file extension, is used as the filename.
    - Ensures that the target directory exists by creating it if necessary.
    - Returns the full path to the content-addressed file.

    The final path structure will look like:
    base_dir/sub1/sub2/sha256[.ext]
    """
    sub1, sub2 = sha256[:2], sha256[2:4]
    fname = sha256 + (ext or "")
    target_dir = os.path.join(base_dir, sub1, sub2)
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, fname)


def _walk_dir(root: str, include: list[str] | None, exclude: list[str] | None):
    """Yield (relpath, abspath) for files under root honoring include/exclude globs."""
    root_p = Path(root)
    for p in root_p.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.relative_to(root_p)).replace("\\", "/")
        if exclude and any(fnmatch.fnmatch(rel, pat) for pat in exclude):
            continue
        if include and not any(fnmatch.fnmatch(rel, pat) for pat in include):
            continue
        yield rel, str(p)


def _tree_manifest_and_hash(root: str, include: list[str] | None, exclude: list[str] | None):
    """
    Build a deterministic manifest of files: [{"path": rel, "sha256": sha, "bytes": n}, ...]
    The tree hash is sha256 over JSON lines: "<rel> <sha> <bytes>\n" sorted by rel.
    """
    entries = []
    lines = []
    for rel, abspath in _walk_dir(root, include, exclude):
        sha, nbytes = _sha256_file(abspath)
        entries.append({"path": rel, "sha256": sha, "bytes": nbytes})
        lines.append(f"{rel}\t{sha}\t{nbytes}\n")
    # sort for determinism
    lines.sort()
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
    tree_sha = h.hexdigest()
    return entries, tree_sha


def _content_addr_dir_path(base_dir: str, tree_sha: str):
    # content-addressed folder to hold manifest (and optional archive)
    sub1, sub2 = tree_sha[:2], tree_sha[2:4]
    target_dir = os.path.join(base_dir, sub1, sub2, tree_sha)
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def _write_json(path: str, obj: dict | list):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def _maybe_cleanup_tmp_parent(tmp_root: str, path: str):
    """Remove empty parent dirs strictly under tmp_root (never _tmp itself)."""
    try:
        parent = os.path.dirname(os.path.abspath(path))
        tmp_root_abs = os.path.abspath(tmp_root)

        # Only operate if `parent` is inside tmp_root
        while (
            os.path.commonpath([parent, tmp_root_abs]) == tmp_root_abs
            and os.path.normcase(parent)
            != os.path.normcase(tmp_root_abs)  # don't delete _tmp itself
        ):
            try:
                os.rmdir(parent)  # only removes if empty
            except OSError:
                break
            parent = os.path.dirname(parent)
    except Exception:
        pass

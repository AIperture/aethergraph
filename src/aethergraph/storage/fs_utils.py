import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


async def to_thread(fn, *a, **k):
    return await asyncio.to_thread(fn, *a, **k)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[BinaryIO]:
    """Hold one advisory cross-process lock for a local storage mutation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


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
    # if u.netloc:
    #     raw = f"//{u.netloc}{u.path}"   # UNC: file://server/share/...
    # else:
    #     raw = u.path                    # Local drive: file:///C:/...
    raw = f"//{u.netloc}{u.path}" if u.netloc else u.path
    return Path(url2pathname(unquote(raw)))

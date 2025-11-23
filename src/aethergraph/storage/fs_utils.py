import asyncio
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


async def to_thread(fn, *a, **k):
    return await asyncio.to_thread(fn, *a, **k)


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

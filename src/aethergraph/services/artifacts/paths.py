# at top of the file
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname


def _from_uri_or_path(s: str | Path) -> Path:
    """Turn a file:// URI or plain path into a local Path (Windows-safe)."""
    if isinstance(s, Path):
        return s
    if not isinstance(s, str):
        raise TypeError(f"Expected str/Path, got {type(s)}")
    if "://" not in s:
        return Path(s)
    u = urlparse(s)
    if (u.scheme or "").lower() != "file":
        # Not a local FS location; return a Path of the original to keep type uniform
        # Callers can decide what to do; or raise if you want to enforce FS-only.
        return Path(s)
    # UNC: file://server/share/path  -> \\server\share\path
    # Local: file:///C:/path         -> C:\path
    raw = (f"//{u.netloc}{u.path}") if u.netloc else u.path
    return Path(url2pathname(unquote(raw)))

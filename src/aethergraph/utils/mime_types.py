"""Canonical filename-based MIME metadata shared across runtime services."""

from __future__ import annotations

import mimetypes
from pathlib import PurePath

SPECIAL_EXTENSION_MIME: dict[str, str] = {
    ".zmx": "application/x-zemax-zmx",
}

EXTENSION_MIME_KIND: dict[str, tuple[str, str]] = {
    ".md": ("text/markdown", "markdown"),
    ".markdown": ("text/markdown", "markdown"),
    ".rst": ("text/x-rst", "text"),
    ".txt": ("text/plain", "text"),
    ".log": ("text/plain", "text"),
    ".csv": ("text/csv", "csv"),
    ".tsv": ("text/tab-separated-values", "csv"),
    ".yaml": ("application/yaml", "yaml"),
    ".yml": ("application/yaml", "yaml"),
    ".toml": ("application/toml", "toml"),
    ".ini": ("text/plain", "text"),
    ".cfg": ("text/plain", "text"),
    ".conf": ("text/plain", "text"),
    ".json": ("application/json", "json"),
    ".jsonl": ("application/x-ndjson", "json"),
    ".ndjson": ("application/x-ndjson", "json"),
    ".xml": ("application/xml", "xml"),
    ".html": ("text/html", "html"),
    ".htm": ("text/html", "html"),
    ".css": ("text/css", "code"),
    ".py": ("text/x-python", "code"),
    ".ipynb": ("application/x-ipynb+json", "json"),
    ".js": ("text/javascript", "code"),
    ".mjs": ("text/javascript", "code"),
    ".cjs": ("text/javascript", "code"),
    ".ts": ("text/typescript", "code"),
    ".tsx": ("text/tsx", "code"),
    ".jsx": ("text/jsx", "code"),
    ".java": ("text/x-java-source", "code"),
    ".c": ("text/x-c", "code"),
    ".h": ("text/x-c", "code"),
    ".cpp": ("text/x-c++", "code"),
    ".hpp": ("text/x-c++", "code"),
    ".cc": ("text/x-c++", "code"),
    ".go": ("text/x-go", "code"),
    ".rs": ("text/x-rust", "code"),
    ".rb": ("text/x-ruby", "code"),
    ".php": ("text/x-php", "code"),
    ".swift": ("text/x-swift", "code"),
    ".kt": ("text/x-kotlin", "code"),
    ".kts": ("text/x-kotlin", "code"),
    ".sh": ("text/x-shellscript", "code"),
    ".ps1": ("text/x-powershell", "code"),
    ".bat": ("text/plain", "code"),
    ".cmd": ("text/plain", "code"),
    ".sql": ("application/sql", "code"),
    ".graphql": ("application/graphql", "code"),
    ".png": ("image/png", "image"),
    ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"),
    ".gif": ("image/gif", "image"),
    ".webp": ("image/webp", "image"),
    ".bmp": ("image/bmp", "image"),
    ".tif": ("image/tiff", "image"),
    ".tiff": ("image/tiff", "image"),
    ".ico": ("image/x-icon", "image"),
    ".svg": ("image/svg+xml", "image"),
    ".pdf": ("application/pdf", "pdf"),
    ".zip": ("application/zip", "archive"),
    ".tar": ("application/x-tar", "archive"),
    ".gz": ("application/gzip", "archive"),
    ".tgz": ("application/gzip", "archive"),
    ".bz2": ("application/x-bzip2", "archive"),
    ".7z": ("application/x-7z-compressed", "archive"),
    ".rar": ("application/vnd.rar", "archive"),
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "document",
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "document",
    ),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "document",
    ),
}


def mime_type_for_filename(filename: str, *declared_values: object) -> str:
    """Resolve one canonical MIME value from declarations and filename evidence.

    Explicit non-generic declarations have precedence. A generic octet-stream
    declaration deliberately yields to the curated extension registry so callers
    that could not classify an upload do not erase stronger filename metadata.
    """

    generic_declared = False
    for value in declared_values:
        if not isinstance(value, str):
            continue
        normalized = value.partition(";")[0].strip().lower()
        if not normalized or "/" not in normalized:
            continue
        if normalized == "application/octet-stream":
            generic_declared = True
            continue
        return normalized

    extension = PurePath(filename).suffix.lower()
    if extension in SPECIAL_EXTENSION_MIME:
        return SPECIAL_EXTENSION_MIME[extension]
    if extension in EXTENSION_MIME_KIND:
        return EXTENSION_MIME_KIND[extension][0]

    guessed, _ = mimetypes.guess_type(filename, strict=False)
    if guessed:
        return guessed.lower()
    if generic_declared:
        return "application/octet-stream"
    return "application/octet-stream"


__all__ = [
    "EXTENSION_MIME_KIND",
    "SPECIAL_EXTENSION_MIME",
    "mime_type_for_filename",
]

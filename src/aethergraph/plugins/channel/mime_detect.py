from __future__ import annotations

from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path


@dataclass(frozen=True)
class MimeDetectResult:
    detected_mime: str
    content_kind: str  # "json" | "markdown" | "code" | "pdf" | "image" | "archive" | "text" | "binary" | "zmx" | ...
    encoding: str | None = None  # e.g. "utf-8"
    reason: str = ""  # for debugging/observability


# -----------------------------------------------------------------------------
# Special / custom formats (you can expand this)
# -----------------------------------------------------------------------------
SPECIAL_EXT_MIME: dict[str, str] = {
    ".zmx": "application/x-zemax-zmx",
}
SPECIAL_EXT_KIND: dict[str, str] = {
    ".zmx": "zmx",
}

# -----------------------------------------------------------------------------
# Curated extension->(mime, kind) for "common" files where mimetypes can be vague
# -----------------------------------------------------------------------------
EXT_MIME_KIND: dict[str, tuple[str, str]] = {
    # docs / text-ish
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
    # code
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
    # images
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
    # pdf
    ".pdf": ("application/pdf", "pdf"),
    # archives
    ".zip": ("application/zip", "archive"),
    ".tar": ("application/x-tar", "archive"),
    ".gz": ("application/gzip", "archive"),
    ".tgz": ("application/gzip", "archive"),
    ".bz2": ("application/x-bzip2", "archive"),
    ".7z": ("application/x-7z-compressed", "archive"),
    ".rar": ("application/vnd.rar", "archive"),
    # office
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "document",
    ),
    ".xlsx": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "document"),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "document",
    ),
}

# # Register custom/curated types so mimetypes.guess_type also improves
# for ext, (mt, _kind) in {**SPECIAL_EXT_MIME, **{k: v[0] for k, v in EXT_MIME_KIND.items()}}.items():
#     mimetypes.add_type(mt, ext)

# 1) special ext mime (value is a string)
for ext, mt in SPECIAL_EXT_MIME.items():
    mimetypes.add_type(mt, ext)

# 2) curated ext mapping (value is (mime, kind))
for ext, (mt, _kind) in EXT_MIME_KIND.items():
    mimetypes.add_type(mt, ext)


def _looks_like_text(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    # Heuristic: if lots of control chars, likely binary
    ctrl = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return (ctrl / max(len(sample), 1)) < 0.02


def _sniff_by_signature(sample: bytes) -> MimeDetectResult | None:
    """
    Detect some common formats via magic bytes (fast, no deps).
    """
    if sample.startswith(b"%PDF-"):
        return MimeDetectResult("application/pdf", "pdf", reason="sig:pdf")

    # PNG
    if sample.startswith(b"\x89PNG\r\n\x1a\n"):
        return MimeDetectResult("image/png", "image", reason="sig:png")

    # JPEG
    if sample.startswith(b"\xff\xd8\xff"):
        return MimeDetectResult("image/jpeg", "image", reason="sig:jpeg")

    # GIF
    if sample.startswith(b"GIF87a") or sample.startswith(b"GIF89a"):
        return MimeDetectResult("image/gif", "image", reason="sig:gif")

    # WEBP (RIFF....WEBP)
    if len(sample) >= 12 and sample[0:4] == b"RIFF" and sample[8:12] == b"WEBP":
        return MimeDetectResult("image/webp", "image", reason="sig:webp")

    # ZIP (also docx/xlsx/pptx are zip containers)
    if (
        sample.startswith(b"PK\x03\x04")
        or sample.startswith(b"PK\x05\x06")
        or sample.startswith(b"PK\x07\x08")
    ):
        return MimeDetectResult("application/zip", "archive", reason="sig:zip")

    # GZIP
    if sample.startswith(b"\x1f\x8b"):
        return MimeDetectResult("application/gzip", "archive", reason="sig:gzip")

    return None


def _classify_text_content(text_prefix: str) -> MimeDetectResult | None:
    """
    Decide between json / yaml / xml / html / markdown / code / plain text.
    Only uses prefix; cheap and robust.
    """
    s = text_prefix.lstrip()

    # JSON object/array
    if s.startswith("{") or s.startswith("["):
        try:
            json.loads(s)
            return MimeDetectResult(
                "application/json", "json", encoding="utf-8", reason="text:json_parse"
            )
        except Exception:
            # might be partial prefix; still likely json
            return MimeDetectResult(
                "application/json", "json", encoding="utf-8", reason="text:json_like"
            )

    # XML / HTML
    if s.startswith("<?xml"):
        return MimeDetectResult(
            "application/xml", "xml", encoding="utf-8", reason="text:xml_prolog"
        )
    if s.startswith("<!DOCTYPE html") or s.startswith("<html") or s.startswith("<HTML"):
        return MimeDetectResult("text/html", "html", encoding="utf-8", reason="text:html")

    # YAML heuristic
    if s.startswith("---") or "\n---" in text_prefix[:200]:
        return MimeDetectResult(
            "application/yaml", "yaml", encoding="utf-8", reason="text:yaml_sep"
        )
    if ":" in s[:120] and "\n" in s[:120] and not s.startswith("#include"):
        # weak but useful
        # NOTE: don't over-classify; this is a heuristic
        return MimeDetectResult(
            "application/yaml", "yaml", encoding="utf-8", reason="text:yaml_heuristic"
        )

    # Markdown heuristic
    md_markers = ("# ", "## ", "### ", "- ", "* ", "> ", "```")
    if any(s.startswith(m) for m in md_markers):
        return MimeDetectResult(
            "text/markdown", "markdown", encoding="utf-8", reason="text:markdown_heuristic"
        )

    # Code heuristic
    code_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "function ",
        "const ",
        "let ",
        "var ",
        "#include",
        "package ",
        "fn ",
        "public class",
        "using ",
    )
    if any(s.startswith(m) for m in code_markers):
        return MimeDetectResult(
            "text/plain", "code", encoding="utf-8", reason="text:code_heuristic"
        )

    return MimeDetectResult("text/plain", "text", encoding="utf-8", reason="text:plain")


def detect_mime_for_path(path: str, *, declared: str | None = None) -> MimeDetectResult:
    p = Path(path)
    ext = p.suffix.lower()

    # 1) Special-case extensions (e.g., .zmx)
    if ext in SPECIAL_EXT_MIME:
        return MimeDetectResult(
            detected_mime=SPECIAL_EXT_MIME[ext],
            content_kind=SPECIAL_EXT_KIND.get(ext, ext.lstrip(".")),
            reason="special_ext_map",
        )

    # 2) Curated extension mapping for common formats
    if ext in EXT_MIME_KIND:
        mt, kind = EXT_MIME_KIND[ext]
        # Still allow signature checks below to override if file lies about extension
        ext_guess = MimeDetectResult(mt, kind, reason="ext:curated")
    else:
        ext_guess = None

    # 3) Read a prefix for sniffing
    with open(path, "rb") as f:
        sample = f.read(64 * 1024)

    # 4) Signature sniff (no deps) for strong types
    sig = _sniff_by_signature(sample)
    if sig is not None:
        # If it's a zip, we can refine office docs by extension
        if sig.detected_mime == "application/zip" and ext in (".docx", ".xlsx", ".pptx"):
            mt, kind = EXT_MIME_KIND[ext]
            return MimeDetectResult(mt, kind, reason="sig:zip+ext:office")
        return sig

    # 5) libmagic best-effort (optional)
    detected_magic = ""
    try:
        import magic  # type: ignore

        m = magic.Magic(mime=True)
        detected_magic = m.from_buffer(sample) or ""
    except Exception:
        detected_magic = ""

    # 6) If it looks like text (or libmagic says text/*), classify deeper
    if _looks_like_text(sample) or detected_magic.startswith("text/"):
        try:
            text_prefix = sample.decode("utf-8", errors="strict")
            classified = _classify_text_content(text_prefix)
            if classified:
                # If libmagic gave a more specific text mime, keep it unless we detected a strong semantic type
                if (
                    detected_magic
                    and classified.content_kind in ("text", "code")
                    and detected_magic.startswith("text/")
                ):
                    return MimeDetectResult(
                        detected_magic,
                        classified.content_kind,
                        "utf-8",
                        reason="magic_text+heuristic",
                    )
                return classified
        except UnicodeDecodeError:
            # Not utf-8; treat as binary unless you add encoding detection
            pass

    # 7) If we had a curated ext guess, use it now
    if ext_guess is not None:
        return ext_guess

    # 8) Fallback to extension-based mimetypes
    guessed, _ = mimetypes.guess_type(p.name)
    if guessed:
        kind = "text" if guessed.startswith("text/") else "binary"
        return MimeDetectResult(guessed, kind, reason="mimetypes_guess")

    # 9) If client declared something reasonable, last-chance use it
    if declared and "/" in declared:
        kind = "text" if declared.startswith("text/") else "binary"
        return MimeDetectResult(declared, kind, reason="declared_fallback")

    # 10) Final fallback
    return MimeDetectResult("application/octet-stream", "binary", reason="default")

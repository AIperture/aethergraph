# aethergraph/v2/utils/data_io.py
from __future__ import annotations

import csv
import hashlib
import io
import os
from typing import Any

# Optional deps
try:
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False

try:
    from pypdf import PdfReader  # lightweight text extractor

    _HAS_PYPDF = True
except Exception:
    _HAS_PYPDF = False


# ---------- URI helpers ----------


def _resolve_local_path(uri: str) -> str | None:
    if uri.startswith("file://"):
        return uri[len("file://") :]
    return None


def load_bytes(artifact_store, uri: str) -> bytes:
    """Load raw bytes from artifact_store via URI (prefer local file path).
    TODO: deprecate this function as artifact_store.get_bytes(uri) is preferred.
    """
    p = _resolve_local_path(uri)
    if p and os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    # Optional: if later add artifact_store.get_bytes(uri), handle here.
    raise FileNotFoundError(f"Cannot resolve bytes for URI: {uri}")


# ---------- MIME normalization & classification ----------

_EXTENSION_TO_MIME = {
    # images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "heic": "image/heic",
    "heif": "image/heif",
    # text/docs
    "txt": "text/plain",
    "log": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "json": "application/json",
    "yaml": "text/yaml",
    "yml": "text/yaml",
    "xml": "application/xml",
    "pdf": "application/pdf",
    # archives
    "zip": "application/zip",
    "gz": "application/gzip",
    "tar": "application/x-tar",
    "7z": "application/x-7z-compressed",
    # office
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # proprietary examples
    "mph": "application/octet-stream",  # COMSOL
}


def normalize_mime(name: str | None, mimetype_hint: str | None) -> str:
    mt = (mimetype_hint or "").lower().strip()
    if mt:
        return mt
    if name:
        n = name.lower()
        if "." in n:
            ext = n.rsplit(".", 1)[-1]
            return _EXTENSION_TO_MIME.get(ext, "application/octet-stream")
    return "application/octet-stream"


def classify_for_processing(mime: str) -> str:
    m = mime.lower()
    if m.startswith("image/") and m != "image/svg+xml":
        return "image"
    if m in ("image/svg+xml", "application/xml", "text/xml"):
        return "xml"
    if m.startswith("text/") or m in ("application/json",):
        return "text"
    if m == "application/pdf":
        return "pdf"
    if m in (
        "application/zip",
        "application/gzip",
        "application/x-tar",
        "application/x-7z-compressed",
    ):
        return "archive"
    if m in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",):
        return "xlsx"
    return "binary"  # unknown/proprietary (e.g., COMSOL .mph)


# ---------- Text helpers ----------


def try_decode_text(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1", errors="replace")


# ---------- CSV helpers ----------


def read_csv_any(b: bytes) -> dict[str, Any]:
    """
    Returns a lightweight preview for CSV/TSV; if pandas is available, also return a DataFrame preview.
    """
    txt = try_decode_text(b)
    # Dialect sniff
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(txt.splitlines()[0] if txt else "")
    except Exception:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(txt), dialect=dialect))
    preview_rows = rows[:10]
    out: dict[str, Any] = {"rows_preview": preview_rows, "num_rows_previewed": len(preview_rows)}
    if _HAS_PANDAS:
        try:
            import pandas as pd  # type: ignore

            # Let pandas infer separator; fallback to comma
            df = pd.read_csv(io.StringIO(txt))
            out["pandas_head"] = df.head(10)  # DataFrame (caller may display)
            out["columns"] = list(df.columns)
            out["shape"] = tuple(df.shape)
        except Exception:
            pass
    return out


# ---------- Image helpers ----------


def decode_image_pil(b: bytes, *, fix_orientation: bool = True, to_rgb: bool = True):
    if not _HAS_PIL:
        raise RuntimeError("Pillow not installed; cannot decode image.")
    from PIL import Image, ImageCms, ImageOps  # local import for safety

    im = Image.open(io.BytesIO(b))
    if fix_orientation:
        im = ImageOps.exif_transpose(im)
    try:
        if "icc_profile" in im.info:
            src = ImageCms.ImageCmsProfile(io.BytesIO(im.info.get("icc_profile")))
            dst = ImageCms.createProfile("sRGB")
            im = ImageCms.profileToProfile(im, src, dst, outputMode=im.mode)
    except Exception:
        pass
    if to_rgb and im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    return im


def pil_to_numpy(im) -> np.ndarray:
    if not _HAS_NUMPY:
        raise RuntimeError("NumPy not installed; cannot convert image to array.")
    import numpy as np  # type: ignore

    arr = np.asarray(im)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


# ---------- PDF helpers ----------


def extract_pdf_text(b: bytes, max_pages: int = 5) -> dict[str, Any]:
    if not _HAS_PYPDF:
        raise RuntimeError("pypdf not installed; cannot extract PDF text.")
    reader = PdfReader(io.BytesIO(b))
    pages = min(len(reader.pages), max_pages)
    texts = []
    for i in range(pages):
        try:
            texts.append(reader.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    return {"num_pages": len(reader.pages), "preview_pages": pages, "text_preview": texts}


# ---------- XLSX helpers (cheap preview without heavy engines) ----------


def preview_xlsx(b: bytes) -> dict[str, Any]:
    if not _HAS_PANDAS:
        return {"note": "pandas not installed; cannot preview xlsx"}
    try:
        import pandas as pd  # type: ignore

        with io.BytesIO(b) as bio:
            xl = pd.ExcelFile(bio)
            sheets = xl.sheet_names
            out: dict[str, Any] = {"sheets": sheets, "previews": {}}
            for s in sheets[:3]:
                try:
                    df = xl.parse(s, nrows=10)
                    out["previews"][s] = df
                except Exception:
                    out["previews"][s] = "unreadable"
            return out
    except Exception as e:
        return {"error": f"xlsx preview failed: {e}"}


# ---------- Dispatcher ----------


def quick_decode(
    artifact_store, name: str | None, mimetype_hint: str | None, uri: str
) -> dict[str, Any]:
    """
    Returns: {
      'uri': str, 'mimetype': str, 'kind': 'image'|'text'|'pdf'|'archive'|'xlsx'|'binary',
      'meta': {...},  # shape/columns/pages etc
      'preview': ...  # small human-friendly preview (safe to log/send)
    }
    """
    b = load_bytes(artifact_store, uri)
    mime = normalize_mime(name, mimetype_hint)
    kind = classify_for_processing(mime)
    sha = hashlib.sha256(b).hexdigest()

    if kind == "image":
        if not _HAS_PIL:
            return {
                "uri": uri,
                "mimetype": mime,
                "kind": kind,
                "meta": {"sha256": sha},
                "preview": "Pillow missing",
            }
        im = decode_image_pil(b)
        w, h = im.size
        meta = {"width": w, "height": h, "mode": im.mode, "sha256": sha}
        if _HAS_NUMPY:
            arr = pil_to_numpy(im)
            meta["array_shape"] = tuple(arr.shape)
        return {
            "uri": uri,
            "mimetype": mime,
            "kind": kind,
            "meta": meta,
            "preview": f"{w}x{h} {im.mode}",
        }

    if kind == "text":
        txt = try_decode_text(b)
        head = "\n".join(txt.splitlines()[:20])
        return {
            "uri": uri,
            "mimetype": mime,
            "kind": kind,
            "meta": {"bytes": len(b), "sha256": sha},
            "preview": head,
        }

    if kind == "pdf":
        if not _HAS_PYPDF:
            return {
                "uri": uri,
                "mimetype": mime,
                "kind": kind,
                "meta": {"bytes": len(b), "sha256": sha},
                "preview": "pypdf missing",
            }
        meta = extract_pdf_text(b)
        return {
            "uri": uri,
            "mimetype": mime,
            "kind": kind,
            "meta": {"bytes": len(b), **meta, "sha256": sha},
            "preview": "\n---\n".join(meta["text_preview"]),
        }

    if kind == "archive":
        # We don't auto-unpack; just list ZIP members if it's zip
        import zipfile

        bio = io.BytesIO(b)
        if zipfile.is_zipfile(bio):
            with zipfile.ZipFile(bio) as z:
                names = z.namelist()[:20]
            return {
                "uri": uri,
                "mimetype": mime,
                "kind": kind,
                "meta": {"bytes": len(b), "sha256": sha},
                "preview": "\n".join(names),
            }
        return {
            "uri": uri,
            "mimetype": mime,
            "kind": kind,
            "meta": {"bytes": len(b), "sha256": sha},
            "preview": "archive (non-zip)",
        }

    if kind == "xlsx":
        meta = preview_xlsx(b)
        return {
            "uri": uri,
            "mimetype": mime,
            "kind": kind,
            "meta": {"bytes": len(b), **meta, "sha256": sha},
            "preview": f"sheets: {', '.join(meta.get('sheets', [])) if isinstance(meta, dict) else meta}",
        }

    # binary / unknown (e.g., COMSOL .mph)
    return {
        "uri": uri,
        "mimetype": mime,
        "kind": "binary",
        "meta": {"bytes": len(b), "sha256": sha},
        "preview": "opaque binary",
    }

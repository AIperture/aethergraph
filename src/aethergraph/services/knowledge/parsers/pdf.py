from __future__ import annotations

from pypdf import PdfReader


def extract_text(path: str) -> tuple[str, dict]:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n\n".join(texts), {"pages": len(reader.pages)}

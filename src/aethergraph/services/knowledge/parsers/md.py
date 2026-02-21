from __future__ import annotations


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) == 3:
        return parts[2].lstrip()
    return text


def extract_text(path: str) -> tuple[str, dict]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    txt = _strip_front_matter(txt)
    return txt, {}

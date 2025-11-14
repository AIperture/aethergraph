from __future__ import annotations


def extract_text(path: str) -> tuple[str, dict]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return txt, {}

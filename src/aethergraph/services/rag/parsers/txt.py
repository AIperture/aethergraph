from __future__ import annotations
from typing import Tuple

def extract_text(path: str) -> Tuple[str, dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    return txt, {}
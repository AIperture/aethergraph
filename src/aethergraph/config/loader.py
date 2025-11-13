# aethergraph/config_loader.py
import os
from pathlib import Path
from typing import Iterable
from .config import AppSettings 

import logging 

def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]

def load_settings() -> AppSettings:
    root = Path(__file__).resolve().parents[3]  # repo root
    cfg_dir = root / "src" / "config"

    # allow an explicit path via env var
    explicit = Path(os.environ["AETHERGRAPH_ENV_FILE"]) if "AETHERGRAPH_ENV_FILE" in os.environ else None

    candidates = _existing([
        explicit or Path("NON_EXISTENT"),  # placeholder if not set
        root / ".env",
        cfg_dir / ".env",                  
        cfg_dir / ".env.local",
        cfg_dir / ".env.secrets",
    ])
    
    if not candidates and explicit:
        raise FileNotFoundError(f"Explicitly specified env file not found: {explicit}")
    
    if len(candidates) == 0:
        log = logging.getLogger("aethergraph.config.loader")
        log.warning("No env files found; using defaults and env vars only.")

    if candidates:
        # Later files override earlier ones
        return AppSettings(_env_file=[str(p) for p in candidates])
    return AppSettings()

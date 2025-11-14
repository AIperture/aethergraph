# aethergraph/config_loader.py
from collections.abc import Iterable
import logging
import os
from pathlib import Path

from .config import AppSettings


def _existing(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p and p.exists()]


def load_settings() -> AppSettings:
    log = logging.getLogger("aethergraph.config.loader")

    # 1) explicit override
    explicit = os.getenv("AETHERGRAPH_ENV_FILE")
    explicit_path = Path(explicit).expanduser().resolve() if explicit else None

    # 2) execution context (project) – where user runs `python ...`
    cwd = Path.cwd()

    # 3) workspace-level (if user sets it)
    workspace = Path(os.getenv("AETHERGRAPH_ROOT", "./aethergraph_data")).expanduser().resolve()

    # 4) user config dir (~/.config/aethergraph/.env or XDG)
    xdg = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser().resolve()
    user_cfg_env = xdg / "aethergraph" / ".env"

    # Optional: keep a *repo dev* fallback if running from source
    # (safe; only used if that path actually exists)
    try:
        repo_root = Path(__file__).resolve().parents[3]
    except Exception:
        repo_root = None
    repo_env = (
        (repo_root / ".env").resolve() if (repo_root and (repo_root / ".env").exists()) else None
    )

    candidates = _existing(
        [
            explicit_path or Path(),  # explicit if set
            cwd / ".env",
            cwd / ".env.local",
            workspace / ".env",
            user_cfg_env,
            repo_env if repo_env else Path(),  # dev fallback only if exists
        ]
    )

    if explicit and not explicit_path.exists():
        raise FileNotFoundError(f"AETHERGRAPH_ENV_FILE not found: {explicit_path}")

    if not candidates:
        log.warning("No .env files found; using OS environment variables only.")
        return AppSettings()

    # Later files override earlier ones
    return AppSettings(_env_file=[str(p) for p in candidates])

"""Resolve canonical AetherGraph application settings."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import AppSettings


def _existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.is_file()]


def load_settings(*, env_file: str | Path | None = None) -> AppSettings:
    """
    Load validated AetherGraph settings from explicit or discovered dotenv data.

    An explicit `env_file` is authoritative and disables discovery. Without an
    argument, `AETHERGRAPH_ENV_FILE` is authoritative when set; otherwise the
    canonical execution, workspace, and user configuration candidates are
    loaded in order.

    Examples:
        Load the canonical discovered configuration:
        ```python
        settings = load_settings()
        assert settings.workspace
        ```

        Load one exact Studio-managed file:
        ```python
        settings = load_settings(env_file=".data/settings/.env")
        assert settings.llm.default.model
        ```

    Args:
        env_file: Optional exact dotenv path. No discovery candidates are read
            when it is supplied.

    Returns:
        AppSettings: The validated settings snapshot.

    Notes:
        A missing explicit file raises `FileNotFoundError`. Later discovered
        files override earlier discovered files according to Pydantic settings
        semantics.
    """

    log = logging.getLogger("aethergraph.config.loader")
    explicit = str(env_file) if env_file is not None else os.getenv("AETHERGRAPH_ENV_FILE")
    explicit_path = Path(explicit).expanduser().resolve() if explicit else None
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"AETHERGRAPH_ENV_FILE not found: {explicit_path}")
        log.info("Loading AetherGraph settings from explicit file %s", explicit_path)
        return AppSettings(_env_file=str(explicit_path))

    cwd = Path.cwd()
    workspace = (
        Path(os.getenv("AETHERGRAPH_WORKSPACE", "./aethergraph_workspace")).expanduser().resolve()
    )
    xdg = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser().resolve()
    candidates = _existing(
        [
            cwd / ".env",
            cwd / ".env.local",
            workspace / ".env",
            xdg / "aethergraph" / ".env",
        ]
    )
    if not candidates:
        log.warning("No .env files found; using OS environment variables only.")
        return AppSettings()

    log.info("Loading AetherGraph settings from discovered files: %s", candidates)
    return AppSettings(_env_file=[str(path) for path in candidates])


__all__ = ["load_settings"]

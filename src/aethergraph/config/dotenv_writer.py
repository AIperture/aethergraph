"""Minimal .env reader/writer that preserves comments and ordering."""

from __future__ import annotations

from pathlib import Path


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict, ignoring comments and blank lines."""
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        # Remove surrounding quotes if present
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def write_dotenv(path: Path, updates: dict[str, str]) -> None:
    """Merge *updates* into an existing .env file.

    - Existing keys are updated in-place (preserving their position).
    - New keys are appended at the end.
    - Comments and blank lines are preserved.
    - The file is created if it doesn't exist.
    """
    lines: list[str] = []
    seen_keys: set[str] = set()

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, _, _ = stripped.partition("=")
                key = key.strip()
                if key in updates:
                    lines.append(f"{key}={updates[key]}")
                    seen_keys.add(key)
                else:
                    lines.append(line)
            else:
                lines.append(line)

    # Append new keys not already in the file
    new_keys = [k for k in updates if k not in seen_keys]
    if new_keys:
        if lines and lines[-1].strip():
            lines.append("")  # blank separator
        for key in new_keys:
            lines.append(f"{key}={updates[key]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

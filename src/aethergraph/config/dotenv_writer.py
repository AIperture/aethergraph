"""Read, merge, and atomically replace dotenv files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from pathlib import Path
import tempfile


def read_dotenv(path: Path) -> dict[str, str]:
    """
    Parse dotenv assignments while ignoring comments and blank lines.

    Surrounding single or double quotes are removed from values. The function
    does not mutate process environment variables.

    Examples:
        Read an existing dotenv file:
        ```python
        values = read_dotenv(Path(".env"))
        print(values.get("AETHERGRAPH_WORKSPACE"))
        ```

        Read a missing file:
        ```python
        assert read_dotenv(Path("missing.env")) == {}
        ```

    Args:
        path: Dotenv file to parse.

    Returns:
        dict[str, str]: Parsed key/value assignments in file order.

    Notes:
        Duplicate keys use the last value encountered.
    """

    result: dict[str, str] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def write_dotenv(path: Path, updates: dict[str, str]) -> None:
    """
    Merge assignments into a dotenv file while preserving unrelated content.

    Existing keys are updated in place, new keys are appended, and unrelated
    comments, blank lines, and assignments retain their order.

    Examples:
        Add a new setting:
        ```python
        write_dotenv(Path(".env"), {"AETHERGRAPH_WORKSPACE": "./workspace"})
        ```

        Update an existing setting:
        ```python
        write_dotenv(Path(".env"), {"AETHERGRAPH_DEPLOY_MODE": "local"})
        ```

    Args:
        path: Dotenv file to create or update.
        updates: Assignments to merge.

    Returns:
        None: The target file is updated in place.

    Notes:
        Use `replace_dotenv` when the target must exactly match one validated
        assignment set.
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
    new_keys = [key for key in updates if key not in seen_keys]
    if new_keys:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(f"{key}={updates[key]}" for key in new_keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replace_dotenv(
    path: Path,
    values: Mapping[str, str],
    *,
    header: Iterable[str] = (),
) -> None:
    """
    Atomically replace a dotenv file with one exact assignment set.

    Content is written to a temporary sibling and moved over the target with
    `os.replace`, so readers observe either the previous complete file or the
    new complete file.

    Examples:
        Replace a managed settings file:
        ```python
        replace_dotenv(
            Path(".data/settings/.env"),
            {"AETHERGRAPH_LLM__DEFAULT__MODEL": "gpt-5-mini"},
        )
        ```

        Add a generated-file header:
        ```python
        replace_dotenv(Path(".env"), {"KEY": "value"}, header=("Managed file",))
        ```

    Args:
        path: Dotenv file to replace.
        values: Complete ordered assignment set for the new file.
        header: Optional comment lines written before the assignments.

    Returns:
        None: The target atomically references the complete new content.

    Notes:
        Keys and values containing newlines are rejected. Existing unrelated
        content is intentionally removed.
    """

    _validate_rows(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [f"# {line}" for line in header]
    assignment_lines = [f"{key}={value}" for key, value in values.items()]
    lines = [*header_lines]
    if header_lines and assignment_lines:
        lines.append("")
    lines.extend(assignment_lines)
    content = "\n".join(lines) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _validate_rows(values: Mapping[str, str]) -> None:
    for key, value in values.items():
        if not key or "=" in key or "\n" in key or "\r" in key:
            raise ValueError(f"Invalid dotenv key: {key!r}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"Dotenv value for {key!r} contains a newline.")


__all__ = ["read_dotenv", "replace_dotenv", "write_dotenv"]

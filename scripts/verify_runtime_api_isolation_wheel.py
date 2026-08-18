#!/usr/bin/env python3
"""Verify the supported runtime-isolation surface in a built AG wheel."""

from __future__ import annotations

from pathlib import Path
import sys
from zipfile import ZipFile

REQUIRED_MEMBERS = {
    "aethergraph/observability/contracts.py",
    "aethergraph/observability/inspection.py",
    "aethergraph/runtime/__init__.py",
    "aethergraph/runtime/contracts.py",
    "aethergraph/runtime/embedded.py",
    "aethergraph/runtime/errors.py",
}
FORBIDDEN_MEMBERS = {
    "aethergraph/api/v1/schemas/inspect.py",
    "aethergraph/observability/studio_translation.py",
}


def main(argv: list[str]) -> int:
    """Verify one built wheel contains only the canonical isolation modules.

    Intro:
        Reads wheel member names without importing or installing the distribution.

    Examples:
        `exit_code = main(["dist/aethergraph-0.1.0-py3-none-any.whl"])`
        `exit_code = main([])`

    Args:
        argv: Command arguments containing exactly one wheel path.

    Returns:
        int: Zero when the wheel surface passes, otherwise one.

    Notes:
        This gate validates package content only; runtime behavior remains covered by
        the embedded-runtime and observability test suites.
    """
    if len(argv) != 1:
        print("usage: verify_runtime_api_isolation_wheel.py <aethergraph-wheel>")
        return 1
    wheel_path = Path(argv[0])
    if not wheel_path.is_file():
        print(f"wheel does not exist: {wheel_path}")
        return 1
    with ZipFile(wheel_path) as wheel:
        members = set(wheel.namelist())
    missing = sorted(REQUIRED_MEMBERS - members)
    superseded = sorted(FORBIDDEN_MEMBERS & members)
    if missing or superseded:
        for member in missing:
            print(f"missing required wheel member: {member}")
        for member in superseded:
            print(f"superseded wheel member remains: {member}")
        return 1
    print("Runtime API isolation wheel audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

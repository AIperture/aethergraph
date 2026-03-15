#!/usr/bin/env python
"""
Create a fresh conda environment, install aethergraph from pip, and verify import.

Default environment name:
  aethergraph-download-test-py311

Examples:
  python scripts/conda_download_test.py
  python scripts/conda_download_test.py --env-name ag-download-test --python-version 3.12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_NAME = "aethergraph-download-test-py311"
DEFAULT_PYTHON_VERSION = "3.11"


def run(cmd: list[str]) -> None:
    """Run a command and fail fast on error."""
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def conda_executable() -> str:
    """Resolve the conda executable."""
    return os.environ.get("CONDA_EXE", "conda")


def remove_env(conda: str, env_name: str) -> None:
    """Delete a pre-existing conda environment if it exists."""
    print(f"[conda] Removing env if it exists: {env_name}")
    result = subprocess.run(
        [conda, "env", "remove", "-n", env_name, "-y"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode == 0:
        print(f"[conda] Removed existing env: {env_name}")
    else:
        print(f"[conda] No existing env removed for: {env_name}")


def create_env(conda: str, env_name: str, python_version: str) -> None:
    """Create a fresh environment with the requested Python version."""
    run([conda, "create", "-n", env_name, f"python={python_version}", "-y"])


def install_package(conda: str, env_name: str, package_name: str) -> None:
    """Install pip and the target package inside the environment."""
    run([conda, "run", "-n", env_name, "python", "-m", "pip", "install", "--upgrade", "pip"])
    run([conda, "run", "-n", env_name, "python", "-m", "pip", "install", package_name])


def verify_import(conda: str, env_name: str) -> None:
    """Verify that aethergraph imports successfully."""
    code = (
        "import aethergraph; "
        "print('aethergraph imported OK'); "
        "print('module:', aethergraph.__file__); "
        "print('version:', getattr(aethergraph, '__version__', None))"
    )
    run([conda, "run", "-n", env_name, "python", "-c", code])


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-name",
        default=DEFAULT_ENV_NAME,
        help=f"Conda environment name. Default: {DEFAULT_ENV_NAME}.",
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_PYTHON_VERSION,
        help=f"Python version for the conda environment. Default: {DEFAULT_PYTHON_VERSION}.",
    )
    parser.add_argument(
        "--package",
        default="aethergraph",
        help="Package name or pip spec to install. Default: aethergraph.",
    )
    return parser.parse_args()


def main() -> None:
    """Create a fresh conda env and run the install smoke test."""
    args = parse_args()
    conda = conda_executable()
    print(f"[info] Project root: {ROOT}")
    print(f"[info] Conda executable: {conda}")
    remove_env(conda, args.env_name)
    create_env(conda, args.env_name, args.python_version)
    install_package(conda, args.env_name, args.package)
    verify_import(conda, args.env_name)
    print("\n[done] Conda download smoke test completed successfully.")


if __name__ == "__main__":
    main()

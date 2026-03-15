#!/usr/bin/env python
"""
Build the AetherGraph package and upload it to PyPI with twine.

PyPI token setup:

Linux/macOS, current shell:
  export TWINE_USERNAME=__token__
  export TWINE_PASSWORD=pypi-<your-token>

Linux/macOS, persistent:
  Add the two export lines above to ~/.bashrc, ~/.zshrc, or your shell profile.
  Then open a new shell or source the profile.

Windows PowerShell, current shell:
  $env:TWINE_USERNAME="__token__"
  $env:TWINE_PASSWORD="pypi-<your-token>"

Windows PowerShell, persistent for the current user:
  [System.Environment]::SetEnvironmentVariable("TWINE_USERNAME", "__token__", "User")
  [System.Environment]::SetEnvironmentVariable("TWINE_PASSWORD", "pypi-<your-token>", "User")
  Then open a new PowerShell session.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command and fail fast on error."""
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def require_twine_credentials() -> None:
    """Ensure twine credentials are available via environment variables."""
    username = os.environ.get("TWINE_USERNAME")
    password = os.environ.get("TWINE_PASSWORD")
    if username and password:
        return

    raise SystemExit(
        "Missing PyPI credentials.\n\n"
        "Set both TWINE_USERNAME and TWINE_PASSWORD before running this script.\n"
        "Use TWINE_USERNAME=__token__ and TWINE_PASSWORD=<your PyPI token>.\n\n"
        "Linux/macOS:\n"
        "  export TWINE_USERNAME=__token__\n"
        "  export TWINE_PASSWORD=pypi-<your-token>\n\n"
        "Windows PowerShell:\n"
        '  $env:TWINE_USERNAME="__token__"\n'
        '  $env:TWINE_PASSWORD="pypi-<your-token>"\n'
    )


def clean_dist() -> None:
    """Remove stale distribution artifacts."""
    if DIST_DIR.exists():
        print(f"[dist] Removing existing {DIST_DIR}")
        shutil.rmtree(DIST_DIR)


def build_package() -> None:
    """Build sdist and wheel artifacts."""
    run([sys.executable, "-m", "pip", "install", "--upgrade", "build", "twine"])
    run([sys.executable, "-m", "build"])


def upload_to_pypi(repository: str, skip_existing: bool) -> None:
    """Upload built artifacts with twine."""
    cmd = [sys.executable, "-m", "twine", "upload", "--repository", repository]
    if skip_existing:
        cmd.append("--skip-existing")
    cmd.extend([str(path) for path in sorted(DIST_DIR.glob("*"))])
    if len(cmd) <= 5:
        raise RuntimeError("No build artifacts found in dist/")
    run(cmd)


def parse_args() -> argparse.Namespace:
    """Parse command line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default="pypi",
        help="Twine repository target, for example 'pypi' or 'testpypi'. Default: pypi.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass --skip-existing to twine upload.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and publish the package."""
    args = parse_args()
    print(f"[info] Project root: {ROOT}")
    require_twine_credentials()
    clean_dist()
    build_package()
    upload_to_pypi(repository=args.repository, skip_existing=args.skip_existing)
    print("\n[done] Build and upload completed successfully.")


if __name__ == "__main__":
    main()

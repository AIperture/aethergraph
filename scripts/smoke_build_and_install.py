#!/usr/bin/env python
"""
End-to-end local smoke test:

1) (Optional) build frontend and copy into src/aethergraph/server/ui_static
2) python -m build
3) create fresh .venv-ag-test
4) pip install the built wheel into that venv
5) import aethergraph and print some basic info
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---- CONFIG: adjust paths if needed ----

ROOT = Path(__file__).resolve().parent.parent  # assuming script is in ./scripts/
REPO_ROOT = ROOT.parent                        # monorepo root (aethergraph-suite)
FRONTEND_DIR = REPO_ROOT / "aethergraph-frontend"
FRONTEND_DIST = FRONTEND_DIR / "dist"
BACKEND_UI_STATIC = ROOT / "src" / "aethergraph" / "server" / "ui_static"
DIST_DIR = ROOT / "dist"
VENV_DIR = ROOT / ".venv-ag-test"

UI_BUILD_COMMANDS = {
    "oss": ["npm", "run", "build:oss"],
}


def npm_executable() -> str:
    """Return the platform-appropriate npm executable name."""
    return "npm.cmd" if os.name == "nt" else "npm"


def run(cmd, cwd=None):
    """Run a command, print it, and fail fast on error."""
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the smoke-build flow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ui-build",
        action="store_true",
        help="Build the frontend bundle and copy it into src/aethergraph/server/ui_static before packaging.",
    )
    parser.add_argument(
        "--ui-app",
        choices=sorted(UI_BUILD_COMMANDS),
        default="oss",
        help="Frontend build target to use with --ui-build (default: oss).",
    )
    return parser.parse_args()


def build_frontend(ui_app: str):
    """Build the React frontend and copy dist/ into backend ui_static/."""
    if not FRONTEND_DIR.exists():
        raise FileNotFoundError(f"[frontend] Frontend directory does not exist: {FRONTEND_DIR}")

    npm_cmd = npm_executable()
    build_cmd = [npm_cmd, *UI_BUILD_COMMANDS[ui_app][1:]]

    print(f"[frontend] Building frontend bundle for '{ui_app}'...")
    run([npm_cmd, "install"], cwd=FRONTEND_DIR)
    run(build_cmd, cwd=FRONTEND_DIR)

    if not FRONTEND_DIST.exists():
        raise RuntimeError(f"[frontend] Build completed but dist/ was not created: {FRONTEND_DIST}")

    dist_items = list(FRONTEND_DIST.iterdir())
    if not dist_items:
        raise RuntimeError(f"[frontend] Build completed but dist/ is empty: {FRONTEND_DIST}")

    if BACKEND_UI_STATIC.exists():
        shutil.rmtree(BACKEND_UI_STATIC)
    BACKEND_UI_STATIC.mkdir(parents=True, exist_ok=True)

    print(f"[frontend] Copying dist/ -> {BACKEND_UI_STATIC}")
    for item in dist_items:
        target = BACKEND_UI_STATIC / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def build_python_package():
    """Run python -m build to create sdist + wheel in ./dist/."""
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)

    print("[python] Building package with python -m build")
    run([sys.executable, "-m", "build"])


def create_fresh_venv():
    """Delete old venv and create a new one."""
    if VENV_DIR.exists():
        print(f"[venv] Removing existing venv at {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    print(f"[venv] Creating venv at {VENV_DIR}")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])

    if os.name == "nt":
        python_path = VENV_DIR / "Scripts" / "python.exe"
    else:
        python_path = VENV_DIR / "bin" / "python"

    return python_path


def install_wheel_in_venv(venv_python: Path):
    """Install the freshly built wheel into the venv."""
    wheels = sorted(DIST_DIR.glob("aethergraph-*.whl"))
    if not wheels:
        raise RuntimeError("No aethergraph-*.whl found in dist/")

    wheel = wheels[-1]  # most recent
    print(f"[venv] Using wheel: {wheel.name}")

    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(venv_python), "-m", "pip", "install", str(wheel)])


def smoke_test_import(venv_python: Path):
    """Run a tiny import test inside the venv."""
    code = r"""
import aethergraph
import sys

print("✅ aethergraph imported OK")
print("   module:", aethergraph.__file__)
version = getattr(aethergraph, "__version__", None)
print("   version:", version)
"""
    run([str(venv_python), "-c", code])


def smoke_test_server(venv_python: Path):
    """Basic server sanity checks: import entrypoint and run CLI help."""
    # 1) Import server entrypoint to ensure it’s packaged correctly
    code_import = r"""
from aethergraph import start_server
print("✅ start_server imported OK:", start_server)
"""
    run([str(venv_python), "-c", code_import])

    # 2) Run CLI help for `aethergraph serve` to ensure argument parsing works
    # This uses the console script entry point via -m to stay inside the venv.
    code_cli = r"""
import sys
from aethergraph.__main__ import main

# Simulate: aethergraph serve --help
sys.argv = ["aethergraph", "serve", "--help"]
print("✅ running `aethergraph serve --help`")
main()
"""
    run([str(venv_python), "-c", code_cli])


def main():
    args = parse_args()
    print(f"[info] Project root: {ROOT}")

    if args.ui_build:
        build_frontend(args.ui_app)
    else:
        print("[frontend] --ui-build not set, skipping frontend build step")

    build_python_package()
    venv_python = create_fresh_venv()
    install_wheel_in_venv(venv_python)
    smoke_test_import(venv_python)
    smoke_test_server(venv_python)

    print("\n🎉 Smoke test completed successfully.")


if __name__ == "__main__":
    main()

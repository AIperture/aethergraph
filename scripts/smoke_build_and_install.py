#!/usr/bin/env python
"""
End-to-end local smoke test:

1) (Optional) build frontend and copy into src/aethergraph/server/ui_static
2) python -m build
3) create fresh .venv-ag-test
4) pip install the built wheel into that venv
5) import aethergraph and print some basic info
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---- CONFIG: adjust paths if needed ----

ROOT = Path(__file__).resolve().parent.parent  # assuming script is in ./scripts/
FRONTEND_DIR = ROOT / "aethergraph-frontend"   # or whatever your frontend dir is
FRONTEND_DIST = FRONTEND_DIR / "dist"
BACKEND_UI_STATIC = ROOT / "src" / "aethergraph" / "server" / "ui_static"
DIST_DIR = ROOT / "dist"
VENV_DIR = ROOT / ".venv-ag-test"

BUILD_FRONTEND = False  # flip to True if you want frontend build in this flow


def run(cmd, cwd=None):
    """Run a command, print it, and fail fast on error."""
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def build_frontend():
    """Build the React frontend and copy dist/ into backend ui_static/."""
    if not FRONTEND_DIR.exists():
        print(f"[frontend] Skipping: {FRONTEND_DIR} does not exist.")
        return

    print("[frontend] Building frontend bundle...")
    run(["npm", "install"], cwd=FRONTEND_DIR)
    run(["npm", "run", "build", "--", "--base=/ui/"], cwd=FRONTEND_DIR)

    if BACKEND_UI_STATIC.exists():
        shutil.rmtree(BACKEND_UI_STATIC)
    BACKEND_UI_STATIC.mkdir(parents=True, exist_ok=True)

    print(f"[frontend] Copying dist/ -> {BACKEND_UI_STATIC}")
    for item in FRONTEND_DIST.iterdir():
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


def main():
    print(f"[info] Project root: {ROOT}")

    if BUILD_FRONTEND:
        build_frontend()
    else:
        print("[frontend] BUILD_FRONTEND=False, skipping frontend build step")

    build_python_package()
    venv_python = create_fresh_venv()
    install_wheel_in_venv(venv_python)
    smoke_test_import(venv_python)

    print("\n🎉 Smoke test completed successfully.")


if __name__ == "__main__":
    main()

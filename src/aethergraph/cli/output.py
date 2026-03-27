from __future__ import annotations

import json
from pathlib import Path

from aethergraph.server.loading import emit_load_errors

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"


def print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, default=str))


def print_load_intro(spec) -> None:
    print("=" * 50)
    print(f"{CYAN}{BOLD}[AetherGraph] 🔄 Loading graphs and agents...{RESET}")
    if spec.modules or spec.paths:
        print(
            f"➕ Importing modules:{RESET}",
            spec.modules,
            "and paths:",
            spec.paths,
            "at project root:",
            spec.project_root,
        )


def print_load_errors(errors, *, strict_load: bool) -> None:
    emit_load_errors(errors, strict_load=strict_load)


def print_load_complete() -> None:
    print(f"✅ Graph/agents loading complete.{RESET}")
    print("=" * 50)


def print_server_banner(*, url: str, workspace: str, log_path: Path, reload_enabled: bool) -> None:
    print("\n" + "=" * 50)
    print(f"{GREEN}{BOLD}[AetherGraph]{RESET} 🚀 Server started at: {url}")
    print(f"{GREEN}{BOLD}[AetherGraph]{RESET} 🖥️  UI: {url}/ui   (if built)")
    print(f"{GREEN}{BOLD}[AetherGraph]{RESET} 📡 API: {url}/api/v1/")
    print(f"{GREEN}{BOLD}[AetherGraph]{RESET} 📂 Workspace: {workspace}")
    print(f"{GREEN}{BOLD}[AetherGraph]{RESET} 🧩 Log Path: {log_path}")
    if reload_enabled:
        print(f"{GREEN}{BOLD}[AetherGraph]{RESET} ♻️  Auto-reload: enabled (uvicorn)")


def print_reload_watch_config(reload_dirs: list[str], reload_includes, reload_excludes) -> None:
    print(f"👀 Watching for changes in dirs:{RESET} {reload_dirs}")
    print(f"👀 Auto-reload include patterns:{RESET} {reload_includes or 'uvicorn defaults'}")
    print(f"👀 Auto-reload exclude patterns:{RESET} {reload_excludes or 'uvicorn defaults'}")
    print("=" * 50 + "\n")

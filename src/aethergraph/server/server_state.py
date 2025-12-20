from __future__ import annotations

from contextlib import contextmanager, suppress
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

STATE_DIR_NAME = ".aethergraph"
STATE_FILE_NAME = "server.json"
LOCK_FILE_NAME = "server.lock"


def _state_dir(workspace: str | Path) -> Path:
    return Path(workspace).resolve() / STATE_DIR_NAME


def state_file_path(workspace: str | Path) -> Path:
    return _state_dir(workspace) / STATE_FILE_NAME


def lock_file_path(workspace: str | Path) -> Path:
    return _state_dir(workspace) / LOCK_FILE_NAME


def ensure_state_dir(workspace: str | Path) -> Path:
    d = _state_dir(workspace)
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def workspace_lock(workspace: str | Path, timeout_s: float = 10.0, poll_s: float = 0.1):
    """
    Cross-platform file lock:
      - Windows: msvcrt.locking
      - Unix: fcntl.flock

    Ensures only one server starts per workspace at a time.
    """
    ensure_state_dir(workspace)
    lp = lock_file_path(workspace)
    f = open(lp, "a+")  # noqa: SIM115 # keep handle open to hold the lock

    start = time.time()
    while True:
        try:
            if os.name == "nt":
                import msvcrt  # type: ignore

                # lock 1 byte
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl  # type: ignore

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as e:
            if time.time() - start > timeout_s:
                f.close()
                raise TimeoutError(f"Timed out acquiring lock for workspace: {workspace}") from e
            time.sleep(poll_s)

    try:
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt  # type: ignore

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl  # type: ignore

                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()


def _tcp_ping(host: str, port: int, timeout_s: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        # On Windows, os.kill(pid, 0) is supported in Python and checks existence.
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_server_state(workspace: str | Path) -> dict[str, Any] | None:
    p = state_file_path(workspace)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_server_state(workspace: str | Path, state: dict[str, Any]) -> None:
    ensure_state_dir(workspace)
    p = state_file_path(workspace)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


def clear_server_state(workspace: str | Path) -> None:
    p = state_file_path(workspace)
    if p.exists():
        with suppress(Exception):
            p.unlink()


def get_running_url_if_any(workspace: str | Path) -> str | None:
    """
    Returns URL if server.json exists AND the process/port looks alive.
    """
    st = read_server_state(workspace)
    if not st:
        return None
    host = st.get("host")
    port = st.get("port")
    url = st.get("url")
    pid = st.get("pid")

    if not isinstance(host, str) or not isinstance(url, str) or not isinstance(port, int):
        return None
    if not isinstance(pid, int):
        pid = -1

    # Strong check: port is responding
    if _tcp_ping(host, port):
        return url

    # If port dead, also consider pid dead -> clear stale file
    if pid != -1 and not _pid_alive(pid):
        clear_server_state(workspace)
    return None


def pick_free_port(requested: int) -> int:
    if requested != 0:
        return requested
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

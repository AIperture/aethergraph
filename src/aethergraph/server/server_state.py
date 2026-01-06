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


class PortInUseError(RuntimeError):
    def __init__(self, host: str, port: int):
        super().__init__(f"Port {host}:{port} is already in use by another process.")
        self.host = host
        self.port = port


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
    """
    Cross-platform check if a PID is alive.

    On Unix: uses os.kill(pid, 0).
    On Windows: uses OpenProcess + GetExitCodeProcess.
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        # Windows: use Win32 API instead of os.kill(pid, 0),
        # which can give WinError 87 / SystemError behavior.
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259

        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            wintypes.DWORD(pid),
        )
        if not handle:
            return False

        try:
            exit_code = wintypes.DWORD()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                # Failed to query exit code -> assume not alive
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)

    # POSIX: classic trick
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


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
    Returns URL if server.json exists AND it looks like *our* server is alive.

    If the port is in use by another process (PID dead but TCP ping works),
    raises PortInUseError so the caller can show a clearer message.
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

    pid_alive = pid > 0 and _pid_alive(pid)
    port_alive = _tcp_ping(host, port)

    # Case 1: PID + port both alive -> this really looks like our server
    if pid_alive and port_alive:
        return url

    # Case 2: PID dead but port alive -> someone else is using that port
    if (not pid_alive) and port_alive:
        # our server isn't there anymore, but the port is taken
        clear_server_state(workspace)  # stale state; don't reuse
        raise PortInUseError(host, port)

    # Case 3: both dead -> stale file
    if (not pid_alive) and (not port_alive):
        clear_server_state(workspace)
        return None

    # Case 4: PID alive but port not responding.
    # This can happen briefly if the process is starting up or shutting down.
    # For CLI UX, it's usually fine to treat it as "running" and let the user retry if needed.
    if pid_alive and not port_alive:
        return url

    # Fallback: be conservative and say "no running server"
    return None


def pick_free_port(requested: int) -> int:
    """
    Port selection strategy:

    - If requested != 0: respect the user's choice exactly.
    - If requested == 0: try our preferred dev ports first (8745–8748),
      and if all are taken, fall back to an OS-assigned ephemeral port.
    """
    if requested != 0:
        return requested

    # Preferred AetherGraph dev ports – unlikely to collide with Jupyter, mkdocs, etc.
    preferred_ports = (8745, 8746, 8747, 8748)

    for port in preferred_ports:
        # Only 127.0.0.1 is relevant here; the server binding uses the real host later.
        if not _tcp_ping("127.0.0.1", port):
            return port

    # All preferred ports taken – fall back to OS-assigned free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

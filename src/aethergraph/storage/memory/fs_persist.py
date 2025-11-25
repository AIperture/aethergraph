from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from aethergraph.contracts.services.memory import Event, Persistence


class FSPersistence(Persistence):
    """
    File-system based persistence for memory events + JSON blobs.

    - Events are written to:
        <base_dir>/mem/<run_id>/events/YYYY-MM-DD.jsonl

    - JSON docs are read/written via file:// URIs:
        file://relative/path.json  -> <base_dir>/relative/path.json
        file:///abs/path.json      -> /abs/path.json (not under base_dir)
    """

    def __init__(self, *, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self._lock = threading.RLock()

    # ---------- Event log (append-only JSONL) ----------

    async def append_event(self, run_id: str, evt: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        path = self.base_dir / "mem" / run_id / "events" / f"{day}.jsonl"

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = asdict(evt)
            # Drop None values but keep [] / {} / 0.
            data = {k: v for k, v in raw.items() if v is not None}
            line = json.dumps(data, ensure_ascii=False) + "\n"
            with self._lock, path.open("a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_write)

    # ---------- JSON blob helpers (file:// URIs) ----------

    def _uri_to_path(self, uri: str) -> Path:
        """
        Convert a file:// URI into a local Path, resolving *relative* paths
        against self.base_dir. Works cross-platform.
        """
        if not uri.startswith("file://"):
            raise ValueError(f"FSPersistence only supports file:// URIs, got {uri!r}")

        raw = uri[len("file://") :]

        # Windows: normalize file:///C:/... -> C:/...
        if (
            os.name == "nt"
            and raw.startswith("/")
            and len(raw) > 2
            and raw[1].isalpha()
            and raw[2] == ":"
        ):
            raw = raw[1:]

        p = Path(raw)

        # Relative paths are resolved under base_dir
        if not p.is_absolute():
            p = self.base_dir / p

        return p

    def _path_to_uri(self, path: Path) -> str:
        """
        Convert a local Path to canonical file:// URI with forward slashes.
        """
        p = path.resolve()
        s = p.as_posix()

        # Ensure absolute paths appear as file:///... (add leading slash on Windows)
        if p.is_absolute() and not s.startswith("/"):
            s = "/" + s

        return f"file://{s}"

    async def save_json(self, uri: str, obj: dict[str, Any]) -> str:
        """
        Save JSON to the location specified by a file:// URI.
        Returns the canonical file:// URI of the saved file.
        """
        path = self._uri_to_path(uri)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with self._lock, tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

        await asyncio.to_thread(_write)
        return self._path_to_uri(path)

    async def load_json(self, uri: str) -> dict[str, Any]:
        """
        Inverse of save_json: load JSON from a file:// URI.
        """
        path = self._uri_to_path(uri)

        def _read() -> dict[str, Any]:
            with self._lock, path.open("r", encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

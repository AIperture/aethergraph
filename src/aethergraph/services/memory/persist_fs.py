from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any

from aethergraph.contracts.services.memory import Event, Persistence


class FSPersistence(Persistence):
    def __init__(self, *, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)

    async def append_event(self, run_id: str, evt: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        rel = os.path.join("mem", run_id, "events", f"{day}.jsonl")
        path = os.path.join(self.base_dir, rel)

        def _write():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            raw = asdict(evt)
            # Drop None values to keep JSON lean, but retain empty lists/dicts and 0.
            data = {k: v for k, v in raw.items() if v is not None}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)

    def _uri_to_path(self, uri: str) -> Path:
        """
        Convert a file:// URI into a local Path, resolving relative paths
        against self.base_dir. Works on all OSes.
        """
        assert uri.startswith("file://"), f"FSPersistence only supports file://, got {uri!r}"

        # Strip scheme; what's left is our path part (absolute or relative)
        raw = uri[len("file://") :]

        # On Windows, normalize file:///C:/... -> C:/...
        if (
            os.name == "nt"
            and raw.startswith("/")
            and len(raw) > 2
            and raw[1].isalpha()
            and raw[2] == ":"
        ):
            raw = raw[1:]  # "/C:/Users/..." -> "C:/Users/..."

        p = Path(raw)

        # If it's not absolute, treat it as relative to base_dir
        if not p.is_absolute():
            p = self.base_dir / p

        return p

    def _path_to_uri(self, path: Path) -> str:
        """
        Convert a local Path to a canonical file:// URI with forward slashes.
        """
        p = path.resolve()
        s = p.as_posix()

        # Ensure absolute paths have leading "/" so we get file:///... form
        if p.is_absolute() and not s.startswith("/"):
            # Windows: "C:/Users/..." -> "/C:/Users/..."
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
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)

        await asyncio.to_thread(_write)

        saved_uri = self._path_to_uri(path)
        return saved_uri

    async def load_json(self, uri: str) -> dict[str, Any]:
        """
        Inverse of save_json: load a JSON object from a file:// URI.
        Accepts both relative (file://logs/foo.json) and absolute
        (file:///home/... or file:///C:/...) URIs.
        """
        path = self._uri_to_path(uri)

        def _read() -> dict[str, Any]:
            with path.open(encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

"""
Simple file-based channel adapter for logging events to local files. 
Channel key format:
    file:<relative_path>

This is an inform-only adapter; it does not support receiving messages. 

Use cases include:
- Logging events to local files for debugging or auditing.
- Storing conversation logs in a structured manner.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Set
import warnings

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent

@dataclass
class FileChannelAdapter(ChannelAdapter):
    """
    Simple inform-only file adapter.

    Channel key format:
      file:<relative_path>

    Examples:
      file:logs/default.log
      file:runs/exp_01.txt
    """

    root: Path  # base directory where logs will be stored

    # Capabilities: we mostly care about text; we log meta/rich as JSON if present
    capabilities: Set[str] = frozenset({"text", "rich", "file", "buttons"})

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path_for(self, channel_key: str) -> Path:
        # channel_key = "file:logs/default.log"
        try:
            _, rel = channel_key.split(":", 1)
        except ValueError:
            # fallback if someone passes just "file"
            rel = "logs/default.log"
        rel = rel or "logs/default.log"
        return (self.root / rel).resolve()

    def _format_line(self, event: OutEvent) -> str:
        ts = datetime.now(timezone.utc).isoformat()
        base = {
            "type": event.type,
            "channel": event.channel,
            "text": event.text,
            "meta": event.meta or {},
            "rich": event.rich or {},
        }
        # We keep the outer format human-readable, but include structured JSON as needed
        line = f"[{ts}] {event.type}: {event.text or ''}"
        extras: dict = {}
        if event.meta:
            extras["meta"] = event.meta
        if event.rich:
            extras["rich"] = event.rich
        if event.file:
            extras["file"] = {
                "name": event.file.get("filename") or event.file.get("name"),
                "mimetype": event.file.get("mimetype"),
            }
        if event.buttons:
            extras["buttons"] = {
                k: {
                    "label": b.label,
                    "value": b.value,
                    "url": b.url,
                    "style": b.style,
                }
                for k, b in event.buttons.items()
            }

        if extras:
            line += " | " + json.dumps(extras, ensure_ascii=False)

        return line + "\n"

    async def send(self, event: OutEvent) -> None:
        path = self._path_for(event.channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = self._format_line(event)

        # Simple sync write is fine for low-volume logging
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            # Best-effort; this is an inform-only channel
            warnings.warn(f"[FileChannelAdapter] Failed to write to {path}: {e}")

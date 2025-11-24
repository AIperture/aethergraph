import asyncio
from datetime import datetime
import json
from pathlib import Path
import threading
import time

from aethergraph.contracts.storage.event_log import EventLog


class FSEventLog(EventLog):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._log_path = self.root / "events.jsonl"

    async def append(self, evt: dict) -> None:
        def _write():
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            row = evt.copy()
            row.setdefault("ts", time.time())
            with self._lock, self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)

    async def query(
        self,
        *,
        scope_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
        tags: list[str] | None = None,
    ) -> list[dict]:
        if not self._log_path.exists():
            return []

        def _read():
            out: list[dict] = []
            t_min = since.timestamp() if since else None
            t_max = until.timestamp() if until else None

            with self._lock, self._log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ts = row.get("ts")
                    if t_min is not None and ts is not None and ts < t_min:
                        continue
                    if t_max is not None and ts is not None and ts > t_max:
                        continue
                    if scope_id is not None and row.get("scope_id") != scope_id:
                        continue
                    if kinds is not None and row.get("kind") not in kinds:
                        continue
                    if tags is not None:
                        # Check that all requested tags are present in the event's tags
                        row_tags = set(row.get("tags", []))
                        if not row_tags.issuperset(tags):
                            continue
                    out.append(row)
                    if limit is not None and len(out) >= limit:
                        break
            return out

        return await asyncio.to_thread(_read)

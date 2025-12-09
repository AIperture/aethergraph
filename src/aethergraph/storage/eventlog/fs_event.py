import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

from aethergraph.contracts.storage.event_log import EventLog


def _to_ts_float(v) -> float | None:
    """
    Normalize event ts field to a float UNIX timestamp.

    Supports:
      - float / int already
      - ISO 8601 string, e.g. '2025-11-27T19:48:09.758687+00:00'
      - ISO with 'Z' suffix, e.g. '2025-11-27T19:48:09Z'
    """
    if v is None:
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        try:
            s = v.replace("Z", "+00:00") if v.endswith("Z") else v
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.timestamp()
    return None


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

            # 🔹 Normalize ts to a float UNIX timestamp
            ts = _to_ts_float(row.get("ts"))
            if ts is None:
                ts = time.time()
            row["ts"] = ts

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
        offset: int = 0,
    ) -> list[dict]:
        # NOTE: FSEventLog reads the single events.jsonl file linearly, applies
        # all filters (scope_id, time window, kinds, tags) in Python, and then
        # slices via offset + limit.
        #
        # This is fine for dev/demo / low event volumes, but if event volume grows
        # we should move to SQLiteEventLog or another DB-backed implementation
        # and use indexed queries + keyset/offset pagination.
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

                    ts_val = _to_ts_float(row.get("ts"))

                    if t_min is not None and ts_val is not None and ts_val < t_min:
                        continue
                    if t_max is not None and ts_val is not None and ts_val > t_max:
                        continue
                    if scope_id is not None and row.get("scope_id") != scope_id:
                        continue
                    if kinds is not None and row.get("kind") not in kinds:
                        continue
                    if tags is not None:
                        row_tags = set(row.get("tags", []))
                        if not row_tags.issuperset(tags):
                            continue

                    out.append(row)
                    if limit is not None and len(out) >= limit:
                        break

            if offset > 0:
                out = out[offset:]
            if limit is not None:
                out = out[:limit]

            return out

        return await asyncio.to_thread(_read)

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
import time
from typing import Literal

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
                dt = dt.replace(tzinfo=UTC)
            return dt.timestamp()
        except Exception:
            return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=UTC)
        return v.timestamp()
    return None


class FSEventLog(EventLog):
    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._log_path = self.root / "events.jsonl"
        self._next_row_id = self._existing_row_count() + 1

    def _existing_row_count(self) -> int:
        if not self._log_path.is_file():
            return 0
        with self._log_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    async def append(self, evt: dict) -> int:
        """Append one JSONL event and return its persistent line cursor.

        Examples:
            Append a development event:
            ```python
            cursor = await event_log.append(event)
            ```

            Retain a cursor for a later query:
            ```python
            after_cursor = await event_log.append(scoped_event)
            ```

        Args:
            evt: Event mapping to normalize and append.

        Returns:
            int: Monotonic JSONL row cursor for this event-log instance.

        Notes:
            The filesystem backend is for local low-volume use; SQLite is the
            durable concurrent Host backend.
        """

        def _write() -> int:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            row = evt.copy()
            partition_scope_id = row.pop("_partition_scope_id", row.get("scope_id"))

            # Normalize ts to a float UNIX timestamp
            ts = _to_ts_float(row.get("ts"))
            if ts is None:
                ts = time.time()
            row["ts"] = ts
            row["scope_id"] = partition_scope_id

            with self._lock, self._log_path.open("a", encoding="utf-8") as f:
                row_id = self._next_row_id
                self._next_row_id += 1
                row["_row_id"] = row_id
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                return row_id

        return await asyncio.to_thread(_write)

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
        user_id: str | None = None,
        org_id: str | None = None,
        client_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        after_id: int | None = None,
        before_id: int | None = None,
        order_dir: Literal["asc", "desc"] = "desc",
    ) -> list[dict]:
        """
        FSEventLog reads the single events.jsonl file linearly, applies
        all filters (scope_id, time window, kinds, tags, tenant) in Python,
        and then slices via offset + limit.

        This is fine for dev/demo / low event volumes. For production,
        prefer SQLiteEventLog or a DB-backed implementation.
        """
        if not self._log_path.exists():
            return []

        direction = "asc" if str(order_dir).lower() == "asc" else "desc"

        def _read() -> list[dict]:
            out: list[tuple[float, int, dict]] = []
            t_min = since.timestamp() if since else None
            t_max = until.timestamp() if until else None

            with self._lock, self._log_path.open("r", encoding="utf-8") as f:
                for seq, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row_id = int(row.get("_row_id") or (seq + 1))

                    if after_id is not None and row_id <= after_id:
                        continue
                    if before_id is not None and row_id >= before_id:
                        continue

                    ts_val = _to_ts_float(row.get("ts"))
                    sort_ts = ts_val if ts_val is not None else 0.0

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
                    if user_id is not None and row.get("user_id") != user_id:
                        continue
                    if org_id is not None and row.get("org_id") != org_id:
                        continue
                    if client_id is not None and row.get("client_id") != client_id:
                        continue
                    if session_id is not None and row.get("session_id") != session_id:
                        continue
                    if run_id is not None and row.get("run_id") != run_id:
                        continue
                    if agent_id is not None and row.get("agent_id") != agent_id:
                        continue
                    if graph_id is not None and row.get("graph_id") != graph_id:
                        continue
                    if node_id is not None and row.get("node_id") != node_id:
                        continue
                    if topic is not None and row.get("topic") != topic:
                        continue
                    if tool is not None and row.get("tool") != tool:
                        continue

                    sort_value = (
                        float(row_id) if after_id is not None or before_id is not None else sort_ts
                    )
                    out.append((sort_value, seq, row))

            out.sort(key=lambda item: (item[0], item[1]), reverse=direction == "desc")
            rows = [row for _, _, row in out]
            if offset > 0:
                rows = rows[offset:]
            if limit is not None:
                rows = rows[:limit]

            return rows

        return await asyncio.to_thread(_read)

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import threading
import time
from typing import Literal

from aethergraph.contracts.storage.event_log import EventLog, StateSnapshotConflictError
from aethergraph.storage.fs_utils import _exclusive_file_lock


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
        self._write_lock_path = self.root / ".events.lock"
        self._next_row_id = self._existing_row_count() + 1

    def _existing_row_count(self) -> int:
        if not self._log_path.is_file():
            return 0
        with self._log_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _prepare_row(self, evt: dict) -> dict:
        row = evt.copy()
        partition_scope_id = row.pop("_partition_scope_id", row.get("scope_id"))
        ts = _to_ts_float(row.get("ts"))
        row["ts"] = time.time() if ts is None else ts
        row["scope_id"] = partition_scope_id
        return row

    def _next_durable_row_id(self) -> int:
        if not self._log_path.is_file():
            return 1
        last_id = 0
        with self._log_path.open("r", encoding="utf-8") as handle:
            for seq, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                last_id = max(last_id, int(row.get("_row_id") or seq))
        return last_id + 1

    @staticmethod
    def _state_revision(row: dict | None) -> int:
        try:
            revision = (((row or {}).get("data") or {}).get("meta") or {}).get("revision", 0)
            return max(0, int(revision or 0))
        except (AttributeError, TypeError, ValueError):
            return 0

    def _append_row(self, row: dict) -> int:
        row_id = self._next_durable_row_id()
        row["_row_id"] = row_id
        data = {key: value for key, value in row.items() if value is not None}
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._next_row_id = max(self._next_row_id, row_id + 1)
        return row_id

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
            row = self._prepare_row(evt)
            with self._lock, _exclusive_file_lock(self._write_lock_path):
                return self._append_row(row)

        return await asyncio.to_thread(_write)

    async def append_state_snapshot_if_revision(
        self,
        evt: dict,
        *,
        state_key: str,
        expected_revision: int,
    ) -> int:
        """Compare and append a revisioned state snapshot under a file lock.

        Intro:
            One lock file serializes state comparisons and JSONL appends across
            local processes using the same filesystem event log.

        Examples:
            Append revision one:
            ```python
            cursor = await log.append_state_snapshot_if_revision(
                event,
                state_key="agent:writer",
                expected_revision=0,
            )
            ```

            Append revision two:
            ```python
            cursor = await log.append_state_snapshot_if_revision(
                next_event,
                state_key="agent:writer",
                expected_revision=1,
            )
            ```

        Args:
            evt: Complete state snapshot Event mapping.
            state_key: Exact logical state key carried by the snapshot.
            expected_revision: Exact current durable enclosing revision.

        Returns:
            int: Monotonic JSONL row cursor.

        Notes:
            A stale comparison raises `StateSnapshotConflictError` before append.
        """
        if isinstance(expected_revision, bool) or int(expected_revision) < 0:
            raise ValueError("expected_revision must be a non-negative integer")

        def _write() -> int:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            row = self._prepare_row(evt)
            proposed_revision = self._state_revision(row)
            if proposed_revision != int(expected_revision) + 1:
                raise ValueError("state snapshot revision must equal expected_revision + 1")
            with self._lock, _exclusive_file_lock(self._write_lock_path):
                actual_revision = 0
                if self._log_path.is_file():
                    with self._log_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            current = json.loads(line)
                            if current.get("scope_id") != row.get("scope_id"):
                                continue
                            if current.get("kind") != row.get("kind"):
                                continue
                            if f"state:{state_key}" not in (current.get("tags") or []):
                                continue
                            actual_revision = self._state_revision(current)
                if actual_revision != int(expected_revision):
                    raise StateSnapshotConflictError(
                        key=state_key,
                        expected_revision=int(expected_revision),
                        actual_revision=actual_revision,
                    )
                return self._append_row(row)

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

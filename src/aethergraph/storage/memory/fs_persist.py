from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from aethergraph.contracts.services.memory import Event, MemoryTenantFilter, Persistence
from aethergraph.contracts.storage.event_log import StateSnapshotConflictError
from aethergraph.services.memory.storage_filters import (
    event_matches_filters,
    event_time,
    summary_matches_filters,
)
from aethergraph.storage.fs_utils import _exclusive_file_lock


class FSPersistence(Persistence):
    """
    File-system based persistence for memory events + JSON blobs.
    """

    def __init__(self, *, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self._lock = threading.RLock()

    def _event_from_row(self, row: dict[str, Any]) -> Event:
        allowed = Event.__dataclass_fields__.keys()
        payload = {k: v for k, v in row.items() if k in allowed}
        return Event(**payload)

    async def append_event(self, timeline_id: str, evt: Event) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime())
        path = self.base_dir / "mem" / timeline_id / "events" / f"{day}.jsonl"

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = asdict(evt)
            raw["timeline_id"] = timeline_id
            data = {k: v for k, v in raw.items() if v is not None}
            line = json.dumps(data, ensure_ascii=False) + "\n"
            lock_path = self.base_dir / "mem" / timeline_id / ".events.lock"
            with self._lock, _exclusive_file_lock(lock_path), path.open("a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_write)

    async def append_state_snapshot_if_revision(
        self,
        timeline_id: str,
        evt: Event,
        *,
        state_key: str,
        expected_revision: int,
    ) -> None:
        """Conditionally append one filesystem state snapshot revision.

        Intro:
            The backend scans and appends the timeline while holding the same
            cross-process lock used by ordinary filesystem Memory writes.

        Examples:
            Append the first state snapshot:
            ```python
            await persistence.append_state_snapshot_if_revision(
                "session-1",
                event,
                state_key="agent:writer",
                expected_revision=0,
            )
            ```

            Append a later state snapshot:
            ```python
            await persistence.append_state_snapshot_if_revision(
                "session-1",
                next_event,
                state_key="agent:writer",
                expected_revision=1,
            )
            ```

        Args:
            timeline_id: Exact Memory timeline directory.
            evt: Complete revisioned state snapshot Event.
            state_key: Exact logical state key carried by the snapshot.
            expected_revision: Exact current durable enclosing revision.

        Returns:
            None: The snapshot is persisted as one JSONL row.

        Notes:
            A stale comparison raises `StateSnapshotConflictError` before append.
        """
        if isinstance(expected_revision, bool) or int(expected_revision) < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        day = time.strftime("%Y-%m-%d", time.gmtime())
        events_dir = self.base_dir / "mem" / timeline_id / "events"
        path = events_dir / f"{day}.jsonl"
        lock_path = self.base_dir / "mem" / timeline_id / ".events.lock"

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = asdict(evt)
            raw["timeline_id"] = timeline_id
            proposed_revision = self._state_revision(raw)
            if proposed_revision != int(expected_revision) + 1:
                raise ValueError("state snapshot revision must equal expected_revision + 1")
            with self._lock, _exclusive_file_lock(lock_path):
                actual_revision = 0
                for history_path in sorted(events_dir.glob("*.jsonl")):
                    with history_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            if not line.strip():
                                continue
                            current = json.loads(line)
                            if current.get("kind") != raw.get("kind"):
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
                data = {key: value for key, value in raw.items() if value is not None}
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(data, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)

    @staticmethod
    def _state_revision(row: dict[str, Any] | None) -> int:
        try:
            revision = (((row or {}).get("data") or {}).get("meta") or {}).get("revision", 0)
            return max(0, int(revision or 0))
        except (AttributeError, TypeError, ValueError):
            return 0

    def _uri_to_path(self, uri: str) -> Path:
        if not uri.startswith("file://"):
            raise ValueError(f"FSPersistence only supports file:// URIs, got {uri!r}")

        raw = uri[len("file://") :]
        if (
            os.name == "nt"
            and raw.startswith("/")
            and len(raw) > 2
            and raw[1].isalpha()
            and raw[2] == ":"
        ):
            raw = raw[1:]

        p = Path(raw)
        if not p.is_absolute():
            p = self.base_dir / p
        return p

    def _path_to_uri(self, path: Path) -> str:
        p = path.resolve()
        s = p.as_posix()
        if p.is_absolute() and not s.startswith("/"):
            s = "/" + s
        return f"file://{s}"

    async def save_json(self, uri: str, obj: dict[str, Any]) -> str:
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
        path = self._uri_to_path(uri)

        def _read() -> dict[str, Any]:
            with self._lock, path.open("r", encoding="utf-8") as f:
                return json.load(f)

        return await asyncio.to_thread(_read)

    async def get_events_by_ids(
        self,
        timeline_id: str,
        event_ids: list[str],
        tenant: MemoryTenantFilter | None = None,
    ) -> list[Event]:
        id_set = set(event_ids)
        events_dir = self.base_dir / "mem" / timeline_id / "events"
        if not events_dir.exists():
            return []

        def _read() -> list[Event]:
            results: list[Event] = []
            for path in sorted(events_dir.glob("*.jsonl")):
                with self._lock, path.open("r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if data.get("event_id") not in id_set:
                            continue
                        if not event_matches_filters(data, tenant=tenant):
                            continue
                        results.append(self._event_from_row(data))
            return results

        return await asyncio.to_thread(_read)

    async def query_events(
        self,
        timeline_id: str,
        *,
        tenant: MemoryTenantFilter | None = None,
        since: str | None = None,
        until: str | None = None,
        kinds: list[str] | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        client_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        topic: str | None = None,
        tool: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        order_dir: str = "desc",
    ) -> list[Event]:
        order_dir = "asc" if str(order_dir).lower() == "asc" else "desc"
        events_dir = self.base_dir / "mem" / timeline_id / "events"
        if not events_dir.exists():
            return []

        def _read() -> list[Event]:
            out: list[tuple[datetime, int, Event]] = []
            seq = 0
            min_dt = datetime.min.replace(tzinfo=UTC)
            for path in sorted(events_dir.glob("*.jsonl")):
                with self._lock, path.open("r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        if not event_matches_filters(
                            data,
                            tenant=tenant,
                            kinds=kinds,
                            tags=tags,
                            since=since,
                            until=until,
                            session_id=session_id,
                            run_id=run_id,
                            agent_id=agent_id,
                            client_id=client_id,
                            graph_id=graph_id,
                            node_id=node_id,
                            topic=topic,
                            tool=tool,
                        ):
                            continue
                        event = self._event_from_row(data)
                        out.append((event_time(event) or min_dt, seq, event))
                        seq += 1
            out.sort(key=lambda item: (item[0], item[1]), reverse=order_dir == "desc")
            events = [event for _, _, event in out]
            if offset:
                events = events[offset:]
            if limit is not None:
                events = events[:limit]
            return events

        return await asyncio.to_thread(_read)

    async def query_summaries(
        self,
        *,
        scope_id: str | None = None,
        timeline_id: str | None = None,
        tenant: MemoryTenantFilter | None = None,
        summary_tag: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        mem_root = self.base_dir / "mem"
        if not mem_root.exists():
            return []

        def _read() -> list[dict[str, Any]]:
            docs: list[dict[str, Any]] = []
            if scope_id:
                candidate_dirs = [mem_root / scope_id / "summaries"]
            elif timeline_id:
                candidate_dirs = [mem_root / timeline_id / "summaries"]
            else:
                candidate_dirs = list(mem_root.glob("*/summaries"))

            for summaries_dir in candidate_dirs:
                if not summaries_dir.exists():
                    continue
                for path in summaries_dir.rglob("*.json"):
                    with self._lock, path.open("r", encoding="utf-8") as f:
                        doc = json.load(f)
                    if not isinstance(doc, dict):
                        continue
                    if not summary_matches_filters(
                        doc,
                        tenant=tenant,
                        scope_id=scope_id,
                        summary_tag=summary_tag,
                    ):
                        continue
                    docs.append(doc)

            docs.sort(key=lambda doc: str(doc.get("ts") or doc.get("created_at") or ""))
            if offset:
                docs = docs[offset:]
            if limit is not None:
                docs = docs[:limit]
            return docs

        return await asyncio.to_thread(_read)

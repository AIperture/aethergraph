from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from aethergraph.contracts.services.memory import Event, MemoryTenantFilter, Persistence
from aethergraph.services.memory.storage_filters import (
    event_matches_filters,
    event_time,
    summary_matches_filters,
)


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
            with self._lock, path.open("a", encoding="utf-8") as f:
                f.write(line)

        await asyncio.to_thread(_write)

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
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Event]:
        events_dir = self.base_dir / "mem" / timeline_id / "events"
        if not events_dir.exists():
            return []

        def _read() -> list[Event]:
            out: list[Event] = []
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
                        ):
                            continue
                        out.append(self._event_from_row(data))
            out.sort(key=lambda e: (event_time(e), e.event_id))
            if offset:
                out = out[offset:]
            if limit is not None:
                out = out[:limit]
            return out

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

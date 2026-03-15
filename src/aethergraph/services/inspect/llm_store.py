from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol


def _parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _sanitize_preview_row(row: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(row)
    sanitized.pop("messages", None)
    sanitized.pop("raw_text", None)
    sanitized.pop("trace_payload", None)
    return sanitized


class LLMObservationStore(Protocol):
    async def query(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]: ...

    async def get(self, call_id: str) -> dict[str, Any] | None: ...


class JsonlLLMObservationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    async def query(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        app_id: str | None = None,
        graph_id: str | None = None,
        node_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self._read_rows()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            created_at = _parse_iso_ts(row.get("created_at"))
            if run_id is not None and row.get("run_id") != run_id:
                continue
            if session_id is not None and row.get("session_id") != session_id:
                continue
            if agent_id is not None and row.get("agent_id") != agent_id:
                continue
            if app_id is not None and row.get("app_id") != app_id:
                continue
            if graph_id is not None and row.get("graph_id") != graph_id:
                continue
            if node_id is not None and row.get("node_id") != node_id:
                continue
            if user_id is not None and row.get("user_id") != user_id:
                continue
            if org_id is not None and row.get("org_id") != org_id:
                continue
            if since is not None and created_at is not None and created_at < since:
                continue
            if until is not None and created_at is not None and created_at > until:
                continue
            filtered.append(_sanitize_preview_row(row))
        filtered.sort(key=lambda row: row.get("created_at") or "", reverse=True)
        if offset > 0:
            filtered = filtered[offset:]
        if limit is not None:
            filtered = filtered[:limit]
        return filtered

    async def get(self, call_id: str) -> dict[str, Any] | None:
        for row in self._read_rows():
            if row.get("call_id") == call_id:
                return dict(row)
        return None

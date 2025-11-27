from datetime import datetime, timedelta, timezone
from typing import Any

from aethergraph.contracts.services.metering import MeteringService, MeteringStore


class EventLogMeteringService(MeteringService):
    """
    MeteringService implementation backed by a MeteringStore (which itself
    is backed by an EventLog).
    """

    def __init__(self, store: MeteringStore):
        self._store = store

    # ---------- helpers -----------

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_window(window: str) -> datetime:
        if not window:
            return datetime.min.replace(tzinfo=timezone.utc)

        unit = window[-1]
        try:
            value = int(window[:-1])
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

        if unit == "h":
            delta = timedelta(hours=value)
        elif unit == "d":
            delta = timedelta(days=value)
        else:
            delta = timedelta(0)

        return datetime.now(timezone.utc) - delta

    async def _query(
        self,
        *,
        window: str,
        kinds: list[str],
        user_id: str | None,
        org_id: str | None,
    ) -> list[dict[str, Any]]:
        cutoff = self._parse_window(window)
        rows = await self._store.query(
            since=cutoff,
            until=None,
            kinds=kinds,
            limit=None,
        )
        # In-store filter by user/org if present
        out: list[dict[str, Any]] = []

        for e in rows:
            if user_id == "local" or org_id == "local":
                # Special case: include all local events
                out.append(e)
                continue
            if user_id is not None and e.get("user_id") != user_id:
                continue
            if org_id is not None and e.get("org_id") != org_id:
                continue
            out.append(e)

        return out

    async def _append(self, event: dict[str, Any]) -> None:
        # Ensure ts is always set
        event.setdefault("ts", self._now().isoformat())
        await self._store.append(event)

    # ---------- record_* methods ----------

    async def record_llm(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int | None = None,
    ) -> None:
        await self._append(
            {
                "kind": "meter.llm",
                "user_id": user_id,
                "org_id": org_id,
                "run_id": run_id,
                "model": model,
                "provider": provider,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "latency_ms": int(latency_ms) if latency_ms is not None else None,
                "tags": ["meter.llm"],
            }
        )

    async def record_run(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        status: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        await self._append(
            {
                "kind": "meter.run",
                "user_id": user_id,
                "org_id": org_id,
                "run_id": run_id,
                "graph_id": graph_id,
                "status": status,
                "duration_s": float(duration_s) if duration_s is not None else None,
                "tags": ["meter.run"],
            }
        )

    async def record_artifact(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        kind: str,
        bytes: int,
        pinned: bool = False,
    ) -> None:
        await self._append(
            {
                "kind": "meter.artifact",
                "user_id": user_id,
                "org_id": org_id,
                "run_id": run_id,
                "graph_id": graph_id,
                "artifact_kind": kind,
                "bytes": int(bytes),
                "pinned": bool(pinned),
                "tags": ["meter.artifact"],
            }
        )

    async def record_event(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        scope_id: str | None = None,
        kind: str,
    ) -> None:
        await self._append(
            {
                "kind": "meter.event",
                "user_id": user_id,
                "org_id": org_id,
                "run_id": run_id,
                "event_kind": kind,
                "scope_id": scope_id,
                "tags": ["meter.event"],
            }
        )

    # ---------- read methods ----------

    async def get_overview(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
    ) -> dict[str, int]:
        llm = await self._query(
            window=window,
            kinds=["meter.llm"],
            user_id=user_id,
            org_id=org_id,
        )

        # runs = await self._query(
        #     window=window,
        #     kinds=["meter.run"],
        #     user_id=user_id,
        #     org_id=org_id,
        # )
        # artifacts = await self._query(
        #     window=window,
        #     kinds=["meter.artifact"],
        #     user_id=user_id,
        #     org_id=org_id,
        # )
        # events = await self._query(
        #     window=window,
        #     kinds=["meter.event"],
        #     user_id=user_id,
        #     org_id=org_id,
        # )

        runs = []
        artifacts = []
        events = []

        return {
            "llm_calls": len(llm),
            "llm_prompt_tokens": sum(e.get("prompt_tokens", 0) for e in llm),
            "llm_completion_tokens": sum(e.get("completion_tokens", 0) for e in llm),
            "runs": len(runs),
            "runs_succeeded": sum(1 for e in runs if e.get("status") == "succeeded"),
            "runs_failed": sum(1 for e in runs if e.get("status") == "failed"),
            "artifacts": len(artifacts),
            "artifact_bytes": sum(e.get("bytes", 0) for e in artifacts),
            "events": len(events),
        }

    async def get_llm_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.llm"],
            user_id=user_id,
            org_id=org_id,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            model = e.get("model", "unknown")
            s = stats.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            s["calls"] += 1
            s["prompt_tokens"] += int(e.get("prompt_tokens", 0))
            s["completion_tokens"] += int(e.get("completion_tokens", 0))
        return stats

    async def get_graph_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.run"],
            user_id=user_id,
            org_id=org_id,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            graph_id = e.get("graph_id") or "unknown"
            s = stats.setdefault(
                graph_id, {"runs": 0, "succeeded": 0, "failed": 0, "total_duration_s": 0}
            )
            s["runs"] += 1
            if e.get("status") == "succeeded":
                s["succeeded"] += 1
            if e.get("status") == "failed":
                s["failed"] += 1
            s["total_duration_s"] += float(e.get("duration_s", 0.0))
        return stats

    async def get_artifact_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
    ) -> dict[str, dict[str, int]]:
        """
        Return aggregated stats by artifact kind, e.g.:

        {
          "json":  {"count": N, "bytes": B, "pinned_count": Pn, "pinned_bytes": Pb},
          "image": {...},
        }
        """
        rows = await self._query(
            window=window,
            kinds=["meter.artifact"],
            user_id=user_id,
            org_id=org_id,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            ak = e.get("artifact_kind") or "unknown"
            s = stats.setdefault(
                ak,
                {"count": 0, "bytes": 0, "pinned_count": 0, "pinned_bytes": 0},
            )
            b = int(e.get("bytes") or 0)
            pinned = bool(e.get("pinned") or False)

            s["count"] += 1
            s["bytes"] += b
            if pinned:
                s["pinned_count"] += 1
                s["pinned_bytes"] += b

        return stats

    async def get_memory_stats(
        self,
        *,
        scope_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.event"],
            user_id=user_id,
            org_id=org_id,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            if scope_id is not None and e.get("scope_id") != scope_id:
                continue

            ek = e.get("event_kind", "")
            if not ek.startswith("memory."):
                continue

            # Group by the full "memory.<kind>" for now
            s = stats.setdefault(ek, {"count": 0})
            s["count"] += 1
        return stats

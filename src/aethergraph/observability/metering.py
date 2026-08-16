from datetime import UTC, datetime, timedelta
from typing import Any

from aethergraph.contracts.services.metering import MeteringService, MeteringStore
from aethergraph.services.scope.scope import Scope

_MAX_METERING_QUERY = 10_000


class EventLogMeteringService(MeteringService):
    """
    MeteringService implementation backed by a MeteringStore (which itself
    is backed by an EventLog).

    Behavior notes:
        - All events are tagged with "meter" for easy filtering
        - Events are stored with their original metadata for later retrieval
        - if run_ids is provided in read methods, only events matching those run_ids are returned.
          This is useful for demo/multi-tenant isolation based on client-tagged runs.
    """

    def __init__(self, store: MeteringStore):
        self._store = store

    # ---------- helpers -----------

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _parse_window(window: str) -> datetime:
        if not window:
            return datetime.min.replace(tzinfo=UTC)

        unit = window[-1]
        try:
            value = int(window[:-1])
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)

        if unit == "h":
            delta = timedelta(hours=value)
        elif unit == "d":
            delta = timedelta(days=value)
        else:
            delta = timedelta(0)

        return datetime.now(UTC) - delta

    @staticmethod
    def _dims_from_scope(
        scope: Scope | None,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Merge identity/execution dimensions from Scope + explicit overrides.

        - If scope is None, we just return the explicit values.
        - If scope is present, its metering_dimensions() provide defaults,
          and explicit args win when provided.
        """
        if scope is None:
            return {
                "user_id": user_id,
                "org_id": org_id,
                "run_id": run_id,
                "graph_id": graph_id,
                "client_id": client_id,
                "app_id": app_id,
                "session_id": session_id,
            }

        s = scope.metering_dimensions()
        return {
            "user_id": user_id if user_id is not None else s.get("user_id"),
            "org_id": org_id if org_id is not None else s.get("org_id"),
            "run_id": run_id if run_id is not None else s.get("run_id"),
            "graph_id": graph_id if graph_id is not None else s.get("graph_id"),
            "client_id": client_id if client_id is not None else s.get("client_id"),
            "app_id": app_id if app_id is not None else s.get("app_id"),
            "session_id": session_id if session_id is not None else s.get("session_id"),
        }

    async def _query(
        self,
        *,
        window: str,
        kinds: list[str],
        user_id: str | None,
        org_id: str | None,
        run_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        cutoff = self._parse_window(window)

        # Dev/local: don't restrict by user/org at the store level
        store_user = None if user_id == "local" or org_id == "local" else user_id
        store_org = None if user_id == "local" or org_id == "local" else org_id

        rows = await self._store.query(
            since=cutoff,
            until=None,
            kinds=kinds,
            limit=_MAX_METERING_QUERY,
            user_id=store_user,
            org_id=store_org,
        )

        out: list[dict[str, Any]] = []

        for e in rows:
            if run_ids is not None:
                rid = e.get("run_id")
                if not rid or rid not in run_ids:
                    continue
                out.append(e)
                continue

            # For non-run_ids code paths, we've *already* filtered by user/org in SQL.
            # Keep the local special case (store_user=None/store_org=None) which
            # means "no tenant filter" for dev.
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
        scope: Scope | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        uncached_input_tokens: int = 0,
        latency_ms: int | None = None,
    ) -> None:
        dims = self._dims_from_scope(
            scope,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            graph_id=graph_id,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
        )
        await self._append(
            {
                "kind": "meter.llm",
                "user_id": dims["user_id"],
                "org_id": dims["org_id"],
                "client_id": dims["client_id"],
                "app_id": dims["app_id"],
                "session_id": dims["session_id"],
                "run_id": dims["run_id"],
                "graph_id": dims["graph_id"],
                "model": model,
                "provider": provider,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "cache_read_tokens": int(cache_read_tokens or 0),
                "cache_write_tokens": int(cache_write_tokens or 0),
                "uncached_input_tokens": int(uncached_input_tokens or 0),
                "latency_ms": int(latency_ms) if latency_ms is not None else None,
                "tags": ["meter.llm"],
            }
        )

    async def record_embedding(
        self,
        *,
        scope: Scope | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
        provider: str,
        model: str,
        num_texts: int,
        tokens: int | None = None,
        usage_availability: str = "unavailable",
        latency_ms: int | None = None,
    ) -> None:
        dims = self._dims_from_scope(
            scope,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            graph_id=graph_id,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
        )
        await self._append(
            {
                "kind": "meter.embedding",
                "user_id": dims["user_id"],
                "org_id": dims["org_id"],
                "client_id": dims["client_id"],
                "app_id": dims["app_id"],
                "session_id": dims["session_id"],
                "run_id": dims["run_id"],
                "graph_id": dims["graph_id"],
                "provider": provider,
                "model": model,
                "num_texts": int(num_texts),
                "tokens": int(tokens) if tokens is not None else None,
                "usage_availability": usage_availability,
                "latency_ms": int(latency_ms) if latency_ms is not None else None,
                "tags": ["meter.embedding"],
            }
        )

    async def record_image_generation(
        self,
        *,
        scope: Scope | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
        provider: str,
        model: str,
        image_count: int,
        size: str | None = None,
        quality: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        usage_availability: str = "unavailable",
        latency_ms: int | None = None,
    ) -> None:
        """Record one logical image-generation invocation.

        Intro:
            Writes image count, request dimensions, typed usage availability,
            and provider token counters to a dedicated meter event.

        Examples:
            Record a token-reporting image call:
                ```python
                await service.record_image_generation(
                    provider="openai",
                    model="gpt-image-1",
                    image_count=1,
                    input_tokens=4,
                    output_tokens=6,
                    total_tokens=10,
                    usage_availability="complete",
                )
                ```

            Record a provider without usage counters:
                ```python
                await service.record_image_generation(
                    provider="google",
                    model="image-model",
                    image_count=2,
                )
                ```

        Args:
            self: Event-log-backed metering service.
            scope: Optional canonical execution scope.
            user_id: Optional explicit user identity.
            org_id: Optional explicit organization identity.
            run_id: Optional explicit run identity.
            graph_id: Optional explicit graph identity.
            client_id: Optional explicit client identity.
            app_id: Optional explicit application identity.
            session_id: Optional explicit session identity.
            provider: Canonical provider identity.
            model: Image-generation model identity.
            image_count: Number of normalized images returned.
            size: Optional requested image dimensions.
            quality: Optional requested quality mode.
            input_tokens: Provider-reported input tokens.
            output_tokens: Provider-reported output tokens.
            total_tokens: Provider-reported or exactly derived total tokens.
            usage_availability: Complete, partial, or unavailable usage state.
            latency_ms: Logical invocation latency in milliseconds.

        Returns:
            None: Appends one `meter.image_generation` event.

        Notes:
            This event is intentionally distinct from `meter.llm`; retries do
            not create additional logical meter events.
        """

        dims = self._dims_from_scope(
            scope,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            graph_id=graph_id,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
        )
        await self._append(
            {
                "kind": "meter.image_generation",
                "user_id": dims["user_id"],
                "org_id": dims["org_id"],
                "client_id": dims["client_id"],
                "app_id": dims["app_id"],
                "session_id": dims["session_id"],
                "run_id": dims["run_id"],
                "graph_id": dims["graph_id"],
                "provider": provider,
                "model": model,
                "image_count": int(image_count),
                "size": size,
                "quality": quality,
                "input_tokens": int(input_tokens) if input_tokens is not None else None,
                "output_tokens": int(output_tokens) if output_tokens is not None else None,
                "total_tokens": int(total_tokens) if total_tokens is not None else None,
                "usage_availability": usage_availability,
                "latency_ms": int(latency_ms) if latency_ms is not None else None,
                "tags": ["meter.image_generation"],
            }
        )

    async def record_run(
        self,
        *,
        scope: Scope | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        duration_s: float | None = None,
    ) -> None:
        dims = self._dims_from_scope(
            scope,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            graph_id=graph_id,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
        )

        await self._append(
            {
                "kind": "meter.run",
                "user_id": dims["user_id"],
                "org_id": dims["org_id"],
                "client_id": dims["client_id"],
                "app_id": dims["app_id"],
                "session_id": dims["session_id"],
                "run_id": dims["run_id"],
                "graph_id": dims["graph_id"],
                "status": status,
                "duration_s": float(duration_s) if duration_s is not None else None,
                "tags": ["meter.run"],
            }
        )

    async def record_artifact(
        self,
        *,
        scope: Scope | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
        kind: str,
        bytes: int,
        pinned: bool = False,
    ) -> None:
        dims = self._dims_from_scope(
            scope,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            graph_id=graph_id,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
        )
        await self._append(
            {
                "kind": "meter.artifact",
                "user_id": dims["user_id"],
                "org_id": dims["org_id"],
                "client_id": dims["client_id"],
                "app_id": dims["app_id"],
                "session_id": dims["session_id"],
                "run_id": dims["run_id"],
                "graph_id": dims["graph_id"],
                "artifact_kind": kind,
                "bytes": int(bytes),
                "pinned": bool(pinned),
                "tags": ["meter.artifact"],
            }
        )

    async def record_event(
        self,
        *,
        scope: Scope | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        client_id: str | None = None,
        app_id: str | None = None,
        session_id: str | None = None,
        scope_id: str | None = None,
        kind: str,
    ) -> None:
        dims = self._dims_from_scope(
            scope,
            user_id=user_id,
            org_id=org_id,
            run_id=run_id,
            graph_id=None,
            client_id=client_id,
            app_id=app_id,
            session_id=session_id,
        )

        await self._append(
            {
                "kind": "meter.event",
                "user_id": dims["user_id"],
                "org_id": dims["org_id"],
                "client_id": dims["client_id"],
                "app_id": dims["app_id"],
                "session_id": dims["session_id"],
                "run_id": dims["run_id"],
                "event_kind": kind,
                "scope_id": scope_id,
                "tags": ["meter.event"],
            }
        )

    # ---------- read methods (unchanged for now) ----------

    async def get_overview(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: set[str] | None = None,
    ) -> dict[str, int]:
        llm = await self._query(
            window=window,
            kinds=["meter.llm"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        embeddings = await self._query(
            window=window,
            kinds=["meter.embedding"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        images = await self._query(
            window=window,
            kinds=["meter.image_generation"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )

        runs = await self._query(
            window=window,
            kinds=["meter.run"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        artifacts = await self._query(
            window=window,
            kinds=["meter.artifact"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        events = await self._query(
            window=window,
            kinds=["meter.event"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )

        return {
            "llm_calls": len(llm),
            "llm_prompt_tokens": sum(e.get("prompt_tokens", 0) for e in llm),
            "llm_completion_tokens": sum(e.get("completion_tokens", 0) for e in llm),
            "llm_cache_read_tokens": sum(e.get("cache_read_tokens", 0) for e in llm),
            "llm_cache_write_tokens": sum(e.get("cache_write_tokens", 0) for e in llm),
            "llm_uncached_input_tokens": sum(e.get("uncached_input_tokens", 0) for e in llm),
            "embedding_calls": len(embeddings),
            "embedding_texts": sum(e.get("num_texts", 0) for e in embeddings),
            "embedding_tokens": sum(e.get("tokens", 0) or 0 for e in embeddings),
            "image_generation_calls": len(images),
            "images_generated": sum(e.get("image_count", 0) or 0 for e in images),
            "image_generation_tokens": sum(e.get("total_tokens", 0) or 0 for e in images),
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
        run_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.llm"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            model = e.get("model", "unknown")
            s = stats.setdefault(
                model,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "uncached_input_tokens": 0,
                },
            )
            s["calls"] += 1
            s["prompt_tokens"] += int(e.get("prompt_tokens", 0))
            s["completion_tokens"] += int(e.get("completion_tokens", 0))
            s["cache_read_tokens"] += int(e.get("cache_read_tokens", 0))
            s["cache_write_tokens"] += int(e.get("cache_write_tokens", 0))
            s["uncached_input_tokens"] += int(e.get("uncached_input_tokens", 0))
        return stats

    async def get_embedding_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.embedding"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            model = e.get("model", "unknown")
            s = stats.setdefault(model, {"calls": 0, "num_texts": 0, "tokens": 0})
            s["calls"] += 1
            s["num_texts"] += int(e.get("num_texts", 0))
            s["tokens"] += int(e.get("tokens", 0) or 0)
        return stats

    async def get_image_generation_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Aggregate image-generation usage by model.

        Intro:
            Reads only dedicated image-generation meter events and aggregates
            logical calls, returned images, and reported tokens.

        Examples:
            Read the default window:
                ```python
                stats = await service.get_image_generation_stats()
                ```

            Restrict aggregation to selected runs:
                ```python
                stats = await service.get_image_generation_stats(
                    window="7d", run_ids={"run-1"}
                )
                ```

        Args:
            self: Event-log-backed metering service.
            user_id: Optional user filter.
            org_id: Optional organization filter.
            window: Relative query window such as `24h` or `7d`.
            run_ids: Optional exact run filter.

        Returns:
            dict[str, dict[str, int]]: Per-model call, image, and token totals.

        Notes:
            Unavailable token receipts contribute zero tokens but still count
            the logical call and returned images.
        """

        rows = await self._query(
            window=window,
            kinds=["meter.image_generation"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        stats: dict[str, dict[str, int]] = {}
        for event in rows:
            model = event.get("model", "unknown")
            values = stats.setdefault(
                model,
                {"calls": 0, "images": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
            values["calls"] += 1
            values["images"] += int(event.get("image_count", 0) or 0)
            values["input_tokens"] += int(event.get("input_tokens", 0) or 0)
            values["output_tokens"] += int(event.get("output_tokens", 0) or 0)
            values["total_tokens"] += int(event.get("total_tokens", 0) or 0)
        return stats

    async def get_graph_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.run"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
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
        run_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.artifact"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
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
        run_ids: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        rows = await self._query(
            window=window,
            kinds=["meter.event"],
            user_id=user_id,
            org_id=org_id,
            run_ids=run_ids,
        )
        stats: dict[str, dict[str, int]] = {}
        for e in rows:
            if scope_id is not None and e.get("scope_id") != scope_id:
                continue

            ek = e.get("event_kind", "")
            if not ek.startswith("memory."):
                continue

            s = stats.setdefault(ek, {"count": 0})
            s["count"] += 1
        return stats

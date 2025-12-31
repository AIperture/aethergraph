from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.memory import Event

# Assuming this external util exists based on original imports
from ..utils import _summary_prefix
from .utils import now_iso, stable_event_id

if TYPE_CHECKING:
    from .types import MemoryFacadeInterface


class DistillationMixin:
    """Methods for memory summarization and distillation."""

    async def load_last_summary(
        self: MemoryFacadeInterface, scope_id: str | None = None, *, summary_tag: str = "session"
    ) -> dict[str, Any] | None:
        scope_id = scope_id or self.memory_scope_id
        prefix = _summary_prefix(scope_id, summary_tag)

        try:
            ids = await self.docs.list()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"load_last_summary error: {e}")
            return None

        candidates = [d for d in ids if d.startswith(prefix)]
        if not candidates:
            return None

        latest_id = sorted(candidates)[-1]
        try:
            return await self.docs.get(latest_id)  # type: ignore
        except Exception:
            return None

    async def load_recent_summaries(
        self: MemoryFacadeInterface,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        scope_id = scope_id or self.memory_scope_id
        prefix = _summary_prefix(scope_id, summary_tag)

        try:
            ids = await self.docs.list()
        except Exception:
            return []

        candidates = sorted(d for d in ids if d.startswith(prefix))
        chosen = candidates[-limit:]

        out = []
        for doc_id in chosen:
            try:
                doc = await self.docs.get(doc_id)
                if doc:
                    out.append(doc)  # type: ignore
            except Exception:
                continue
        return out

    async def distill_long_term(
        self: MemoryFacadeInterface,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        scope_id = scope_id or self.memory_scope_id

        if use_llm:
            if not self.llm:
                raise RuntimeError("LLM client not configured")
            from aethergraph.services.memory.distillers.llm_long_term import LLMLongTermSummarizer

            d = LLMLongTermSummarizer(
                llm=self.llm,
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal if min_signal is not None else self.default_signal_threshold,
            )
        else:
            from aethergraph.services.memory.distillers.long_term import LongTermSummarizer

            d = LongTermSummarizer(
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal if min_signal is not None else self.default_signal_threshold,
            )

        return await d.distill(
            run_id=self.run_id,
            timeline_id=self.timeline_id,
            scope_id=scope_id or self.memory_scope_id,
            hotlog=self.hotlog,
            persistence=self.persistence,
            indices=self.indices,
            docs=self.docs,
        )

    async def soft_hydrate_last_summary(
        self: MemoryFacadeInterface,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
    ) -> dict[str, Any] | None:
        scope_id = scope_id or self.memory_scope_id
        summary = await self.load_last_summary(scope_id=scope_id, summary_tag=summary_tag)
        if not summary:
            return None

        text = summary.get("text") or ""
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        evt = Event(
            scope_id=self.memory_scope_id or self.run_id,
            event_id=stable_event_id(
                {
                    "ts": now_iso(),
                    "run_id": self.run_id,
                    "kind": f"{summary_kind}_hydrate",
                    "summary_tag": summary_tag,
                    "preview": preview[:200],
                }
            ),
            ts=now_iso(),
            run_id=self.run_id,
            kind=f"{summary_kind}_hydrate",
            stage="hydrate",
            text=preview,
            tags=["summary", "hydrate", summary_tag],
            data={"summary": summary},
            metrics={"num_events": summary.get("num_events", 0)},
            severity=1,
            signal=0.4,
        )

        await self.hotlog.append(self.timeline_id, evt, ttl_s=self.hot_ttl_s, limit=self.hot_limit)
        await self.persistence.append_event(self.timeline_id, evt)
        return summary

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.memory import Event
from aethergraph.services.scope.scope import ScopeLevel

# Assuming this external util exists based on original imports

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import MemoryFacadeProtocol


class DistillationMixin:
    """Methods for memory summarization and distillation."""

    async def _collect_events_for_distillation(
        self: MemoryFacadeProtocol,
        *,
        include_kinds: list[str] | None,
        include_tags: list[str] | None,
        max_events: int,
        level: ScopeLevel = "scope",
    ) -> list[Event]:
        """
        Collect candidate events for distillation.

        This helper prefers persisted history and falls back to hotlog when
        persistence views are unavailable.

        Examples:
            Collect default candidates for distillation:
            ```python
            events = await context.memory()._collect_events_for_distillation(
                include_kinds=None,
                include_tags=None,
                max_events=200,
            )
            ```

            Collect only tagged chat events at user scope:
            ```python
            events = await context.memory()._collect_events_for_distillation(
                include_kinds=["chat.turn"],
                include_tags=["important"],
                max_events=100,
                level="user",
            )
            ```

        Args:
            include_kinds: Optional event kinds to include.
            include_tags: Optional required tags.
            max_events: Maximum number of events to return after filtering.
            level: Scope level used for persisted/hotlog retrieval.

        Returns:
            list[Event]: Candidate events in chronological order.
        """
        overfetch_mult = 2
        if include_tags:
            overfetch_mult = 8

        fetch_limit = max(max_events * overfetch_mult, 200)

        try:
            events: list[Event] = await self.recent_persisted(
                kinds=include_kinds,
                tags=include_tags,
                limit=fetch_limit,
                level=level,
            )
        except Exception:
            events = await self.recent(
                kinds=include_kinds,
                limit=fetch_limit,
                level=level,
            )

        if not events:
            return []

        return events[-max_events:]

    async def distill_long_term(
        self: MemoryFacadeProtocol,
        *,
        level: ScopeLevel | None = None,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        include_kinds: list[str] | None = None,
        include_tags: list[str] | None = None,
        max_events: int = 200,
        min_signal: float | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Distill long-term memory summaries based on specified criteria.
        This method generates a long-term memory summary by either using a
        Long-Term Summarizer or an LLM-based Long-Term Summarizer, depending
        on the `use_llm` flag. The summaries are filtered and configured
        based on the provided arguments.

        Examples:
            Using the default summarizer:
            ```python
            result = await context.memory().distill_long_term(
                include_kinds=["note", "event"],
                max_events=100
            )
            ```
            Using an LLM-based summarizer:
            ```python
            result = await context.memory().distill_long_term(
                use_llm=True,
                summary_tag="custom_summary",
                min_signal=0.5
            )
            ```

        Args:

            summary_tag: A tag to categorize the generated summary. Defaults
                to `"session"`.
            summary_kind: The kind of summary to generate. Defaults to
                `"long_term_summary"`.
            include_kinds: A list of memory kinds to include in the summary.
                If None, all kinds are included.
            include_tags: A list of tags to filter the memories. If None, no
                tag filtering is applied.
            max_events: The maximum number of events to include in the
                summary. Defaults to 200.
            min_signal: The minimum signal threshold for filtering events.
                If None, the default signal threshold is used.
            use_llm: Whether to use an LLM-based summarizer. Defaults to False.

        Returns:
            dict[str, Any]: A dictionary containing the generated summary.

        Example return value:
            ```python
            {
                "uri": "file://mem/scope_123/summaries/long_term/2023-10-01T12:00:00Z.json",
                "summary_kind": "long_term_summary",
                "summary_tag": "session",
                "time_window": {"start": "2023-09-01", "end": "2023-09-30"},
                "num_events": 150,
                "included_kinds": ["note", "event"],
                "included_tags": ["important", "meeting"],
            }
            ```
        """

        eff_level = level or "scope"
        min_signal = min_signal if min_signal is not None else self.default_signal_threshold

        # 1) Collect candidate events
        events = await self._collect_events_for_distillation(
            include_kinds=include_kinds,
            include_tags=include_tags,
            max_events=max_events,
            level=eff_level,
        )
        if not events:
            return {}

        # Optional: filter by signal here if needed before passing to summarizer; summarizers may also apply their own filtering
        filtered: list[Event] = []
        for e in events:
            sig = getattr(e, "signal", None) or 0.0
            if sig < min_signal:
                continue
            filtered.append(e)
        if not filtered:
            return {}

        # 2) Choose summarizer
        if use_llm:
            if not self.llm:
                raise RuntimeError("LLM client not configured")
            from aethergraph.services.memory.distillers.llm_long_term import LLMLongTermSummarizer

            summarizer = LLMLongTermSummarizer(
                llm=self.llm,
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal,
            )
        else:
            from aethergraph.services.memory.distillers.long_term import LongTermSummarizer

            summarizer = LongTermSummarizer(
                summary_kind=summary_kind,
                summary_tag=summary_tag,
                include_kinds=include_kinds,
                include_tags=include_tags,
                max_events=max_events,
                min_signal=min_signal,
            )

        # 3) Let the summarizer transform the events into a structured payload
        summary = await summarizer.distill(events=filtered)

        if not summary:
            return {}

        # expected keys: summary, key_facts, open_loops, time_window, num_events, ...
        text = summary.get("summary", "") or summary.get("text", "")
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        # time_window = summary.get("time_window", {})
        num_events = summary.get("num_events", len(filtered))

        # 4) Record as a memory event; this writes to HotLog, Persistence, and indices
        stage = "summary_llm" if use_llm else "summary"
        tags = ["summary", summary_tag]
        if use_llm:
            tags.append("llm")

        evt = await self.record_raw(
            base={
                "kind": summary_kind,
                "stage": stage,
                "tags": tags,
                "data": summary,  # full summary object lives here
                "severity": 2,
                "signal": 0.7 if use_llm else None,
            },
            text=preview,
            metrics={"num_events": num_events},
        )

        # Attach event_id for downstream inspection
        summary["event_id"] = evt.event_id
        summary["summary_kind"] = summary_kind
        summary["summary_tag"] = summary_tag

        return summary

    async def distill_meta_summary(
        self,
        *,
        level: ScopeLevel | None = None,
        source_kind: str = "long_term_summary",
        source_tag: str = "session",
        summary_kind: str = "meta_summary",
        summary_tag: str = "meta",
        max_summaries: int = 20,
        min_signal: float | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a meta-summary by distilling existing summary events.

        This method creates a meta-summary by processing existing long-term
        summaries. It uses an LLM-based summarizer to generate a higher-level
        summary based on the provided arguments.

        Examples:
            Using the default configuration:
            ```python
            result = await context.memory().distill_meta_summary(
                source_kind="long_term_summary",
                source_tag="session",
            )
            ```

            Customizing the summary kind and tag:
            ```python
            result = await context.memory().distill_meta_summary(
                summary_kind="meta_summary",
                summary_tag="weekly",
                max_summaries=10,
            )
            ```

        Args:
            source_kind: The kind of source summaries to process. Defaults to
                `"long_term_summary"`.
            source_tag: A tag to filter the source summaries. Defaults to
                `"session"`.
            summary_kind: The kind of meta-summary to generate. Defaults to
                `"meta_summary"`.
            summary_tag: A tag to categorize the generated meta-summary.
                Defaults to `"meta"`.
            max_summaries: The maximum number of source summaries to process.
                Defaults to 20.
            min_signal: The minimum signal threshold for filtering summaries.
                If None, the default signal threshold is used.
            use_llm: Whether to use an LLM-based summarizer. Defaults to True.

        Returns:
            dict[str, Any]: A dictionary containing the generated meta-summary.

        Example return value:
            ```python
            {
                "uri": "file://mem/scope_123/summaries/meta/2023-10-01T12:00:00Z.json",
                "summary_kind": "meta_summary",
                "summary_tag": "meta",
                "time_window": {"start": "2023-09-01", "end": "2023-09-30"},
                "num_source_summaries": 15,
            }
            ```
        """
        eff_level = level or "scope"
        min_signal = min_signal if min_signal is not None else self.default_signal_threshold

        if not use_llm:
            raise NotImplementedError("Non-LLM meta summarization is not implemented yet")
        if not self.llm:
            raise RuntimeError("LLM client not configured in MemoryFacade for meta distillation")

        # 1) Fetch candidate summary events (persistence view)
        events = await self.recent_persisted(
            kinds=[source_kind],
            tags=["summary", source_tag],
            limit=max_summaries * 4,
            level=eff_level,
        )

        # return {}  # short-circuit for now
        if not events:
            return {}

        # sort by ts ascending and keep last max_summaries
        events = sorted(events, key=lambda e: e.ts)[-max_summaries:]

        # optional: filter by signal
        filtered = []
        for e in events:
            sig = getattr(e, "signal", None) or 0.0
            if sig < min_signal:
                continue
            filtered.append(e)
        if not filtered:
            return {}

        from aethergraph.services.memory.distillers.llm_meta_summary import LLMMetaSummaryDistiller

        d = LLMMetaSummaryDistiller(
            llm=self.llm,
            source_kind=source_kind,
            source_tag=source_tag,
            summary_kind=summary_kind,
            summary_tag=summary_tag,
            max_summaries=max_summaries,
            min_signal=min_signal,
        )

        summary = await d.distill(
            events=filtered,
        )

        if not summary:
            return {}

        text = summary.get("summary", "") or summary.get("text", "")
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        num_summaries = summary.get("num_source_summaries", len(filtered))
        time_window = summary.get("time_window", {})

        evt = await self.record_raw(
            base={
                "kind": summary_kind,
                "stage": "meta_summary_llm",
                "tags": ["summary", "llm", summary_tag],
                "data": summary,
                "severity": 2,
                "signal": 0.8,
            },
            text=preview,
            metrics={"num_source_summaries": num_summaries},
        )

        summary["event_id"] = evt.event_id
        summary["summary_kind"] = summary_kind
        summary["summary_tag"] = summary_tag
        summary["time_window"] = time_window

        return summary

    async def load_last_summary(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        level: ScopeLevel | None = "scope",
    ) -> dict[str, Any] | None:
        """
        Load the most recent JSON summary for the specified memory scope and tag.

        This method retrieves the latest summary document from the `DocStore`
        based on the provided `scope_id` and `summary_tag`. Summaries are
        identified using the following pattern:
        `mem/{scope_id}/summaries/{summary_tag}/{ts}`.

        Examples:
            Load the last session summary:
            ```python
            summary = await context.memory().load_last_summary(scope_id="user123", summary_tag="session")
            ```

            Load the last project summary:
            ```python
            summary = await context.memory().load_last_summary(scope_id="project456", summary_tag="project")
            ```

        Args:
            scope_id: Optional scope identifier. If None, uses the facade scope.
            summary_tag: The tag used to filter summaries (e.g., "session", "project").
                Defaults to "session".
            summary_kind: Summary event kind to load.
            level: Scope level used for persisted retrieval.

        Returns:
            dict[str, Any] | None: The most recent summary as a dictionary, or None if no summary is found.
        """
        scope_id = scope_id or self.memory_scope_id

        events = await self.recent_persisted(
            kinds=[summary_kind],
            tags=["summary", summary_tag],
            limit=1,
            level=level,
        )
        if not events:
            return None

        evt = events[-1]
        # Prefer structured data
        if evt.data:
            return evt.data  # type: ignore[return-value]

        # Fallback: reconstruct from text
        return {
            "summary": evt.text or "",
            "summary_kind": summary_kind,
            "summary_tag": summary_tag,
            "event_id": evt.event_id,
            "ts": evt.ts,
        }

    async def load_recent_summaries(
        self,
        *,
        summary_tag: str = "session",
        limit: int = 3,
        summary_kind: str = "long_term_summary",
        level: ScopeLevel | None = "scope",
    ) -> list[dict[str, Any]]:
        """
        Load the most recent JSON summaries for the specified scope and tag.

        This method retrieves up to `limit` summaries from the `DocStore`
        based on the provided `scope_id` and `summary_tag`. Summaries are
        identified using the following pattern:
        `mem/{scope_id}/summaries/{summary_tag}/{ts}`.

        Examples:
            Load the last three session summaries:
            ```python
            summaries = await context.memory().load_recent_summaries(
                summary_tag="session",
                limit=3
            )
            ```

            Load the last two project summaries:
            ```python
            summaries = await context.memory().load_recent_summaries(
                summary_tag="project",
                limit=2
            )
            ```

        Args:
            summary_tag: The tag used to filter summaries (e.g., "session", "project").
                Defaults to "session".
            limit: The maximum number of summaries to return. Defaults to 3.

        Returns:
            list[dict[str, Any]]: A list of summary dictionaries, ordered from oldest to newest.
        """
        events: list[Event] = await self.recent_persisted(
            kinds=[summary_kind],
            tags=["summary", summary_tag],
            limit=limit,
            level=level,
        )

        if not events:
            return []

        # Ensure chronological order
        events = sorted(events, key=lambda e: e.ts)
        out: list[dict[str, Any]] = []
        for evt in events:
            if evt.data:
                out.append(evt.data)  # type: ignore[arg-type]
            else:
                out.append(
                    {
                        "summary": evt.text or "",
                        "summary_kind": summary_kind,
                        "summary_tag": summary_tag,
                        "event_id": evt.event_id,
                        "ts": evt.ts,
                    }
                )
        return out

    async def soft_hydrate_last_summary(
        self,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
        level: ScopeLevel | None = "scope",
    ) -> dict[str, Any] | None:
        """
        Load the most recent summary for the specified scope and tag, and log a hydrate event.

        This method retrieves the latest summary document for the configured
        memory scope and `summary_tag`. If a summary is found, it logs a hydrate
        event into the current run's hotlog and persistence layers.

        Examples:
            Hydrate the last session summary:
            ```python
            summary = await context.memory().soft_hydrate_last_summary(
                summary_tag="session"
            )
            ```

        Args:
            summary_tag: The tag used to filter summaries (e.g., "session", "project").
                Defaults to "session".
            summary_kind: The kind of summary (e.g., "long_term_summary", "project_summary").
                Defaults to "long_term_summary".
            level: Scope level used to locate the latest summary.

        Returns:
            dict[str, Any] | None: The loaded summary dictionary if found, otherwise None.

        Side Effects:
            Appends a hydrate event to HotLog and Persistence for the current timeline.
        """
        summary: dict[str, Any] | None = await self.load_last_summary(
            summary_tag=summary_tag,
            summary_kind=summary_kind,
            level=level,
        )
        if not summary:
            return None

        text = summary.get("summary") or summary.get("text") or ""
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        await self.record_raw(
            base={
                "kind": f"{summary_kind}_hydrate",
                "stage": "hydrate",
                "tags": ["summary", "hydrate", summary_tag],
                "data": {"summary": summary},
                "severity": 1,
                "signal": 0.4,
            },
            text=preview,
            metrics={"num_events": summary.get("num_events", 0)},
        )

        return summary

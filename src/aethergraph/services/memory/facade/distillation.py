from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Assuming this external util exists based on original imports
from ..utils import _summary_prefix

if TYPE_CHECKING:
    from .types import MemoryFacadeInterface


class DistillationMixin:
    """Methods for memory summarization and distillation."""

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
                scope_id="scope_123",
                include_kinds=["note", "event"],
                max_events=100
            )
            ```
            Using an LLM-based summarizer:
            ```python
            result = await context.memory().distill_long_term(
                scope_id="scope_456",
                use_llm=True,
                summary_tag="custom_summary",
                min_signal=0.5
            )
            ```

        Args:
            scope_id: The scope ID for the memory to summarize. If None,
                defaults to the instance's `memory_scope_id`.
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

        result = await d.distill(
            run_id=self.run_id,
            timeline_id=self.timeline_id,
            scope_id=scope_id or self.memory_scope_id,
            hotlog=self.hotlog,
            docs=self.docs,
        )

        # If nothing returned, return empty dict
        if not result:
            return {}

        # Record the summary as a memory event via record_raw
        preview = result.get("preview", "")
        num_events = result.get("num_events", 0)
        time_window = result.get("time_window", {})

        # Use a different stage + tags depending on LLM or not
        stage = "summary_llm" if use_llm else "summary"
        tags = ["summary", summary_tag]
        if use_llm:
            tags.append("llm")

        evt = await self.record_raw(
            base={
                "kind": summary_kind,  # e.g. "long_term_summary"
                "stage": stage,  # "summary_llm" or "summary"
                "tags": tags,
                "data": {
                    "summary_doc_id": result.get("summary_doc_id"),
                    "summary_tag": summary_tag,
                    "time_window": time_window,
                    "num_events": num_events,
                },
                "scope_id": scope_id,
                # run_id / graph_id / node_id / session_id / user/org/client
                # etc. are filled in by record_raw from self.scope.
                "severity": 2,
                # optional: slight bias; record_raw will compute a default signal if None
                "signal": 0.7 if use_llm else None,
            },
            text=preview,
            metrics={"num_events": num_events},
        )

        # Optionally return the event_id with the result
        result["event_id"] = evt.event_id
        return result

    async def distill_meta_summary(
        self,
        scope_id: str | None = None,
        *,
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
                scope_id="scope_123",
                source_kind="long_term_summary",
                source_tag="session",
            )
            ```

            Customizing the summary kind and tag:
            ```python
            result = await context.memory().distill_meta_summary(
                scope_id="scope_456",
                summary_kind="meta_summary",
                summary_tag="weekly",
                max_summaries=10,
            )
            ```

        Args:
            scope_id: The scope ID for the memory to summarize. If None,
                defaults to the instance's `memory_scope_id`.
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
        scope_id = scope_id or self.memory_scope_id  # order of precedence

        if not use_llm:
            # Placeholder for a future non-LLM meta summarizer if desired.
            raise NotImplementedError("Non-LLM meta summarization is not implemented yet")

        if not self.llm:
            raise RuntimeError("LLM client not configured in MemoryFacade for meta distillation")

        from aethergraph.services.memory.distillers.llm_meta_summary import (
            LLMMetaSummaryDistiller,
        )

        d = LLMMetaSummaryDistiller(
            llm=self.llm,
            source_kind=source_kind,
            source_tag=source_tag,
            summary_kind=summary_kind,
            summary_tag=summary_tag,
            max_summaries=max_summaries,
            min_signal=min_signal if min_signal is not None else self.default_signal_threshold,
        )
        result = await d.distill(
            run_id=self.run_id,
            timeline_id=self.timeline_id,
            scope_id=scope_id or self.memory_scope_id,
            hotlog=self.hotlog,
            docs=self.docs,
        )

        # If nothing returned, return empty dict
        if not result:
            return {}
        # Record the meta-summary as a memory event via record_raw
        preview = result.get("preview", "")
        num_summaries = result.get("num_source_summaries", 0)
        time_window = result.get("time_window", {})
        evt = await self.record_raw(
            base={
                "kind": summary_kind,  # e.g. "meta_summary"
                "stage": "meta_summary_llm",
                "tags": ["summary", "llm", summary_tag],
                "data": {
                    "summary_doc_id": result.get("summary_doc_id"),
                    "summary_tag": summary_tag,
                    "time_window": time_window,
                    "num_source_summaries": num_summaries,
                },
                "scope_id": scope_id,
                # run_id / graph_id / node_id / session_id / user/org/client
                # etc. are filled in by record_raw from self.scope.
                "severity": 2,
                "signal": 0.8,
            },
            text=preview,
            metrics={"num_source_summaries": num_summaries},
        )
        result["event_id"] = evt.event_id
        return result

    async def load_last_summary(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
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
            scope_id: The memory scope ID. If None, defaults to the current memory scope.
            summary_tag: The tag used to filter summaries (e.g., "session", "project").
                Defaults to "session".

        Returns:
            dict[str, Any] | None: The most recent summary as a dictionary, or None if no summary is found.
        """
        scope_id = scope_id or self.memory_scope_id
        prefix = _summary_prefix(scope_id, summary_tag)

        try:
            ids = await self.docs.list()
        except Exception as e:
            self.logger and self.logger.warning("load_last_summary: doc_store.list() failed: %s", e)
            return None

        # Filter and take the latest
        candidates = [d for d in ids if d.startswith(prefix)]
        if not candidates:
            return None

        latest_id = sorted(candidates)[-1]
        try:
            return await self.docs.get(latest_id)  # type: ignore[return-value]
        except Exception as e:
            self.logger and self.logger.warning(
                "load_last_summary: failed to load %s: %s", latest_id, e
            )
            return None

    async def load_recent_summaries(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        limit: int = 3,
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
                scope_id="user123",
                summary_tag="session",
                limit=3
            )
            ```

            Load the last two project summaries:
            ```python
            summaries = await context.memory().load_recent_summaries(
                scope_id="project456",
                summary_tag="project",
                limit=2
            )
            ```

        Args:
            scope_id: The memory scope ID. If None, defaults to the current memory scope.
            summary_tag: The tag used to filter summaries (e.g., "session", "project").
                Defaults to "session".
            limit: The maximum number of summaries to return. Defaults to 3.

        Returns:
            list[dict[str, Any]]: A list of summary dictionaries, ordered from oldest to newest.
        """
        scope_id = scope_id or self.memory_scope_id
        prefix = _summary_prefix(scope_id, summary_tag)

        try:
            ids = await self.docs.list()
        except Exception as e:
            self.logger and self.logger.warning(
                "load_recent_summaries: doc_store.list() failed: %s", e
            )
            return []

        candidates = sorted(d for d in ids if d.startswith(prefix))
        if not candidates:
            return []

        chosen = candidates[-limit:]
        out: list[dict[str, Any]] = []
        for doc_id in chosen:
            try:
                doc = await self.docs.get(doc_id)
                if doc is not None:
                    out.append(doc)  # type: ignore[arg-type]
            except Exception:
                continue
        return out

    async def soft_hydrate_last_summary(
        self,
        scope_id: str | None = None,
        *,
        summary_tag: str = "session",
        summary_kind: str = "long_term_summary",
    ) -> dict[str, Any] | None:
        """
        Load the most recent summary for the specified scope and tag, and log a hydrate event.

        This method retrieves the latest summary document from the `DocStore` based on the
        provided `scope_id` and `summary_tag`. If a summary is found, it logs a hydrate
        event into the current run's HotLog and Persistence layers.

        Examples:
            Hydrate the last session summary:
            ```python
            summary = await context.memory().soft_hydrate_last_summary(
                scope_id="user123",
                summary_tag="session"
            )
            ```

        Args:
            scope_id: The memory scope ID. If None, defaults to the current memory scope.
            summary_tag: The tag used to filter summaries (e.g., "session", "project").
                Defaults to "session".
            summary_kind: The kind of summary (e.g., "long_term_summary", "project_summary").
                Defaults to "long_term_summary".

        Returns:
            dict[str, Any] | None: The loaded summary dictionary if found, otherwise None.

        Side Effects:
            Appends a hydrate event to HotLog and Persistence for the current timeline.
        """
        scope_id = scope_id or self.memory_scope_id
        summary = await self.load_last_summary(scope_id=scope_id, summary_tag=summary_tag)
        if not summary:
            return None

        text = summary.get("text") or summary.get("summary") or ""  # try both fields
        preview = text[:2000] + (" …[truncated]" if len(text) > 2000 else "")

        await self.record_raw(
            base={
                "kind": f"{summary_kind}_hydrate",
                "stage": "hydrate",
                "tags": ["summary", "hydrate", summary_tag],
                "data": {"summary": summary},
                "scope_id": scope_id,
                # run_id / graph_id / node_id / session_id / user/org/client
                # etc. are filled in by record_raw from self.scope.
                "severity": 1,
                "signal": 0.4,
            },
            text=preview,
            metrics={"num_events": summary.get("num_events", 0)},
        )

        return summary

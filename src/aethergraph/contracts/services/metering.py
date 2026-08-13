from datetime import datetime
from typing import Any, Protocol


class MeteringService(Protocol):
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
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        uncached_input_tokens: int = 0,
        latency_ms: int | None = None,
    ) -> None:
        """Record an LLM usage event."""
        ...

    async def record_embedding(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
        provider: str,
        model: str,
        num_texts: int,
        tokens: int | None = None,
        usage_availability: str = "unavailable",
        latency_ms: int | None = None,
    ) -> None:
        """Record an embedding usage event."""
        ...

    async def record_image_generation(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        graph_id: str | None = None,
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
        """Record one logical image-generation usage event.

        Intro:
            Defines the operation-specific meter boundary for returned images
            and provider-reported token counters.

        Examples:
            Record complete usage:
                ```python
                await meter.record_image_generation(
                    provider="openai", model="gpt-image-1", image_count=1
                )
                ```

            Record an unavailable receipt:
                ```python
                await meter.record_image_generation(
                    provider="google",
                    model="image-model",
                    image_count=2,
                    usage_availability="unavailable",
                )
                ```

        Args:
            self: Concrete metering service.
            user_id: Optional user identity.
            org_id: Optional organization identity.
            run_id: Optional run identity.
            graph_id: Optional graph identity.
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
            None: Records one logical operation event.

        Notes:
            Implementations must not project this call into ordinary Chat usage.
        """
        ...

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
        """Record a run usage event."""
        ...

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
        """Record an artifact usage event."""
        ...

    async def record_event(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        run_id: str | None = None,
        scope_id: str | None = None,
        kind: str,
    ) -> None:
        """Record an event usage event."""
        ...

    # ----- Read methods ----- #
    async def get_overview(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",  # e.g., "24h", "7d", "30d"
        run_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Get an overview of usage metrics."""
        ...

    async def get_llm_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get LLM usage statistics."""
        ...

    async def get_embedding_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get embedding usage statistics."""
        ...

    async def get_image_generation_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get image-generation usage statistics.

        Intro:
            Aggregates dedicated image-generation meter events without mixing
            them into Chat or embedding totals.

        Examples:
            Read the default window:
                ```python
                stats = await meter.get_image_generation_stats()
                ```

            Restrict results to one run:
                ```python
                stats = await meter.get_image_generation_stats(run_ids=["run-1"])
                ```

        Args:
            self: Concrete metering service.
            user_id: Optional user filter.
            org_id: Optional organization filter.
            window: Relative aggregation window.
            run_ids: Optional exact run filter.

        Returns:
            dict[str, dict[str, int]]: Per-model image-generation statistics.

        Notes:
            Unavailable token counters remain absent from token totals.
        """
        ...

    async def get_graph_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get graph usage statistics."""
        ...

    async def get_artifact_stats(
        self,
        *,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get artifact usage statistics."""
        ...

    async def get_memory_stats(
        self,
        *,
        scope_id: str | None = None,
        user_id: str | None = None,
        org_id: str | None = None,
        window: str = "24h",
        run_ids: list[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get memory usage statistics."""
        ...

    # Other possible methods -- channel events, embeddings, and tool calls


class MeteringStore(Protocol):
    async def append(self, event: dict[str, Any]) -> None: ...
    async def query(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        kinds: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

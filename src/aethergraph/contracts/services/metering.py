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
        latency_ms: int | None = None,
    ) -> None:
        """Record an LLM usage event."""
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

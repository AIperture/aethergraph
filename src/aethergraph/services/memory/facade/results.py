from __future__ import annotations

from typing import TYPE_CHECKING, Any
import warnings

from aethergraph.contracts.services.memory import Event

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import MemoryFacadeProtocol


class ResultMixin:
    """Methods for recording tool execution results."""

    async def record_tool_result(
        self: MemoryFacadeProtocol,
        *,
        tool: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event:
        """
        Record a tool execution result in normalized event form.

        Examples:
            Record a tool result:
            ```python
            evt = await context.memory().record_tool_result(
                tool="planner",
                inputs=[{"q": "status"}],
                outputs=[{"ok": True}],
                message="Planner completed.",
            )
            ```

        Args:
            tool: Tool identifier.
            inputs: Optional list of input payload dictionaries.
            outputs: Optional list of output payload dictionaries.
            tags: Optional tags for filtering/search.
            metrics: Optional numeric metrics.
            message: Optional human-readable summary text.
            severity: Event severity.

        Returns:
            Event: Persisted tool-result event.
        """
        return await self.record_raw(
            base={
                "tool": tool,
                "kind": "tool_result",
                "severity": severity,
                "tags": tags or [],
                "inputs": inputs or [],
                "outputs": outputs or [],
            },
            text=message,
            metrics=metrics,
        )

    async def recent_tool_results(
        self,
        *,
        tool: str,
        limit: int = 10,
        return_event: bool = True,
    ) -> list[Any]:
        """
        Retrieve recent tool-result events for a specific tool.

        Examples:
            Return Event objects:
            ```python
            rows = await context.memory().recent_tool_results(tool="planner", limit=5)
            ```

            Return normalized dictionaries:
            ```python
            rows = await context.memory().recent_tool_results(
                tool="planner",
                limit=5,
                return_event=False,
            )
            ```

        Args:
            tool: Tool name to filter by.
            limit: Maximum number of results.
            return_event: If True return Event objects; otherwise normalized dictionaries.

        Returns:
            list[Any]: Event rows or normalized dict payloads.
        """
        events = await self.recent(kinds=["tool_result"], limit=100, return_event=True)
        tool_events = [e for e in events if getattr(e, "tool", None) == tool]
        return self.normalize_recent_output(tool_events[:limit], return_event=return_event)

    async def recent_tool_result_data(
        self,
        *,
        tool: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Deprecated helper for simplified tool-result dictionaries.

        Examples:
            Legacy usage:
            ```python
            rows = await context.memory().recent_tool_result_data(tool="planner", limit=5)
            ```

        Args:
            tool: Tool name to filter by.
            limit: Maximum number of rows to return.

        Returns:
            list[dict[str, Any]]: Simplified tool-result dictionaries.
        """
        warnings.warn(
            "recent_tool_result_data() is deprecated and will be removed in a future version. "
            "Use recent_tool_results(..., return_event=False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        rows = await self.recent_tool_results(tool=tool, limit=limit, return_event=False)
        return rows  # type: ignore[return-value]

    async def write_result(
        self: MemoryFacadeProtocol,
        *,
        tool: str | None = None,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
        topic: str | None = None,
    ) -> Event:
        """
        Deprecated alias for `record_tool_result()`.

        Examples:
            Legacy usage with `topic` alias:
            ```python
            evt = await context.memory().write_result(topic="planner", message="done")
            ```

        Args:
            tool: Tool identifier.
            inputs: Tool inputs.
            outputs: Tool outputs.
            tags: Optional tags.
            metrics: Optional metrics.
            message: Optional summary text.
            severity: Event severity.
            topic: Legacy alias of `tool`.

        Returns:
            Event: Persisted tool-result event.
        """
        warnings.warn(
            "write_result() is deprecated and will be removed in a future version. "
            "Use record_tool_result().",
            DeprecationWarning,
            stacklevel=2,
        )
        if tool is None and topic is not None:
            tool = topic
        if tool is None:
            raise ValueError("write_result requires a 'tool' (or legacy 'topic') name")
        return await self.record_tool_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

    async def write_tool_result(
        self: MemoryFacadeProtocol,
        *,
        tool: str,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event:
        """
        Deprecated alias for `record_tool_result()`.

        Examples:
            Legacy usage:
            ```python
            evt = await context.memory().write_tool_result(tool="planner", message="done")
            ```

        Args:
            tool: Tool identifier.
            inputs: Tool inputs.
            outputs: Tool outputs.
            tags: Optional tags.
            metrics: Optional metrics.
            message: Optional summary text.
            severity: Event severity.

        Returns:
            Event: Persisted tool-result event.
        """
        warnings.warn(
            "write_tool_result() is deprecated and will be removed in a future version. "
            "Use record_tool_result().",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self.record_tool_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

    async def record_result(
        self: MemoryFacadeProtocol,
        *,
        tool: str | None = None,
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
    ) -> Event:
        """
        Deprecated alias for `record_tool_result()`.

        Examples:
            Legacy usage:
            ```python
            evt = await context.memory().record_result(tool="planner", message="done")
            ```

        Args:
            tool: Tool identifier.
            inputs: Tool inputs.
            outputs: Tool outputs.
            tags: Optional tags.
            metrics: Optional metrics.
            message: Optional summary text.
            severity: Event severity.

        Returns:
            Event: Persisted tool-result event.
        """
        warnings.warn(
            "record_result() is deprecated and will be removed in a future version. "
            "Use record_tool_result().",
            DeprecationWarning,
            stacklevel=2,
        )
        if tool is None:
            raise ValueError("record_result requires a 'tool' name")
        return await self.record_tool_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

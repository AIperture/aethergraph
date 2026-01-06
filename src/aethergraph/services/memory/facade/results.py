from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aethergraph.contracts.services.memory import Event

if TYPE_CHECKING:
    from .types import MemoryFacadeInterface


class ResultMixin:
    """Methods for recording tool execution results.
    NOTE: there are many potentially overlapping methods here. We will deprecate most of them
    over time in favor of a smaller, clearer set.

    Include methods:
    - write_result (general)
    - write_tool_result (deprecated, use record_tool_result)
    - record_result (general alias)
    - record_tool_result (dedicated to tools)
    - last_tool_result
    - recent_tool_result_data

    The following are convenience wrappers. TODO: standardize naming
    - last_by_name
    - last_output_by_name
    - last_outputs_by_topic
    - last_tool_result_outputs
    - latest_refs_by_kind
    """

    async def record_tool_result(
        self: MemoryFacadeInterface,
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
        Record the result of a tool execution in a normalized format.

        This method provides the method to log tool execution results with standardized metadata.
        Interally, it constructs an `Event` object encapsulating details about the tool execution,
        including inputs, outputs, tags, metrics, and a descriptive message.

        Examples:
            Recording a tool result with inputs and outputs:
            ```python
            await context.memory().record_tool_result(
                tool="data_cleaner",
                inputs=[{"raw_data": "some raw input"}],
                outputs=[{"cleaned_data": "processed output"}],
                tags=["data", "cleaning"],
                metrics={"execution_time": 1.23},
                message="Tool executed successfully.",
                severity=2,
            )
            ```

            Logging a tool result with minimal metadata:
            ```python
            await context.memory().record_tool_result(
                tool="simple_logger",
                message="Logged an event.",
            )
            ```

        Args:
            tool: The name of the tool that generated the result.
            inputs: A list of dictionaries representing the tool's input data.
            outputs: A list of dictionaries representing the tool's output data.
            tags: A list of string labels for categorization.
            metrics: A dictionary of numerical metrics (e.g., execution time, accuracy).
            message: A descriptive message about the tool's execution or result.
            severity: An integer (1-5) indicating the importance or severity of the result.
            (1=Lowest, 5=Highest).

        Returns:
            Event: The fully persisted `Event` object containing the generated ID and timestamp.
        """
        return await self.write_tool_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

    async def recent_tool_results(
        self,
        *,
        tool: str,
        limit: int = 10,
    ) -> list[Event]:
        """
        Retrieve recent tool execution results for a specific tool.

        This method filters and returns the most recent `tool_result` events
        associated with the specified tool, allowing you to analyze or process
        the results of tool executions.

        Examples:
            Fetching the 5 most recent results for a tool:
            ```python
            recent_results = await context.memory().recent_tool_results(
                tool="data_cleaner",
                limit=5,
            )
            for result in recent_results:
                print(result)
            ```

            Retrieving all available results for a tool (up to the default limit):
            ```python
            recent_results = await context.memory().recent_tool_results(
                tool="simple_logger",
            )
            ```

        Args:
            tool: The name of the tool whose results are being queried.
            limit: The maximum number of results to return (default is 10).

        Returns:
            list[Event]: A list of `Event` objects representing the recent
            `tool_result` events for the specified tool, ordered by recency.
        """
        events = await self.recent(kinds=["tool_result"], limit=100)
        tool_events = [e for e in events if e.tool == tool]
        return tool_events[:limit]

    async def recent_tool_result_data(
        self,
        *,
        tool: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Return a simplified view over recent tool_result events.

        This method provides a developer-friendly way to retrieve recent tool execution results
        in a normalized format, including metadata such as timestamps, inputs, outputs, and tags.

        Examples:
            Fetching recent tool result data:
            ```python
            recent_data = await context.memory().recent_tool_result_data(
                tool="data_cleaner",
                limit=5,
            )
            for entry in recent_data:
                print(entry)
            ```

        Args:
            tool: The name of the tool whose results are being queried.
            limit: The maximum number of recent results to retrieve.

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing:
                - "ts": The timestamp of the event.
                - "tool": The name of the tool.
                - "message": A descriptive message about the tool's execution.
                - "inputs": The input data provided to the tool.
                - "outputs": The output data generated by the tool.
                - "tags": A list of string labels associated with the event.
        """
        events = await self.recent_tool_results(tool=tool, limit=limit)
        out: list[dict[str, Any]] = []
        for e in events:
            out.append(
                {
                    "ts": getattr(e, "ts", None),
                    "tool": e.tool,
                    "message": e.text,
                    "inputs": getattr(e, "inputs", None),
                    "outputs": getattr(e, "outputs", None),
                    "tags": list(e.tags or []),
                }
            )
        return out

    async def write_result(
        self: MemoryFacadeInterface,
        *,
        tool: str | None = None,  # back compatibility with 'topic'
        inputs: list[dict[str, Any]] | None = None,
        outputs: list[dict[str, Any]] | None = None,
        tags: list[str] | None = None,
        metrics: dict[str, float] | None = None,
        message: str | None = None,
        severity: int = 3,
        topic: str | None = None,  # alias for tool, backwards compatibility
    ) -> Event:
        """
        Convenience for recording a “tool/agent/flow result” with typed I/O.

        `tool`    : tool/agent/flow identifier (also used by KVIndices.last_outputs_by_topic)
        `inputs`  : List[Value]-like dicts
        `outputs` : List[Value]-like dicts
        `tags`    : labels like ["rag","qa"] for filtering/search
        """
        if tool is None and topic is not None:
            tool = topic
        if tool is None:
            raise ValueError("write_result requires a 'tool' (or legacy 'topic') name")

        inputs = inputs or []
        outputs = outputs or []

        evt = await self.record_raw(
            base=dict(
                tool=tool,
                kind="tool_result",
                severity=severity,
                tags=tags or [],
                inputs=inputs,
                outputs=outputs,
            ),
            text=message,
            metrics=metrics,
        )
        await self.indices.update(self.timeline_id, evt)
        return evt

    async def write_tool_result(
        self: MemoryFacadeInterface,
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
        Convenience wrapper around write_result() for tool results.
        """
        return await self.write_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

    async def record_result(
        self: MemoryFacadeInterface,
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
        Alias for write_result(); symmetric with record_tool_result().

        Use this when you conceptually have a "result" but don't care whether
        it's a tool vs agent vs flow.
        """
        return await self.write_result(
            tool=tool,
            inputs=inputs,
            outputs=outputs,
            tags=tags,
            metrics=metrics,
            message=message,
            severity=severity,
        )

    async def last_tool_result(self, tool: str) -> Event | None:
        """
        Convenience: return the most recent tool_result Event for a given tool.
        """
        events = await self.recent_tool_results(tool=tool, limit=1)
        return events[-1] if events else None

    async def last_by_name(self, name: str):
        """Return the last output value by `name` from Indices (fast path)."""
        return await self.indices.last_by_name(self.timeline_id, name)

    async def last_output_by_name(self, name: str):
        """Return the last output value (Value.value) by `name` from Indices (fast path)."""
        out = await self.indices.last_by_name(self.timeline_id, name)
        if out is None:
            return None
        return out.get("value")  # type: ignore

    async def last_outputs_by_topic(self, topic: str):
        """Return the last output map for a given topic (tool/flow/agent) from Indices."""
        return await self.indices.last_outputs_by_topic(self.timeline_id, topic)

    # replace last_tool_result_outputs
    async def last_tool_result_outputs(self, tool: str) -> dict[str, Any] | None:
        """
        Convenience wrapper around KVIndices.last_outputs_by_topic for this run.
        Returns the last outputs map for a given tool, or None.
        """
        return await self.indices.last_outputs_by_topic(self.timeline_id, tool)

    async def latest_refs_by_kind(self, kind: str, *, limit: int = 50):
        """Return latest ref outputs by ref.kind (fast path, KV-backed)."""
        return await self.indices.latest_refs_by_kind(self.timeline_id, kind, limit=limit)

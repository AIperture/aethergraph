"""Provider-neutral typed streaming events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from .tool_calling import ModelResponse
from .usage import ModelUsage


@dataclass(frozen=True)
class ModelTextDelta:
    """Carry one ordered assistant-text delta.

    Intro:
        Text deltas preserve provider arrival order without exposing provider
        event names or transport frames.

    Examples:
        Build a first delta:
            ```python
            event = ModelTextDelta(delta="Hello", index=0)
            ```

        Append ordered text:
            ```python
            text = "".join(event.delta for event in events)
            ```

    Args:
        delta: Exact incremental assistant text.
        index: Zero-based text-delta arrival index.

    Returns:
        ModelTextDelta: Immutable provider-neutral text event.

    Notes:
        Empty deltas are rejected because they carry no observable progress.
    """

    delta: str
    index: int

    def __post_init__(self) -> None:
        """Validate one assistant-text delta.

        Intro:
            Validation normalizes dynamic string input and protects monotonic
            event indexing before a consumer observes the event.

        Examples:
            Accept a first event:
                ```python
                assert ModelTextDelta("Hi", 0).index == 0
                ```

            Reject an empty event:
                ```python
                try:
                    ModelTextDelta("", 0)
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized text event.

        Returns:
            None: Validates and normalizes the frozen event.

        Notes:
            Whitespace-only text remains meaningful and is preserved.
        """

        delta = str(self.delta)
        if not delta:
            raise ValueError("model text delta must not be empty")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("model text delta index must be a non-negative integer")
        object.__setattr__(self, "delta", delta)


@dataclass(frozen=True)
class ModelReasoningDelta:
    """Carry one ordered private-reasoning display delta.

    Intro:
        Reasoning deltas are separated from assistant output so callers can
        render or suppress them according to their own disclosure policy.

    Examples:
        Build a reasoning delta:
            ```python
            event = ModelReasoningDelta(delta="Checking", index=0)
            ```

        Detect its stream kind:
            ```python
            assert isinstance(event, ModelReasoningDelta)
            ```

    Args:
        delta: Exact incremental reasoning-summary text.
        index: Zero-based reasoning-delta arrival index.

    Returns:
        ModelReasoningDelta: Immutable provider-neutral reasoning event.

    Notes:
        The contract carries displayable reasoning summaries, not hidden model
        state or provider-encrypted reasoning payloads.
    """

    delta: str
    index: int

    def __post_init__(self) -> None:
        """Validate one reasoning delta.

        Intro:
            Validation preserves exact text and requires a non-negative event
            index before the event crosses the public stream boundary.

        Examples:
            Accept a reasoning event:
                ```python
                assert ModelReasoningDelta("Plan", 0).delta == "Plan"
                ```

            Reject a negative index:
                ```python
                try:
                    ModelReasoningDelta("Plan", -1)
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized reasoning event.

        Returns:
            None: Validates and normalizes the frozen event.

        Notes:
            Whitespace-only deltas remain meaningful and are preserved.
        """

        delta = str(self.delta)
        if not delta:
            raise ValueError("model reasoning delta must not be empty")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("model reasoning delta index must be a non-negative integer")
        object.__setattr__(self, "delta", delta)


@dataclass(frozen=True)
class ModelUsageUpdate:
    """Carry one cumulative provider usage snapshot observed during streaming.

    Intro:
        Usage updates expose provider-reported progress without treating a
        cumulative snapshot as an additive token delta or accounting receipt.

    Examples:
        Build a partial update:
            ```python
            event = ModelUsageUpdate(
                usage=ModelUsage.from_provider_usage({"input_tokens": 3}),
                index=0,
            )
            ```

        Inspect cumulative output:
            ```python
            assert event.usage.output_tokens is None
            ```

    Args:
        usage: Normalized cumulative usage known at this stream position.
        index: Zero-based usage-update arrival index.

    Returns:
        ModelUsageUpdate: Immutable provider-neutral usage snapshot event.

    Notes:
        The terminal `ModelStreamCompleted.response.usage` remains authoritative
        for quota reconciliation and metering. Consumers must not sum updates.
    """

    usage: ModelUsage
    index: int

    def __post_init__(self) -> None:
        """Validate one cumulative usage update.

        Intro:
            The event rejects unavailable receipts and invalid indexes because
            neither represents observable usage progress.

        Examples:
            Accept a complete update:
                ```python
                event = ModelUsageUpdate(
                    ModelUsage.from_provider_usage(
                        {"input_tokens": 1, "output_tokens": 1}
                    ),
                    0,
                )
                ```

            Reject unavailable usage:
                ```python
                try:
                    ModelUsageUpdate(ModelUsage.unavailable(), 0)
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized usage-update event.

        Returns:
            None: Validates the frozen event.

        Notes:
            Partial and complete snapshots are both valid provider progress.
        """

        if not isinstance(self.usage, ModelUsage):
            raise TypeError("model usage update requires ModelUsage")
        if self.usage.availability == "unavailable":
            raise ValueError("model usage update requires reported usage")
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("model usage update index must be a non-negative integer")


@dataclass(frozen=True)
class ModelStreamCompleted:
    """Carry the authoritative terminal response for one model stream.

    Intro:
        The terminal event contains the same ordered response and typed usage
        contract returned by non-streaming canonical generation.

    Examples:
        Read terminal text:
            ```python
            assert event.response.text == "Done"
            ```

        Inspect terminal usage:
            ```python
            assert event.response.usage.availability in {"complete", "partial", "unavailable"}
            ```

    Args:
        response: Authoritative completed canonical model response.

    Returns:
        ModelStreamCompleted: Immutable terminal event.

    Notes:
        Exactly one terminal event is emitted on successful completion. Failures
        raise their typed exception through the async iterator instead.
    """

    response: ModelResponse

    def __post_init__(self) -> None:
        """Validate the terminal response value.

        Intro:
            Stream completion accepts only the canonical response type so usage
            and output ordering cannot diverge between streaming and generation.

        Examples:
            Accept a canonical response:
                ```python
                event = ModelStreamCompleted(response)
                ```

            Reject a legacy tuple:
                ```python
                try:
                    ModelStreamCompleted(("text", {}))
                except TypeError:
                    pass
                ```

        Args:
            self: Newly initialized completion event.

        Returns:
            None: Validates the frozen completion event.

        Notes:
            Completion does not duplicate delta content outside the response.
        """

        if not isinstance(self.response, ModelResponse):
            raise TypeError("model stream completion requires ModelResponse")


ModelEvent: TypeAlias = (
    ModelTextDelta | ModelReasoningDelta | ModelUsageUpdate | ModelStreamCompleted
)

__all__ = [
    "ModelEvent",
    "ModelReasoningDelta",
    "ModelStreamCompleted",
    "ModelTextDelta",
    "ModelUsageUpdate",
]

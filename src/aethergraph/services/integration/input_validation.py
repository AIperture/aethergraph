"""Pure accepted-event validation shared by ingress admission callers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from aethergraph.contracts.integration import AcceptedEventContract
from aethergraph.core.schema_validation import SchemaValidationIssue, first_schema_issue


class IngressInputError(RuntimeError):
    """Report a stable accepted-event type or payload contract rejection."""

    def __init__(
        self,
        *,
        code: Literal[
            "integration.event_type_not_accepted",
            "integration.event_payload_invalid",
        ],
        message: str,
        issue: SchemaValidationIssue | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.issue = issue


def validate_accepted_event_input(
    accepted_events: Iterable[AcceptedEventContract],
    *,
    event_type: str,
    payload: Mapping[str, Any],
    payload_path: str = "$.input.payload",
) -> AcceptedEventContract:
    """
    Validate one external Event against an immutable accepted-event collection.

    The function performs no persistence, dispatch, logging, or transport mapping.
    It returns the exact matched contract and raises a stable integration error for
    unknown types or invalid payloads.

    Examples:
        Validate a declared Event:
            ```python
            contract = validate_accepted_event_input(
                (accepted,),
                event_type="simulation.tick",
                payload={"step": 1},
            )
            assert contract.type == "simulation.tick"
            ```

        Reject an unknown Event type:
            ```python
            try:
                validate_accepted_event_input(
                    (),
                    event_type="simulation.tick",
                    payload={},
                )
            except IngressInputError as exc:
                assert exc.code == "integration.event_type_not_accepted"
            ```

    Args:
        accepted_events: Immutable accepted-event contracts published by one build.
        event_type: Exact external Event type submitted by the caller.
        payload: JSON-object payload to validate against the matched contract.
        payload_path: Root path prefixed to any payload validation diagnostic.

    Returns:
        AcceptedEventContract: The exact contract matched by `event_type`.

    Notes:
        The caller owns authorization and maps the stable integration error into its
        transport-specific response without changing the error code.
    """

    contract = next(
        (item for item in accepted_events if item.type == event_type),
        None,
    )
    if contract is None:
        raise IngressInputError(
            code="integration.event_type_not_accepted",
            message=f"Event type {event_type!r} is not accepted by this System.",
        )
    issue = first_schema_issue(payload, contract.payload_schema, path=payload_path)
    if issue is not None:
        raise IngressInputError(
            code="integration.event_payload_invalid",
            message=f"{issue.path}: {issue.message}",
            issue=issue,
        )
    return contract


__all__ = ["IngressInputError", "validate_accepted_event_input"]

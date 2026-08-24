from __future__ import annotations

import pytest

from aethergraph.contracts.integration import AcceptedEventContract
from aethergraph.services.integration import (
    IngressInputError,
    validate_accepted_event_input,
)


def _contract() -> AcceptedEventContract:
    return AcceptedEventContract(
        type="simulation.tick",
        title="Simulation tick",
        payload_schema={
            "type": "object",
            "properties": {"step": {"type": "integer", "minimum": 0}},
            "required": ["step"],
            "additionalProperties": False,
        },
        example_payload={"step": 1},
    )


def test_validate_accepted_event_input_returns_exact_contract() -> None:
    contract = _contract()

    assert (
        validate_accepted_event_input(
            (contract,), event_type="simulation.tick", payload={"step": 2}
        )
        is contract
    )


def test_validate_accepted_event_input_preserves_stable_rejections() -> None:
    with pytest.raises(IngressInputError) as unknown:
        validate_accepted_event_input((_contract(),), event_type="simulation.finished", payload={})
    with pytest.raises(IngressInputError) as invalid:
        validate_accepted_event_input(
            (_contract(),), event_type="simulation.tick", payload={"step": -1}
        )

    assert unknown.value.code == "integration.event_type_not_accepted"
    assert unknown.value.issue is None
    assert invalid.value.code == "integration.event_payload_invalid"
    assert invalid.value.issue is not None
    assert invalid.value.issue.path == "$.input.payload.step"

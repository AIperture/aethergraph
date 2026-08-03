from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from aethergraph.contracts.services.channel import ChoiceResult, FileInteractionResult


def test_choice_result_is_typed_and_immutable() -> None:
    result = ChoiceResult(
        choice="safe",
        choice_label="Safe Mode",
        text="safe",
        matched=True,
    )

    assert result.choice == "safe"
    assert not isinstance(result, dict)
    with pytest.raises(FrozenInstanceError):
        result.choice = "fast"  # type: ignore[misc]


def test_file_interaction_result_uses_immutable_file_collection() -> None:
    file_ref = {"id": "file-1", "name": "report.pdf", "mimetype": "application/pdf"}
    result = FileInteractionResult(text="attached", files=(file_ref,))

    assert result.text == "attached"
    assert result.files == (file_ref,)
    assert isinstance(result.files, tuple)

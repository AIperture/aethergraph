from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .specs import HarnessRunResult, HarnessScenario


@dataclass
class ExactOutputScorer:
    name: str = "exact_output"

    async def score(self, scenario: HarnessScenario, result: HarnessRunResult) -> dict[str, Any]:
        expected = scenario.expected_outputs or {}
        actual = result.outputs or {}
        mismatches: dict[str, dict[str, Any]] = {}
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if actual_value != expected_value:
                mismatches[key] = {"expected": expected_value, "actual": actual_value}
        return {
            "name": self.name,
            "passed": not mismatches,
            "mismatches": mismatches,
        }

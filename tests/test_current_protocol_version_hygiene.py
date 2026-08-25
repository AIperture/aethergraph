"""Keep current protocol identities centralized in AetherGraph contracts."""

from pathlib import Path

from aethergraph.contracts.integration import SEMANTIC_EVENT_PROTOCOL_VERSION


def test_current_semantic_event_version_is_not_hardcoded_in_runtime_tests() -> None:
    tests_root = Path(__file__).parent
    allowed = {"test_integration_contracts.py"}
    violations = [
        str(path.relative_to(tests_root))
        for path in tests_root.rglob("*.py")
        if path != Path(__file__)
        and path.name not in allowed
        and SEMANTIC_EVENT_PROTOCOL_VERSION in path.read_text(encoding="utf-8", errors="ignore")
    ]

    assert violations == []

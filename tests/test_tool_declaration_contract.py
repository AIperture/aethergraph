from __future__ import annotations

from typing import Literal

import pytest

from aethergraph import tool


def test_tool_attaches_versioned_definition_without_runtime_builder() -> None:
    @tool(
        description="Search the current workspace.",
        approval="expensive",
    )
    def search(
        query: str,
        limit: int = 10,
        domain_state: object | None = None,
    ) -> list[str]:
        return [query] * limit

    definition = search.__aether_tool_definition__

    assert definition.api_version == "aethergraph.tool/v1"
    assert definition.name == "search"
    assert definition.approval == "expensive"
    assert definition.inputs == ("query", "limit")
    assert definition.injections == (("domain_state", "domain_state"),)
    assert definition.args_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "additionalProperties": False,
        "required": ["query"],
    }
    assert definition.result_schema == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_tool_supports_literal_schema_and_explicit_public_name() -> None:
    @tool(name="choose", version="2")
    def choose_value(value: Literal["first", "second"]) -> str:
        return value

    definition = choose_value.__aether_tool_definition__

    assert definition.name == "choose"
    assert definition.version == "2"
    assert definition.args_schema["properties"]["value"] == {
        "enum": ["first", "second"],
        "type": "string",
    }


def test_tool_rejects_unknown_approval_policy() -> None:
    with pytest.raises(ValueError, match="approval"):

        @tool(approval="sometimes")
        def invalid() -> None:
            return None

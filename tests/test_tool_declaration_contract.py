from __future__ import annotations

from typing import Literal

import pytest

from aethergraph import ToolDiscoveryMetadata, tool
from aethergraph.core.tools.schema import validate_tool_args


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

    assert definition.api_version == "aethergraph.tool/v5"
    assert definition.exposure == "immediate"
    assert definition.discovery is None
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
    @tool(
        name="choose",
        version="2",
        examples=[{"args": {"value": "first"}, "result": "first"}],
        availability="plan_proposal",
    )
    def choose_value(value: Literal["first", "second"]) -> str:
        return value

    definition = choose_value.__aether_tool_definition__

    assert definition.name == "choose"
    assert definition.version == "2"
    assert definition.examples == ({"args": {"value": "first"}, "result": "first"},)
    assert definition.availability == "plan_proposal"
    assert definition.args_schema["properties"]["value"] == {
        "enum": ["first", "second"],
        "type": "string",
    }


def test_tool_declares_deferred_exposure_and_discovery_metadata() -> None:
    @tool(
        description="Read one document.",
        exposure="deferred",
        discovery=ToolDiscoveryMetadata(
            namespace="docs",
            summary="Read a workspace document.",
            aliases=("open file",),
            tags=("read",),
            effects=("workspace_read",),
        ),
    )
    def read_document(path: str) -> dict[str, str]:
        return {"path": path}

    definition = read_document.__aether_tool_definition__

    assert definition.exposure == "deferred"
    assert definition.discovery is not None
    assert definition.to_dict()["discovery"] == {
        "namespace": "docs",
        "summary": "Read a workspace document.",
        "aliases": ["open file"],
        "tags": ["read"],
        "effects": ["workspace_read"],
    }


def test_deferred_tool_requires_discovery_metadata() -> None:
    with pytest.raises(ValueError, match="require discovery metadata"):

        @tool(exposure="deferred")
        def hidden_without_metadata() -> None:
            return None


def test_deferred_tool_requires_summary_and_bounds_discovery_metadata() -> None:
    with pytest.raises(ValueError, match="non-empty discovery summary"):

        @tool(
            exposure="deferred",
            discovery=ToolDiscoveryMetadata(namespace="docs"),
        )
        def hidden_without_summary() -> None:
            return None

    with pytest.raises(ValueError, match="more than 32"):
        ToolDiscoveryMetadata(
            namespace="docs",
            aliases=tuple(f"alias {index}" for index in range(33)),
        )


def test_tool_normalizes_compact_argument_schema_to_canonical_object() -> None:
    @tool(
        args_schema={
            "query": {"type": "string", "required": True},
            "limit": {"type": "integer", "default": 5},
        }
    )
    def search(**kwargs) -> dict[str, object]:
        return kwargs

    assert search.__aether_tool_definition__.args_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 5},
        },
        "additionalProperties": False,
        "required": ["query"],
    }


def test_tool_preserves_and_validates_local_schema_references() -> None:
    schema = {
        "type": "object",
        "$defs": {
            "operation": {
                "type": "object",
                "properties": {
                    "kind": {"const": "create"},
                    "name": {"type": "string"},
                },
                "required": ["kind", "name"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "operations": {
                "type": "array",
                "items": {"$ref": "#/$defs/operation"},
            }
        },
        "required": ["operations"],
        "additionalProperties": False,
    }

    @tool(args_schema=schema)
    def mutate(**kwargs) -> dict[str, object]:
        return kwargs

    assert mutate.__aether_tool_definition__.args_schema == schema


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (
            {
                "type": "object",
                "properties": {"value": {"$ref": "https://example.com/value"}},
            },
            "only local",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/missing"}},
            },
            "unresolved",
        ),
    ],
)
def test_tool_rejects_unsafe_or_unresolved_schema_references(
    schema: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):

        @tool(args_schema=schema)
        def invalid(**kwargs) -> dict[str, object]:
            return kwargs


def test_tool_rejects_unknown_approval_policy() -> None:
    with pytest.raises(ValueError, match="approval"):

        @tool(approval="sometimes")
        def invalid() -> None:
            return None


def test_tool_rejects_non_object_explicit_result_schema() -> None:
    with pytest.raises(ValueError, match="structured result data payload"):

        @tool(result_schema={"type": "string"})
        def invalid_result_schema() -> str:
            return "invalid"


def test_tool_accepts_arbitrary_object_explicit_result_schema() -> None:
    @tool(result_schema={"type": "object", "additionalProperties": True})
    def arbitrary_result_schema() -> dict[str, object]:
        return {"arbitrary": "value"}

    assert arbitrary_result_schema.__aether_tool_definition__.result_schema == {
        "type": "object",
        "additionalProperties": True,
    }


def test_tool_declares_minimal_semantic_slot_outputs() -> None:
    @tool(slot_outputs=[{"slot_key": "report", "required": True}])
    def build_report() -> dict[str, str]:
        return {"summary": "created"}

    definition = build_report.__aether_tool_definition__

    assert definition.slot_outputs == ({"slot_key": "report", "required": True},)
    assert definition.to_dict()["slot_outputs"] == [{"slot_key": "report", "required": True}]


def test_tool_schema_reports_exact_discriminated_operator_error() -> None:
    schema = {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "op": {"const": "add_step"},
                                "step": {"type": "object"},
                            },
                            "required": ["op", "step"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "op": {"const": "drop_step"},
                                "step_id": {"type": "string"},
                            },
                            "required": ["op", "step_id"],
                            "additionalProperties": False,
                        },
                    ]
                },
            }
        },
        "required": ["operations"],
        "additionalProperties": False,
    }

    issue = validate_tool_args(
        {"operations": [{"op": "change_step", "step_id": "step-1"}]},
        schema,
    )

    assert issue is not None
    assert issue.path == "args.operations[0].op"
    assert issue.validator == "const"
    assert issue.invalid_value == '"change_step"'
    assert issue.expected == ("add_step", "drop_step")
    assert "not an allowed op" in issue.message


def test_tool_schema_selects_matching_discriminator_branch() -> None:
    schema = {
        "type": "object",
        "properties": {
            "operation": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "add_step"},
                            "step": {"type": "object"},
                        },
                        "required": ["op", "step"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "op": {"const": "drop_step"},
                            "step_id": {"type": "string"},
                        },
                        "required": ["op", "step_id"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    issue = validate_tool_args({"operation": {"op": "drop_step"}}, schema)

    assert issue is not None
    assert issue.path == "args.operation"
    assert issue.validator == "required"
    assert "step_id" in issue.message


def test_tool_schema_reports_fields_that_match_prohibited_branch() -> None:
    schema = {
        "type": "object",
        "properties": {
            "mode": {"const": "summary"},
            "queries": {"type": "array"},
            "include": {"type": "array"},
        },
        "required": ["mode"],
        "not": {
            "anyOf": [
                {"required": ["queries"]},
                {"required": ["include"]},
            ]
        },
    }

    issue = validate_tool_args(
        {"mode": "summary", "queries": [], "include": []},
        schema,
    )

    assert issue is not None
    assert issue.path == "args.queries"
    assert issue.validator == "not"
    assert "'queries', 'include'" in issue.message
    assert "Remove prohibited fields" in issue.repair_hint

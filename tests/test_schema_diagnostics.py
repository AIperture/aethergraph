from aethergraph.core.schema_validation import collect_schema_issues, first_schema_issue


def test_collects_independent_siblings_with_stable_order_and_limits():
    schema = {
        "type": "object",
        "properties": {"samples": {"type": "integer"}, "wave": {"minimum": 1}},
        "required": ["format"],
        "additionalProperties": False,
    }
    report = collect_schema_issues({"wave": -1, "samples": "many", "extra": 1}, schema, path="args")
    assert len(report.issues) == 4
    assert report == collect_schema_issues(
        {"extra": 1, "samples": "many", "wave": -1}, schema, path="args"
    )
    limited = collect_schema_issues(
        {"wave": -1, "samples": "many", "extra": 1}, schema, max_issues=2
    )
    assert len(limited.issues) == 2
    assert limited.omitted_count == 2
    assert first_schema_issue({}, {"type": "object"}) is None


def test_selected_union_branch_reports_all_its_errors_only():
    branches = [
        {
            "type": "object",
            "properties": {
                "kind": {"const": kind},
                "x": {"type": "integer"},
                "y": {"type": "boolean"},
            },
            "required": ["kind", "x", "y"],
        }
        for kind in ("a", "b")
    ]
    schema = {"type": "array", "items": {"oneOf": branches}}
    report = collect_schema_issues([{"kind": "a", "x": "wrong", "y": 2}], schema)
    assert [issue.path for issue in report.issues] == ["$[0].x", "$[0].y"]
    unknown = collect_schema_issues([{"kind": "c"}], schema)
    assert len(unknown.issues) == 1
    assert unknown.issues[0].expected == ("a", "b")


def test_invalid_parent_blocks_children_and_messages_are_bounded():
    report = collect_schema_issues(
        "x" * 10000,
        {
            "type": "object",
            "properties": {"child": {"type": "integer"}},
        },
    )
    assert len(report.issues) == 1
    assert len(report.issues[0].message) <= 500
    assert report.issues[0].validator == "type"


def test_array_root_and_standalone_credentials():
    from aethergraph.server.security import sanitize_text

    report = collect_schema_issues(
        [{"extra": 1}],
        {"type": "array", "items": {"type": "object", "additionalProperties": False}},
    )
    assert report.issues[0].path == "$[0].extra"
    assert "sk-syntheticsecret123" not in sanitize_text("token sk-syntheticsecret123")

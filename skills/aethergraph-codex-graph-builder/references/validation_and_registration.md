# Validation And Registration

Source of truth:
- `src/aethergraph/core/graph/graphify_validation.py`
- `src/aethergraph/services/registry/registration_service.py`
- `tests/test_registry_api.py`

Use this loop every time.

## Loop

1. Generate or update the workflow module.
2. Validate the source using current graphify validation rules.
3. Import the module and call `<graph_name>.build()`.
4. Register it as an app.
5. If any step fails, patch the module and retry until all three pass.

## Registration Checks

- Registration uses `validate_graphify_source(...)` before executing source.
- Validation failures commonly include:
- `missing_decorator_kw`
- `graphify_async_def`
- `graphify_control_flow_non_deterministic`
- `graphify_unsupported_condition_expr`
- `tool_nested_tool_call_disallowed`
- Warnings can still matter; fix them when they point to broken graph intent.

## Build Checks

After validation passes, force graph construction:

```python
from workflow import my_graph

task_graph = my_graph.build()
print(task_graph.spec)
```

Use `.build()` to catch graph wiring issues that static validation does not catch.

## App Registration Pattern

Prefer `as_app` directly in the decorator:

```python
@graphify(
    name="report_workflow",
    inputs=["request_text"],
    outputs=["final_artifact_id"],
    as_app={
        "id": "report-workflow",
        "name": "Report Workflow",
        "description": "Builds a report and returns the saved artifact id.",
    },
)
def report_workflow(request_text: str):
    ...
```

For general AG UI workflows, also include a typed `input_schema` with sensible defaults so users do not have to hand-author form inputs:

```python
@graphify(
    name="report_workflow",
    inputs={
        "request_text": "Summarize the latest artifact set.",
        "max_items": 25,
        "mode": "fast",
    },
    outputs=["final_artifact_id"],
    as_app={
        "id": "report-workflow",
        "name": "Report Workflow",
        "description": "Builds a report and saves the final result as an artifact.",
        "input_schema": [
            {
                "name": "request_text",
                "label": "Request",
                "widget": "textarea",
                "description": "High-level workflow request.",
                "default": "Summarize the latest artifact set.",
            },
            {
                "name": "max_items",
                "label": "Max Items",
                "widget": "number",
                "description": "Maximum number of items to process.",
                "default": 25,
            },
            {
                "name": "mode",
                "label": "Mode",
                "widget": "text",
                "description": "Execution mode for the workflow.",
                "default": "fast",
            },
        ],
    },
)
def report_workflow(request_text: str, max_items: int, mode: str):
    ...
```

Important:
- The canonical `as_app.input_schema` shape is a `list[dict]`, not a dict keyed by input name.
- Each item should include `"name": "<input_name>"`.
- Older dict-shaped metadata can be tolerated by some runtime paths for compatibility, but new examples should always use the list form.

Practical rules:
- Keep defaults realistic so the app is runnable immediately from UI.
- Use number widgets for numeric inputs, textarea for long prompts, text for short strings, switch for booleans, and json only when the user truly needs raw structured input.
- Prefer graph outputs like `final_artifact_id` or `result_artifact_uri` so AG UI users can inspect results in the artifact store.

If you need explicit registration through the service facade, keep app config minimal and aligned with the graph name.

## Practical Rule

- Do not stop after writing code.
- Finish only after validation, `.build()`, and app registration succeed.

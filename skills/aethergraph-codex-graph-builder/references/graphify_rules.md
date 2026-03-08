# Graphify Rules

Source of truth:
- `src/aethergraph/core/graph/graphify.py`
- `src/aethergraph/core/graph/graphify_validation.py`
- `tests/test_graphify_fail_fast_validation.py`

Use this when generating or repairing `@graphify` code.

## Required Rules

- `@graphify` must include `name=`, `inputs=`, and `outputs=`.
- `@graphify` must decorate a synchronous `def`, not `async def`.
- Keep orchestration in `@graphify`; tool bodies should not call other tools.
- Do not `await` inside `@graphify`.
- Avoid unsupported control flow in `@graphify`: `for`, `while`, `try`, `match`, and complex `if` shapes will fail validation.
- Prefer declarative `_condition={...}` on tool calls for branching.
- Return keys must match declared outputs.

## IO Guidance

- When a step produces large structured data, save it as an artifact and return its id or uri as a string output.
- Use function annotations so registry IO typing is as accurate as possible.

## Minimal Pattern

```python
from aethergraph import NodeContext, graphify, tool


@tool(name="step_one", outputs=["artifact_id"])
async def step_one(source_text: str, *, context: NodeContext) -> dict[str, str]:
    artifact = await context.artifacts().save_text(
        source_text.upper(),
        kind="text",
        name="step-one-output.txt",
    )
    return {"artifact_id": str(artifact.artifact_id)}


@tool(name="step_two", outputs=["result"])
async def step_two(artifact_id: str, *, context: NodeContext) -> dict[str, str]:
    text = await context.artifacts().load_text_by_id(artifact_id)
    return {"result": text[:200]}


@graphify(
    name="example_workflow",
    inputs=["source_text"],
    outputs=["result"],
    as_app={"id": "example-workflow"},
)
def example_workflow(source_text: str):
    prepared = step_one(source_text=source_text)
    final = step_two(artifact_id=prepared.artifact_id)
    return {"result": final.result}
```

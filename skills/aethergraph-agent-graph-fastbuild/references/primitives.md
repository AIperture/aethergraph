# Primitives

This file captures source-of-truth usage for `@tool`, `@graph_fn`, and `@graphify`.

## Imports

```python
from aethergraph import NodeContext, graph_fn, graphify, tool
```

## `@tool`

Source: `src/aethergraph/core/tools/toolkit.py`

Signature:

```python
tool(outputs: list[str], inputs: list[str] | None = None, *, name: str | None = None, version: str = "0.1.0")
```

Rules:
- Always declare `outputs=[...]`.
- Returned dict must include all declared outputs.
- In graph-building mode, tool calls return `NodeHandle` values.
- Outside graph mode, tool runs immediately.

Example:

```python
@tool(name="normalize_text", outputs=["text"])
def normalize_text(raw: str) -> dict:
    return {"text": raw.strip().lower()}
```

## `@graph_fn`

Source: `src/aethergraph/core/graph/graph_fn.py`

Signature:

```python
graph_fn(
    name: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    version: str = "0.1.0",
    *,
    entrypoint: bool = False,
    flow_id: str | None = None,
    tags: list[str] | None = None,
    as_agent: dict[str, Any] | None = None,
    as_app: dict[str, Any] | None = None,
    description: str | None = None,
)
```

Use when:
- You want fastest path from user intent to async runnable behavior.
- You need loops, branching, or immediate orchestration.

Example:

```python
@graph_fn(name="quick_echo", inputs=["text"], outputs=["result"])
async def quick_echo(text: str, *, context: NodeContext) -> dict:
    await context.channel().send_text("Running quick_echo")
    return {"result": text}
```

## `@graphify`

Source: `src/aethergraph/core/graph/graphify.py`

Signature:

```python
graphify(
    name="default_graph",
    inputs=(),
    outputs=None,
    version="0.1.0",
    *,
    entrypoint: bool = False,
    flow_id: str | None = None,
    tags: list[str] | None = None,
    as_agent: dict[str, Any] | None = None,
    as_app: dict[str, Any] | None = None,
    description: str | None = None,
)
```

Use when:
- You want explicit tool-node composition.
- You want steps to be independently reusable/testable.

Example:

```python
@tool(name="token_count", outputs=["count"])
def token_count(text: str) -> dict:
    return {"count": len(text.split())}

@graphify(name="count_graph", inputs=["text"], outputs=["count"])
def count_graph(text):
    c = token_count(text=text)
    return {"count": c.count}
```

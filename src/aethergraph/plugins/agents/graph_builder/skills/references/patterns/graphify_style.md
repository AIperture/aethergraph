---
id: ag-graph-builder-graphify-style
title: Graphy Style
---

# Graphify + Tool Style Guide

## Core graph rules

- `@graphify` function must be synchronous (`def`), because it declares the DAG.
- Each `@tool` invocation inside `@graphify` creates a node.
- Keep orchestration in `@graphify`; do not call one `@tool` from inside another `@tool`.
- Use data refs (`node.output_key`) for real data edges.
- Use `_after=` only when you need ordering without passing data.
- Do not pass `context` from graph code. `context` is injected at runtime for tool functions.
- Bare decorators are invalid. Always include explicit attributes.
- Graph inputs should be typed via function annotations so runtime/UI can infer schemas.

## Tool rules

- Prefer explicit tool `name=...` for stable identity and checkpoints.
- Each tool must declare `outputs=[...]`.
- Tool return dict keys must include every declared output key.
- If a tool uses `context.channel("ui:run")` or `context.artifacts()`, define it as:
  - `async def tool_name(..., *, context: NodeContext) -> dict`

## Example: explicit DAG with `_after`

```python
from aethergraph import NodeContext, graphify, tool

@tool(name="read_input", outputs=["clean_text"])
def read_input(text: str) -> dict:
    return {"clean_text": text.strip()}

@tool(name="announce_start", outputs=["ok"])
async def announce_start(step: str, *, context: NodeContext) -> dict:
    await context.channel("ui:run").send_text(f"starting {step}")
    return {"ok": True}

@tool(name="expensive_step", outputs=["result"])
def expensive_step(clean_text: str) -> dict:
    return {"result": clean_text.upper()}

@graphify(name="text_pipeline", inputs=["text"], outputs=["result"])
def text_pipeline(text: str):
    start = announce_start(step="expensive_step")
    parsed = read_input(text=text)
    final = expensive_step(clean_text=parsed.clean_text, _after=[start])
    return {"result": final.result}
```

## Pitfalls to avoid

- Do not write `async def` for the `@graphify` function.
- Do not `await` tool calls inside `@graphify`.
- Do not add fake data inputs just to force ordering; use `_after`.
- Do not emit:
  - `@tool` (bare)
  - `@graphify` (bare)
  - tuple returns from graph (`return a, b`)

## Invalid vs valid decorator usage

```python
# INVALID
@tool
async def create_video(...):
    return "file.mp4"

@graphify
def workflow(...):
    ...
    return final_json, video_path
```

```python
# VALID
@tool(name="create_video", outputs=["video_output"])
async def create_video(..., *, context: NodeContext) -> dict:
    return {"video_output": "file.mp4"}

@graphify(name="workflow", inputs=["args"], outputs=["final_lens_json", "video_output"])
def workflow(args: dict):
    ...
    return {"final_lens_json": final.final_lens_json, "video_output": video.video_output}
```

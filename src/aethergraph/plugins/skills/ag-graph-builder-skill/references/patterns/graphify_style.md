---
id: ag-graph-builder-graphiy-style
title: Graphy Style
---

# Graphify + Tool Style Guide

## Tool rules

- Prefer explicit tool `name=...` for stable checkpoints and readability.
- Each tool must declare `outputs=[...]`.
- Tools return a dict with keys matching outputs.

## Graphify rules

- `@graphify(name=..., inputs=[...], outputs=[...])`
- Compose tools explicitly. List comprehensions are allowed.
- Prefer fan-out/fan-in patterns for map-reduce like flows.

## Concrete example (fan-out / fan-in)

```python
from aethergraph import tool, graphify

@tool(name="pick", outputs=["result"])
async def pick(items: list[int], index: int) -> dict:
    return {"result": items[index]}

@tool(name="work", outputs=["out"])
async def work(x: int) -> dict:
    print(f"Working on {x}...")
    return {"out": x * 2}

@tool(name="reduce_sum", outputs=["sum"])
async def reduce_sum(xs: list[int]) -> dict:
    return {"sum": sum(xs)}

@graphify(name="map_reduce", inputs=["vals"], outputs=["sum"])
def map_reduce(vals):
    results = [pick(items=vals, index=i) for i in range(len(vals))]  # fan-out (pick)
    outs = [work(x=v.result) for v in results]                       # fan-out (work)
    total = reduce_sum(xs=[o.out for o in outs])                     # fan-in (reduce)
    return {"sum": total.sum}
```

## Notes

- For large fan-outs, consider chunking or batching tools if the runtime supports it.
- When a `work` step is expensive, add checkpointing around it (see checkpointing.md).

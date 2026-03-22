# Examples

These examples prioritize speed-to-working-code.

## Example 1: Fast `@graph_fn` Agent

```python
from aethergraph import NodeContext, graph_fn

@graph_fn(
    name="draft_reply_agent",
    inputs=["message"],
    outputs=["response"],
    as_agent={
        "id": "draft-reply-agent",
        "title": "Draft Reply Agent",
        "description": "Creates a short draft reply",
        "mode": "chat_v1",
    },
)
async def draft_reply_agent(message: str, *, context: NodeContext) -> dict:
    context.logger().info("draft_reply_agent start")
    await context.channel().send_text("Composing reply...")
    return {"response": f"Draft: {message}"}
```

## Example 2: `@tool + @graphify` Workflow

```python
from aethergraph import graphify, tool

@tool(name="clean_query", outputs=["query"])
def clean_query(text: str) -> dict:
    return {"query": " ".join(text.strip().split())}

@tool(name="count_words", outputs=["count"])
def count_words(text: str) -> dict:
    return {"count": len(text.split())}

@graphify(name="query_stats", inputs=["text"], outputs=["query", "count"])
def query_stats(text):
    q = clean_query(text=text)
    c = count_words(text=q.query)
    return {"query": q.query, "count": c.count}
```

## Example 3: Run Locally

```python
import asyncio
from aethergraph.runner import run_async

async def main():
    out = await run_async(query_stats, inputs={"text": "  hello   world  "})
    print(out)

asyncio.run(main())
```

## Example 4: Start Server For UI/API

```python
from aethergraph import start_server

url, handle = start_server(
    workspace="./aethergraph_workspace",
    port=0,
    load_paths=["./my_graphs.py"],
    project_root=".",
    return_handle=True,
)
print("AetherGraph server:", url)
handle.block()
```

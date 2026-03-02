---
id: aethergraph-graph-agent-app-creator
title: Agent Builder
description: Create AetherGraph graphs, graph_fn agents, graphify flows, and apps from user intent using @graph_fn, @graphify, @tool, as_agent/as_app metadata, and NodeContext runtime APIs (channel, artifacts, memory, skills, run orchestration).
---

# AetherGraph Graph/Agent/App Creator

## When to use
Use this skill when the user asks to:
- create or update an AetherGraph `@graph_fn`
- create or update a static `@graphify` DAG with `@tool` nodes
- register a graph as an agent via `as_agent={...}`
- register a graph as an app via `as_app={...}`
- wire runtime behavior through `NodeContext` for:
  - `channel`
  - `artifacts`
  - `memory`
  - `skills`
  - spawning/running child graphs

## Hard constraints
- Use only AetherGraph primitives: `@graph_fn`, `@graphify`, `@tool`, `NodeContext`.
- If runtime services are needed, function signature must include `*, context: NodeContext`.
- Do not invent new context APIs. Stay within NodeContext methods:
  - `context.channel(...)`
  - `context.memory()`
  - `context.skills()`
  - `context.spawn_run(...)`
  - `context.run_and_wait(...)`
  - `context.wait_run(...)`
  - `context.cancel_run(...)`
- Prefer explicit `inputs=[...]` and `outputs=[...]` in decorators.

## Intent to implementation mapping
1. If user wants a single async workflow entrypoint:
   use `@graph_fn`.
2. If user wants explicit DAG composition and reusable node-level tools:
   use `@tool` + `@graphify`.
3. If user wants an agent visible/launchable as an agent:
   add `as_agent={...}` to `@graph_fn` or `@graphify`.
4. If user wants an app visible/launchable as an app:
   add `as_app={...}` to `@graph_fn` or `@graphify`.

## Metadata rules

### graph_fn / graphify common metadata
- `name`: stable graph identifier
- `version`: default `"0.1.0"` unless user specifies
- `entrypoint`, `flow_id`, `tags`, `description`: optional, include when meaningful

### as_agent metadata (recommended minimum)
```python
as_agent={
    "id": "agent_id",
    "title": "Human Title",
    "description": "What this agent does",
    "mode": "chat_v1",
}
```

Notes:
- `chat_v1` expects inputs compatible with chat-agent shape:
  `["message", "files", "context_refs", "session_id", "user_meta"]`.

### as_app metadata (recommended minimum)
```python
as_app={
    "id": "app_id",
    "name": "Human App Name",
    "description": "What this app does",
    "mode": "no_input_v1",
}
```

## NodeContext usage patterns

### Channel
```python
await context.channel().send_text("message")
text = await context.channel().ask_text("question")
```

### Artifacts
```python
art = await context.artifacts().save_text("hello", filename="out.txt")
data = await context.artifacts().search(query="hello", top_k=5)
```

### Memory
```python
await context.memory().record(kind="note", data={"x": 1}, tags=["demo"])
hits = await context.memory().search(query="note", top_k=5)
```

### Skills
```python
skill = context.skills().get("skill_id")
```

### Run orchestration
```python
run_id = await context.spawn_run("child_graph", inputs={"x": 1})
record = await context.wait_run(run_id)
child_run_id, outputs, has_waits, conts = await context.run_and_wait(
    "child_graph", inputs={"x": 1}
)
await context.cancel_run(run_id)
```

## Generation procedure
1. Parse user intent into:
   - graph type (`graph_fn` or `graphify`)
   - runtime needs (channel/artifacts/memory/skills/child runs)
   - registration target (plain graph vs agent vs app)
2. Choose decorator and define explicit inputs/outputs.
3. Add `context: NodeContext` if any runtime need exists.
4. Implement minimal happy-path logic first.
5. Add metadata (`as_agent` / `as_app`) only when requested.
6. Return structured outputs as dict matching declared outputs.
7. Provide runnable `if __name__ == "__main__"` snippet when user asks for executable example.

## Templates

### Template A: graph_fn
```python
from aethergraph import NodeContext, graph_fn

@graph_fn(
    name="my_graph",
    inputs=["input_text"],
    outputs=["result"],
    tags=["example"],
)
async def my_graph(input_text: str, *, context: NodeContext) -> dict:
    await context.channel().send_text(f"Input: {input_text}")
    await context.memory().record(kind="input", data={"text": input_text})
    return {"result": input_text}
```

### Template B: graph_fn as agent
```python
from aethergraph import NodeContext, graph_fn

@graph_fn(
    name="support_agent_graph",
    inputs=["message", "files", "context_refs", "session_id", "user_meta"],
    outputs=["response"],
    as_agent={
        "id": "support-agent",
        "title": "Support Agent",
        "description": "Handles support chat",
        "mode": "chat_v1",
    },
)
async def support_agent(
    message: str,
    files: list,
    context_refs: list,
    session_id: str,
    user_meta: dict,
    *,
    context: NodeContext,
) -> dict:
    await context.channel().send_text("Processing support request...")
    return {"response": f"Received: {message}"}
```

### Template C: tool + graphify
```python
from aethergraph import graphify, tool

@tool(name="normalize_text", outputs=["text"])
def normalize_text(raw: str) -> dict:
    return {"text": raw.strip().lower()}

@graphify(
    name="normalize_graph",
    inputs=["raw"],
    outputs=["text"],
)
def normalize_graph(raw):
    n = normalize_text(raw=raw)
    return {"text": n.text}
```


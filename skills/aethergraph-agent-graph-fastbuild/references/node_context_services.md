# NodeContext Services

Source: `src/aethergraph/core/runtime/node_context.py`

Focus services for fast agent/graph creation.

## Core Accessors

- `context.channel(channel_key: str | None = None) -> ChannelSession`
- `context.artifacts() -> ArtifactFacade`
- `context.memory() -> MemoryFacade`
- `context.indices() -> ScopedIndices`
- `context.kb() -> NodeKB`  (knowledge)
- `context.kv()`
- `context.llm(...) -> LLMClientProtocol`
- `context.logger()`
- `context.runner() -> RunFacade`
- `context.triggers() -> TriggerFacade`
- `context.viz() -> VizFacade`
- `context.skills() -> SkillRegistry`
- `context.scope` (dataclass field; current scope object)

## Quick Usage Snippets

Channel:

```python
await context.channel().send_text("Started")
reply = await context.channel().ask_text("Need approval?")
```

Artifacts:

```python
saved = await context.artifacts().save_text("hello", filename="hello.txt")
```

Memory:

```python
await context.memory().record(kind="note", data={"k": "v"}, tags=["demo"])
```

Indices / Knowledge:

```python
idx = context.indices()
kb = context.kb()
```

KV:

```python
kv = context.kv()
```

LLM:

```python
llm = context.llm()
text = await llm.complete("Summarize this")
```

Logger:

```python
context.logger().info("node step complete")
```

Runner:

```python
run_id = await context.runner().spawn_run("child_graph", inputs={"x": 1})
record = await context.runner().wait_run(run_id)
```

Triggers / Viz / Skills:

```python
tr = context.triggers()
vz = context.viz()
sk = context.skills()
```

## Practical Rules

- Add `*, context: NodeContext` only when using runtime services.
- Keep first version simple: one service at a time.
- Prefer `context.runner()` APIs over deprecated `context.spawn_run()/run_and_wait()/wait_run()` wrappers.
